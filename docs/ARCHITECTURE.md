# Архитектура Interviewer's Little Helper

Документ для разработчика: как устроена система, какие у неё контракты и где её
расширять. Пользовательская инструкция — в [README.md](../README.md), исходный
план с бюджетами памяти — в [PLAN.md](../PLAN.md), план тестирования — в
[TESTING.md](TESTING.md).

## 1. Обзор

Локальное приложение для качественных интервью по Zoom. Два аудиоканала
(микрофон = интервьюер, виртуальный кабель = респондент) транскрибируются
Whisper'ом чанками по паузам; раз в 4 минуты локальная LLM сравнивает новую
часть разговора с гайдом и обновляет чеклист покрытия тем + рекомендации.
UI — веб-страница 600×400 в always-on-top окне pywebview.

Ключевые свойства:

- **Всё локально.** Сервер слушает только `127.0.0.1`, фронтенд не делает ни
  одного внешнего запроса, телеметрии нет. Сеть нужна один раз — скачать веса.
- **Разделение говорящих на уровне ОС**, не диаризацией: канал = говорящий.
- **Чанковая, не потоковая транскрипция** — цикл анализа в 4 минуты позволяет
  жертвовать латентностью ради простоты и стабильности.
- **Монотонное running-состояние покрытия** + периодические сверки против
  дрейфа (см. §4).

## 2. Компоненты

```
app/
  config.py            Вся конфигурация (dataclasses + config.json поверх)
  domain.py            Speaker, AudioChunk, Segment, fmt_ts
  logging_setup.py     Консоль + logs/app.log (ротация)
  main.py              Вход: pywebview + uvicorn-поток; --browser; проверка порта
  audio/
    capture.py         list_input_devices, RingBuffer, ChannelCapture (уровень, watchdog-метка)
    vad.py             SpeechDetector: Silero (onnx из faster-whisper) / Energy (фолбэк)
    chunker.py         ChunkAssembler (чистая нарезка по паузам) + ChunkerThread
  asr/
    base.py            ASRBackend.load()/transcribe() -> ASRResult
    faster_backend.py  CTranslate2: cuda->cpu фолбэк, NVIDIA DLL из pip-пакетов
    mlx_backend.py     mlx-whisper (Metal)
    factory.py         Выбор бэкенда по платформе/конфигу
    worker.py          Единственный ASR-поток: очередь чанков -> сегменты
  transcript/store.py  Потокобезопасный список сегментов + дельта-курсоры + подписчики
  llm/ollama_client.py Health-check, chat со structured output, ретраи, деградации
  guide/
    schemas.py         Guide/Section/Topic (pydantic)
    parser.py          Текст -> структура через LLM; id назначаются детерминированно
    library.py         guides/*.json: сохранение/список/загрузка/удаление
  coverage/
    schemas.py         Статусы, TopicState, Recommendation, Finding, JSON-схемы ответов
    prompts.py         Системные промпты (live/final), блоки состояния и дельты
    engine.py          Планировщик, режимы delta/reconcile/final, слияние, ручные метки
  storage/
    session_store.py   Папка сессии: append-по-ходу, атомарные перезаписи, флаги
    report.py          Сборка report.md + промпт резюме
  server/
    hub.py             WebSocket-хаб (broadcast + threadsafe-мост)
    controller.py      Оркестратор: запуск/остановка, монитор уровней, watchdog
    app.py             FastAPI: REST + WS + статика
  web/                 index.html / app.css / app.js — без сборки и CDN
tools/simulate.py      Прогон движка на текстовом транскрипте (без аудио)
```

## 3. Поток данных и модель конкурентности

```
PortAudio callback (×2) ──▶ RingBuffer (×2) ──▶ ChunkerThread (×2) ──▶ Queue ──▶ ASRWorker (×1)
     (только копирование)     (drop-oldest)       (VAD, нарезка)      (cap 60)     (одна модель)
                                                                                      │ сегмент
                            SessionStore.append_segment ◀── listeners ── TranscriptStore
                            WsHub.broadcast_threadsafe  ◀──┘                  │ delta_since(cursor)
                                                                              ▼
   asyncio (uvicorn loop): scheduler_task ── CoverageEngine ──▶ OllamaClient ──▶ слияние ──▶ WS/диск
                           watchdog_task (уровни + живость устройств)
                           monitor_task (уровни до старта сессии)
```

Потоки ОС: 2 callback'а PortAudio, 2 чанкера, 1 ASR-воркер. Асинхронщина —
в event loop uvicorn: планировщик анализа, watchdog, монитор, HTTP к Ollama.

Мосты между мирами:

- поток → loop: `WsHub.broadcast_threadsafe` (`run_coroutine_threadsafe`);
- loop → поток: `asyncio.to_thread` для блокирующих операций (join, open/stop
  устройств);
- события остановки: `threading.Event` (`thread_stop`) для потоков,
  `asyncio.Event` (`engine_stop`, `manual_event`) для задач.

Правила:

- В audio-callback — только копирование в буфер и обновление пикового уровня.
- **Один экземпляр Whisper** на оба канала (экономия VRAM); порядок сегментов
  восстанавливается сортировкой по `t0` на чтении.
- Очередь ASR ограничена 60 чанками: живой, но отстающий бэкенд жалуется в лог,
  мёртвый — не съедает память.
- Остановка строго упорядочена (`controller._stop_audio_asr`): планировщик →
  захват → чанкеры (flush хвоста) → ASR дорабатывает очередь → финальная
  сверка → отчёт → финальная запись.

Таймстампы: `t0/t1` сегмента — секунды от старта захвата (позиция в потоке
16 kHz), не wall-clock; wall-clock фиксируется в `created_at` и `meta.json`.

## 4. Движок покрытия

Состояние: `topic_id -> {status, confidence, evidence, manual, last_update_iteration}`,
статусы `not_covered < partial < covered`.

Три режима анализа (`engine._run_analysis(segments, mode)`):

| Режим | Когда | Вход | Особенности |
|---|---|---|---|
| `delta` | каждый цикл (240 c / кнопка) | сегменты с прошлого анализа | рекомендации + пробы |
| `reconcile` | каждый N-й цикл (`reconcile_every`) | окно с прошлой **сверки** | ловит темы, пропущенные в отдельных дельтах, не раздувая контекст |
| `final` | на «Стоп» | весь транскрипт, окнами ≤ `max_delta_chars` | без рекомендаций; собирает `findings` для отчёта |

Правила слияния (`_apply_response`):

1. **Монотонность:** LLM может только повышать статус. Понижения игнорируются —
   защита от «мигания» между итерациями.
2. **Ручная метка сильнее LLM:** `manual=True` — обновления LLM по теме
   игнорируются. Сам исследователь может ставить любой статус, в т.ч. ниже
   текущего (`set_manual_status`); `status=None` снимает метку.
3. Неизвестные `topic_id` отбрасываются с логом; повреждённые элементы ответа
   валидируются поштучно (битые выбрасываются, остальные живут).
4. Рекомендации `coverage_gap` хранятся по одной на тему (последняя версия),
   исчезают при `covered`; скрытые (`dismiss`) темы не показываются до ручного
   изменения статуса.
5. Пробы (`probe`) живут ровно один цикл и заменяются целиком.
6. `findings` копятся только в `final` (дедупликация по точному совпадению).
7. **Курсоры двигаются только при успехе** — упавший цикл ничего не теряет,
   его дельта уйдёт в следующий.

## 5. Контракт с LLM

- Транспорт: Ollama `/api/chat`, `stream:false`, `keep_alive:-1`,
  `options.num_ctx` задаётся явно.
- **Structured output:** `format=<JSON-схема>` (без `$ref`). Деградации для
  старых серверов: схема → `format:"json"`; параметр `think:false` (qwen3)
  убирается, если сервер его не принимает. Поверх — `robust_json_parse`
  (срезание `<think>`, ограждений, вырезание объекта) + один повторный запрос
  с текстом ошибки.
- **Кэш префикса:** гайд лежит в системном промпте и не меняется всю сессию —
  Ollama переиспользует KV-кэш, prefill повторных циклов почти бесплатен.
  Поэтому порядок «система+гайд → состояние → дельта» менять нельзя.
- Вызовы: парсинг гайда (1 раз), live-цикл, final-окна, резюме отчёта.
  Все обмены логируются в `analysis_log.jsonl` (ответ обрезается до 20 КБ).

## 6. REST API (все — `127.0.0.1:8756`)

| Метод и путь | Что делает |
|---|---|
| `GET /`, `/app.css`, `/app.js` | статика UI |
| `GET /api/state` | полный снапшот (для загрузки/переподключения UI) |
| `GET /api/devices` | входные аудиоустройства; при сбое `{devices:[], error}` без 500 |
| `GET /api/llm/status` | `{ok, server_up, model_found, version, error}` |
| `POST /api/guide/parse` | `{text}` → структура гайда (LLM) |
| `GET /api/guides` / `POST /api/guides` | библиотека: список / сохранить `{guide, source_text}` |
| `GET /api/guides/{id}` / `DELETE .../{id}` | загрузить / удалить |
| `POST /api/monitor/start` | `{mic_index?, system_index?}` — уровни до старта сессии |
| `POST /api/monitor/stop` | остановить монитор |
| `POST /api/session/start` | `{mic_index, system_index, guide, duration_min?, asr_vocabulary?}` |
| `POST /api/session/stop` | остановка: хвост ASR → финальная сверка → отчёт |
| `POST /api/session/analyze` | ручной запуск цикла |
| `POST /api/session/flag` | `{note?}` — отметить момент |
| `POST /api/topics/status` | `{topic_id, status\|null}` — ручная метка |
| `POST /api/recommendations/dismiss` | `{topic_id}` — скрыть подсказку |
| `WS /ws` | события; при подключении сразу приходит `snapshot` |

Ошибки уровня приложения — `409 {detail: "человекочитаемый текст"}`
(`ControllerError`); текст показывается в UI как есть.

## 7. События WebSocket

Конверт: `{"type": ..., "payload": {...}}`.

| type | payload (главное) | Источник |
|---|---|---|
| `snapshot` | как `GET /api/state` | при подключении WS |
| `segment` | сегмент транскрипта | ASR-воркер (threadsafe) |
| `coverage` | `{topics, counts, iteration}` | после каждого анализа / ручной метки |
| `recommendations` | `{items, probes, iteration}` | после live-анализа / dismiss |
| `analysis` | `{phase: started\|done\|failed, mode, ...}` | движок |
| `timer` | `{next_analysis_at, interval_s}` | планировщик; клиент сам ведёт отсчёт |
| `status` | `{state?, message, asr?}` | контроллер/движок |
| `error` | `{message}` | любой сбой, требующий внимания |
| `session` | `{state, session_id, started_at, duration_min}` | старт/стоп |
| `levels` | `{monitor, interviewer?, respondent?}` ~8 Гц | монитор/watchdog |
| `channel` | `{speaker, alive, device}` | watchdog при смене живости |
| `flag` | `{t, note, ts}` | добавление флага |

## 8. Данные на диске

`sessions/<YYYY-MM-DD_HH-MM-SS>/`:

- `transcript.jsonl` — append+flush после каждого сегмента (переживает сбой);
- `coverage_state.json`, `meta.json`, `guide.json` — атомарная перезапись
  (tmp + rename);
- `recommendations.jsonl`, `analysis_log.jsonl`, `flags.jsonl` — append-логи;
- `transcript.md` — перегенерируется каждый цикл (флаги вшиты хронологически);
- `report.md` — после «Стоп»: резюме (LLM), темы со статусами и findings,
  «не раскрыто», флаги.

`guides/<timestamp>-<slug>.json` — `{saved_at, source_text, guide}`; `source_text`
хранится, чтобы гайд можно было править и перепарсивать. Имена файлов
проверяются на path traversal.

## 9. Конфигурация

`config.json` в корне (см. `config.example.json`) накладывается на дефолты из
`app/config.py`; неизвестные ключи игнорируются с предупреждением.

| Ключ | Дефолт | Смысл |
|---|---|---|
| `server.host/port` | `127.0.0.1:8756` | менять host не рекомендуется (приватность) |
| `analysis.interval_s` | 240 | период цикла |
| `analysis.max_recommendations` | 6 | лимит подсказок в выдаче |
| `analysis.max_probes` | 2 | проб за цикл; 0 = выключить |
| `analysis.reconcile_every` | 4 | каждый N-й цикл — сверка; 0 = выкл |
| `analysis.final_sweep` / `report` | true | финальная сверка / отчёт на «Стоп» |
| `analysis.default_duration_min` | 60 | дефолт длительности для темпа |
| `audio.max_chunk_s / min_pause_s / min_speech_s / pad_s` | 25 / 0.7 / 0.3 / 0.2 | нарезка |
| `audio.vad` | auto | auto → Silero, фолбэк Energy |
| `audio.watchdog_silence_s` | 12 | нет сэмплов дольше — канал «мёртв» |
| `asr.backend` | auto | auto/mlx/faster/faster-cpu |
| `asr.model / mlx_model` | large-v3-turbo | модель Whisper |
| `asr.compute_type` | auto | cuda→float16, cpu→int8; `int8_float16` для экономии |
| `asr.vocabulary` | "" | термины проекта → initial_prompt |
| `llm.model` | qwen3:8b | любой тег Ollama |
| `llm.num_ctx` | 8192 | обязательно явно: дефолт Ollama меньше |
| `llm.max_delta_chars` | 12000 | потолок фрагмента транскрипта в промпте |
| `storage.sessions_dir / guides_dir` | sessions / guides | пути |

## 10. Точки расширения

- **Новый тип рекомендаций** (как было с `probe`): добавить в
  `RECOMMENDATION_TYPES`, при необходимости — поля в `Recommendation` и
  live-схему, правило в системный промпт, ветку рендера в `renderRecs()`.
  Ядро слияния не трогается: неизвестные типы оно уже пропускает насквозь.
- **Новый ASR-бэкенд:** класс с `load()/transcribe()/describe()` +
  ветка в `asr/factory.py`. Больше нигде правок не нужно.
- **Не-Ollama LLM (LM Studio, llama-server):** движок использует клиент
  утиной типизацией (`check()`, `chat_json(system, user, schema)`), см.
  `FakeLLM` в `tests/test_e2e.py` как минимальный контракт. Для
  OpenAI-совместимых серверов достаточно альтернативного клиента.
- **Локализация UI:** все строки — в `app.js`/`index.html`; серверные
  сообщения — в контроллере/клиенте Ollama.

## 11. Инварианты (не ломать)

1. Никакие данные интервью не покидают машину; фронт не тянет внешние ресурсы.
2. `transcript.jsonl` — append-only и flush сразу: убитый процесс не теряет
   распознанное.
3. Статусы тем не понижаются автоматикой; ручная метка неприкосновенна для LLM.
4. Курсоры анализа двигаются только после успешного цикла.
5. Один экземпляр ASR-модели; тяжёлые зависимости импортируются лениво
   (ядро тестируется без sounddevice/whisper/Ollama).
6. Системный промпт цикла стабилен внутри сессии (кэш префикса).
