# Архитектура Interviewer's Little Helper

Документ для разработчика: как устроена система, какие у неё контракты и где её
расширять. Пользовательская инструкция — в [README.md](../README.md), исходный
план с бюджетами памяти — в [PLAN.md](../PLAN.md), план тестирования — в
[TESTING.md](TESTING.md).

## 1. Обзор

Локальное приложение для качественных интервью по Zoom. Два аудиоканала
(микрофон = интервьюер, виртуальный кабель = респондент) режутся на реплики
потоковым VAD и транскрибируются Parakeet'ом; раз в 4 минуты локальная LLM
сравнивает новую часть разговора с гайдом и обновляет чеклист покрытия тем +
рекомендации. UI — веб-страница 600×400 в always-on-top окне pywebview.

Ключевые свойства:

- **Всё локально.** Сервер слушает только `127.0.0.1`, фронтенд не делает ни
  одного внешнего запроса, телеметрии нет. Сеть нужна один раз — скачать веса.
- **Разделение говорящих на уровне ОС**, не диаризацией: канал = говорящий.
- **Реплика — единица обработки.** Потоковый Silero VAD закрывает реплику через
  `vad_redemption_s` тишины, и она сразу уходит в ASR. Не потоковое
  распознавание (партиалов нет), но и не ожидание заполнения буфера.
- **ASR не занимает VRAM.** Parakeet в int8 идёт на CPU; видеопамять целиком
  под LLM. Whisper остаётся переключаемой опцией (см. §11.7).
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
    speech_events.py   SpeechStart/SpeechEnd, SpeechStateMachine, EnergyStreamProcessor
    silero_stream.py   SileroStreamProcessor: покадровая инференция (onnxruntime)
    data/              silero_vad.onnx (~2.3 МБ, MIT, вендорится в репозиторий)
    vad.py             Батчевый SpeechDetector для запасного пути: Silero / Energy
    chunker.py         StreamingChunkAssembler (основной) + ChunkAssembler + ChunkerThread
  asr/
    base.py            ASRBackend.load()/transcribe() -> ASRResult, warnings()
    parakeet_backend.py Parakeet TDT v3 через onnx-asr (бэкенд по умолчанию)
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
tools/bench_asr.py     RTF по каждому доступному ASR-бэкенду на одном WAV
tools/fetch_asr_model.py  Предзагрузка весов ASR
install.py             Установка окружения, один скрипт на все платформы
```

## 3. Поток данных и модель конкурентности

```
PortAudio callback (×2) ──▶ RingBuffer (×2) ──▶ ChunkerThread (×2) ──▶ Queue ──▶ ASRWorker (×1)
     (только копирование)     (drop-oldest)     (потоковый VAD,      (cap 60)     (одна модель)
                                                 реплика = чанк)
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
- **Один экземпляр ASR-модели** на оба канала; порядок сегментов
  восстанавливается сортировкой по `t0` на чтении.
- **У каждого канала свой экземпляр VAD.** Машина состояний Silero хранит LSTM-
  состояние потока — делить его между каналами нельзя.
- Очередь ASR ограничена 60 чанками: живой, но отстающий бэкенд жалуется в лог,
  мёртвый — не съедает память.
- Остановка строго упорядочена (`controller._stop_audio_asr`): планировщик →
  захват → чанкеры (flush хвоста) → ASR дорабатывает очередь → финальная
  сверка → отчёт → финальная запись.

Таймстампы: `t0/t1` сегмента — секунды от старта захвата (позиция в потоке
16 kHz), не wall-clock; wall-clock фиксируется в `created_at` и `meta.json`.

### 3.1 Как режется реплика

`StreamingChunkAssembler` держит скользящее окно недавнего аудио и абсолютную
позицию его начала. `SpeechEventSource` (Silero или энергия) отдаёт события:

| Событие | Что делает ассемблер |
|---|---|
| `SpeechStart` | запоминает начало реплики |
| `SpeechEnd` | вырезает `[начало − pre_pad_s; конец + pad_s]` и отдаёт чанк |
| нет события, речь идёт дольше `max_chunk_s` | принудительный разрез без pad'ов |
| `flush()` на остановке | закрывает незавершённую реплику |

Два инварианта, которые легко сломать при правках:

1. `_emitted_through` помнит, до какого сэмпла аудио уже отдано. После
   принудительного разреза `SpeechEnd` приносит **исходное** начало реплики —
   без этой отсечки кусок ушёл бы в ASR дважды и слова задвоились бы в
   транскрипте.
2. Pre-pad добавляется только на настоящем начале реплики, post-pad — только на
   настоящем конце. На стыке принудительного разреза pad'ы дали бы перекрытие.

Задержка сегмента = `vad_redemption_s` + `poll_interval_s` + время
декодирования. Поэтому `poll_interval_s` (0.2 с) заметно меньше
`vad_redemption_s` (0.6 с): опрос реже съел бы весь выигрыш.

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
| `audio.poll_interval_s` | 0.2 | как часто чанкер забирает аудио = гранулярность задержки |
| `audio.max_chunk_s / pad_s / pre_pad_s` | 25 / 0.2 / 0.3 | потолок монолога, post-pad, pre-pad |
| `audio.min_pause_s / min_speech_s` | 0.7 / 0.3 | только батчевая нарезка (`vad: silero\|energy`) |
| `audio.vad` | auto | auto/silero-stream/energy-stream — потоковый; silero/energy — батчевый |
| `audio.vad_positive_threshold / negative` | 0.50 / 0.35 | гистерезис входа/выхода из речи |
| `audio.vad_min_speech_s` | 0.25 | короче — щелчок, не реплика |
| `audio.vad_redemption_s` | 0.6 | столько тишины = конец реплики; главный рычаг задержки |
| `audio.watchdog_silence_s` | 12 | нет сэмплов дольше — канал «мёртв» |
| `asr.backend` | auto | auto/parakeet/mlx/faster/faster-cpu |
| `asr.parakeet_model` | nemo-parakeet-tdt-0.6b-v3 | любая модель onnx-asr (напр. `gigaam-v2-rnnt`) |
| `asr.parakeet_quantization` | int8 | int8 ≈ 650 МБ против ~2.4 ГБ fp32 |
| `asr.providers` | ["CPUExecutionProvider"] | CUDA требует onnxruntime-gpu — см. §11.7 |
| `asr.model / mlx_model` | large-v3-turbo | модель Whisper (бэкенды mlx/faster) |
| `asr.compute_type` | auto | cuda→float16, cpu→int8; `int8_float16` для экономии |
| `asr.vocabulary` | "" | термины проекта → initial_prompt; **Parakeet не поддерживает** |
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
  ветка в `asr/factory.py`. Больше нигде правок не нужно. Если бэкенд не умеет
  что-то из настроек (как Parakeet — `asr.vocabulary`), верните это из
  `warnings()`: воркер покажет текст в шапке UI, вместо того чтобы молча
  игнорировать настройку.
  Готовый кандидат: **Parakeet через parakeet-mlx** — на Apple Silicon Metal
  заметно быстрее, чем нынешний onnx-asr на CPU. У stenoai это отдельный модуль
  за диспетчером по `sys.platform`, публичная поверхность та же.
- **Новый источник событий речи:** класс с `process()/flush()/in_speech/reset()`
  (протокол `SpeechEventSource`) + ветка в `create_stream_processor`. Полезно
  для WebRTC VAD или готового детектора из другой библиотеки.
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
   (ядро тестируется без sounddevice/whisper/onnxruntime/Ollama). Поэтому
   `EnergyStreamProcessor` живёт в `speech_events.py`, а не рядом с Silero:
   на нём гоняются тесты нарезки в CI.
6. Системный промпт цикла стабилен внутри сессии (кэш префикса).
7. **`onnxruntime` — только CPU-сборка.** `onnxruntime-gpu` конфликтует с ней в
   одном окружении, а CPU-версию тянет и faster-whisper. Перевод Parakeet на
   `CUDAExecutionProvider` — это смена зависимости и отказ от faster-whisper
   как опции, а не правка одного ключа конфига. Выигрыш неочевиден: fp32-энкодер
   занял бы больше VRAM, чем занимал Whisper, а её забирает LLM.
8. `StreamingChunkAssembler` не отдаёт один и тот же сэмпл дважды
   (`_emitted_through`) и не оставляет дыр между соседними чанками — см. §3.1.
