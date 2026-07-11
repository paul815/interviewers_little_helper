/* Interviewer's Little Helper — фронтенд. Vanilla JS, все данные через
   REST + WebSocket с локального сервера, никаких внешних запросов. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = {
  state: "idle",
  analyzing: false,
  nextAt: null,
  intervalS: 240,
  llmOk: false,
  devices: [],
  guideParsed: null,     // результат парсинга (на подтверждении)
  guideConfirmed: null,  // подтверждённая структура
  coverage: null,        // {topics, counts, iteration}
  recommendations: [],
  probes: [],
  segments: [],
  flags: [],
  startedAt: null,       // epoch, секунды
  durationMin: null,
  channels: {},          // interviewer/respondent -> alive
  libGuides: [],
  monitorOn: false,
  statusMsg: "",
  statusIsError: false,
};

/* ------------------------------------------------------------------ API */

async function api(path, opts = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(detail);
  }
  return resp.json();
}

/* ------------------------------------------------------------ WebSocket */

let ws = null;
let wsRetry = 1000;

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { wsRetry = 1000; };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    handleMessage(msg.type, msg.payload || {});
  };
  ws.onclose = () => {
    setStatus("Нет связи с сервером — переподключение…", true);
    setTimeout(connectWS, wsRetry);
    wsRetry = Math.min(wsRetry * 2, 8000);
  };
}

function handleMessage(type, p) {
  switch (type) {
    case "snapshot":
      applySnapshot(p);
      break;
    case "segment":
      S.segments.push(p);
      appendTranscriptRow(segmentHtml(p));
      break;
    case "coverage":
      S.coverage = p;
      renderTopics();
      renderHeader();
      renderPace();
      break;
    case "recommendations":
      S.recommendations = p.items || [];
      S.probes = p.probes || [];
      renderRecs();
      break;
    case "timer":
      S.nextAt = p.next_analysis_at;
      if (p.interval_s) S.intervalS = p.interval_s;
      break;
    case "analysis":
      S.analyzing = p.phase === "started";
      renderHeader();
      if (p.phase === "done") {
        setStatus(`Анализ #${p.iteration}${p.mode === "reconcile" ? " (сверка)" : ""} готов (${p.duration_s} с)`);
      }
      if (p.phase === "started" && p.mode !== "final") {
        setStatus(p.mode === "reconcile" ? "Сверочный анализ…" : (p.manual ? "Ручной анализ…" : "Анализ…"));
      }
      break;
    case "levels":
      renderLevels(p);
      break;
    case "channel":
      S.channels[p.speaker] = p.alive;
      renderChannelHealth();
      break;
    case "flag":
      S.flags.push(p);
      appendTranscriptRow(flagHtml(p));
      setStatus(`🚩 Момент отмечен (${fmtT(p.t)})`);
      break;
    case "session":
      S.state = p.state;
      if (p.state === "running") {
        S.startedAt = p.started_at || null;
        S.durationMin = p.duration_min || null;
      }
      if (p.state === "idle") {
        S.nextAt = null; S.analyzing = false;
        S.startedAt = null; S.channels = {};
      }
      renderHeader();
      renderPace();
      syncMonitor();
      break;
    case "status":
      if (p.state) { S.state = p.state; syncMonitor(); }
      if (p.message) setStatus(p.message, false);
      renderHeader();
      break;
    case "error":
      setStatus(p.message || "Ошибка", true);
      break;
  }
}

function applySnapshot(p) {
  S.state = p.state;
  S.analyzing = !!p.analyzing;
  S.nextAt = p.next_analysis_at;
  S.intervalS = p.interval_s || S.intervalS;
  S.segments = p.segments || [];
  S.flags = p.flags || [];
  S.coverage = p.coverage;
  S.recommendations = (p.recommendations && p.recommendations.items) || [];
  S.probes = (p.recommendations && p.recommendations.probes) || [];
  S.startedAt = p.started_at || null;
  S.durationMin = p.duration_min || null;
  S.channels = p.channels || {};
  if (p.guide) S.guideConfirmed = p.guide;
  const dur = $("duration");
  if (!dur.dataset.touched && p.default_duration_min) dur.value = p.default_duration_min;
  renderAll();
  syncMonitor();
  if (S.state === "running") setStatus("Сессия идёт");
  else if (!S.statusMsg) setStatus("Готов");
}

/* ------------------------------------------------------------- рендеры */

function setStatus(msg, isError = false) {
  S.statusMsg = msg;
  S.statusIsError = isError;
  const el = $("status-text");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

function renderAll() {
  renderHeader();
  renderRecs();
  renderTopics();
  renderTranscript();
  renderGuideBadge();
  renderPace();
  renderChannelHealth();
}

function canStart() {
  const mic = $("sel-mic").value, sys = $("sel-sys").value;
  return S.llmOk && S.guideConfirmed && mic !== "" && sys !== "" && mic !== sys;
}

function renderHeader() {
  const dot = $("status-dot");
  dot.className = "dot " + (S.statusIsError ? "error" : S.state);
  const btn = $("btn-startstop");
  if (S.state === "running" || S.state === "starting" || S.state === "stopping") {
    btn.textContent = S.state === "stopping" ? "Стоп…" : "Стоп";
    btn.className = "btn small danger";
    btn.disabled = S.state !== "running";
  } else {
    btn.textContent = "Старт";
    btn.className = "btn small primary";
    btn.disabled = !canStart();
  }
  $("btn-analyze").disabled = S.state !== "running" || S.analyzing;
  $("btn-flag").classList.toggle("hidden", S.state !== "running");
  const c = S.coverage && S.coverage.counts;
  $("cov-counter").textContent = c ? `${c.covered}/${c.total}` : "";
}

function fmtT(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function renderTimer() {
  const el = $("timer");
  if (S.analyzing) { el.textContent = "анализ…"; return; }
  if (S.state !== "running" || !S.nextAt) { el.textContent = "–:––"; return; }
  const left = Math.max(0, Math.round(S.nextAt - Date.now() / 1000));
  el.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
}
setInterval(() => { renderTimer(); renderPace(); }, 500);

/* ---- темп интервью ---- */

function renderPace() {
  const el = $("pace-line");
  const c = S.coverage && S.coverage.counts;
  if (S.state !== "running" || !S.startedAt || !S.durationMin || !c || !c.total) {
    el.classList.add("hidden");
    return;
  }
  const elapsedMin = (Date.now() / 1000 - S.startedAt) / 60;
  const expected = Math.min(elapsedMin / S.durationMin, 1);
  const actual = (c.covered + 0.5 * c.partial) / c.total;
  let cls = "ontrack", word = "в графике";
  if (actual < expected - 0.15) { cls = "behind"; word = "отстаёте"; }
  else if (actual > expected + 0.15) { cls = "ahead"; word = "с опережением"; }
  el.className = "pace " + cls;
  el.innerHTML =
    `<span>⏱ ${Math.floor(elapsedMin)}/${S.durationMin} мин</span>` +
    `<span>· покрыто ${c.covered}${c.partial ? `+${c.partial}◐` : ""} из ${c.total}</span>` +
    `<span class="pace-word">· ${word}</span>`;
  el.classList.remove("hidden");
}

/* ---- подсказки и пробы ---- */

const STATUS_ICON = { covered: "●", partial: "◐", not_covered: "○" };
const STATUS_RU = { covered: "покрыто", partial: "частично", not_covered: "не покрыто" };

function copyButtonsBind(box, list) {
  box.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", () => {
    const r = list[+b.dataset.i];
    copyText(r.suggested_question || r.topic_question || "");
    b.textContent = "✓";
    setTimeout(() => { b.textContent = "⧉"; }, 1200);
  }));
}

function renderRecs() {
  const pbox = $("probes-list");
  pbox.innerHTML = S.probes.map((r, i) => `
    <div class="probe">
      <div class="rec-head">
        <span class="chip">🔍 копнуть</span>
        <span class="rec-topic" title="${esc(r.topic_question || "")}">${esc(r.topic_question || "по свежей реплике")}</span>
        <button class="copy" data-i="${i}" title="Скопировать вопрос">⧉</button>
      </div>
      ${r.quote ? `<div class="probe-quote">«${esc(r.quote)}»</div>` : ""}
      <div class="rec-q">${esc(r.suggested_question || "")}</div>
      ${r.note ? `<div class="rec-note">${esc(r.note)}</div>` : ""}
    </div>`).join("");
  copyButtonsBind(pbox, S.probes);

  const box = $("recs-list");
  if (!S.recommendations.length && !S.probes.length) {
    const txt = S.state === "running"
      ? (S.coverage && S.coverage.iteration > 0
          ? "Все темы покрыты 🎉"
          : "Ждём первого анализа…")
      : "Подсказки появятся после первого анализа.";
    box.innerHTML = `<div class="placeholder">${txt}</div>`;
    return;
  }
  box.innerHTML = S.recommendations.map((r, i) => `
    <div class="rec ${r.urgency === "high" ? "high" : ""}">
      <div class="rec-head">
        <span class="chip">${esc(STATUS_RU[r.status] || r.type)}</span>
        <span class="rec-topic" title="${esc(r.topic_question || "")}">${esc(r.section_title || "")}</span>
        <button class="copy" data-i="${i}" title="Скопировать вопрос">⧉</button>
      </div>
      <div class="rec-q">${esc(r.suggested_question || r.topic_question || "")}</div>
      ${r.note ? `<div class="rec-note">${esc(r.note)}</div>` : ""}
    </div>`).join("");
  copyButtonsBind(box, S.recommendations);
}

/* ---- темы ---- */

function renderTopics() {
  const box = $("topics-list");
  const guide = S.guideConfirmed;
  if (!guide) {
    box.innerHTML = `<div class="placeholder">Сначала распарсьте и подтвердите гайд во вкладке «Настройка».</div>`;
    return;
  }
  const topics = (S.coverage && S.coverage.topics) || {};
  box.innerHTML = guide.sections.map((sec) => `
    <div class="sec-title">${esc(sec.title)}</div>
    ${sec.topics.map((t) => {
      const st = (topics[t.id] && topics[t.id].status) || "not_covered";
      const ev = topics[t.id] && topics[t.id].evidence;
      return `<div class="topic-row" title="${esc(ev || "")}">
        <span class="st-ico ${st}">${STATUS_ICON[st]}</span>
        <span class="topic-q ${st}">${esc(t.question)}</span>
      </div>`;
    }).join("")}`).join("");
}

/* ---- транскрипт (сегменты + флаги) ---- */

function segmentHtml(s) {
  const isI = s.speaker === "INTERVIEWER";
  return `<div class="seg">
    <span class="spk ${isI ? "i" : "r"}">${isI ? "И" : "Р"}</span>
    <span class="seg-t">${fmtT(s.t0)}</span>
    <span class="seg-x">${esc(s.text)}</span>
  </div>`;
}

function flagHtml(f) {
  return `<div class="seg flag-row">
    <span class="spk">🚩</span>
    <span class="seg-t">${fmtT(f.t)}</span>
    <span class="seg-x">${esc(f.note || "отмеченный момент")}</span>
  </div>`;
}

function renderTranscript() {
  const box = $("transcript-list");
  if (!S.segments.length && !S.flags.length) {
    box.innerHTML = `<div class="placeholder">Транскрипт появится после старта сессии.</div>`;
    return;
  }
  const rows = [
    ...S.segments.map((s) => ({ t: s.t0, html: segmentHtml(s) })),
    ...S.flags.map((f) => ({ t: f.t, html: flagHtml(f) })),
  ].sort((a, b) => a.t - b.t);
  box.innerHTML = rows.map((r) => r.html).join("");
  box.scrollTop = box.scrollHeight;
}

function appendTranscriptRow(html) {
  const box = $("transcript-list");
  const pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  if (box.querySelector(".placeholder")) box.innerHTML = "";
  box.insertAdjacentHTML("beforeend", html);
  if (pinned) box.scrollTop = box.scrollHeight;
}

/* ------------------------------------------------------------ настройка */

function renderGuideBadge() {
  const badge = $("guide-badge");
  if (S.guideConfirmed) {
    const n = S.guideConfirmed.sections.reduce((a, s) => a + s.topics.length, 0);
    badge.innerHTML = `<span class="ok">✓ «${esc(S.guideConfirmed.title)}», тем: ${n}</span>`;
  } else {
    badge.textContent = "не задан";
  }
}

async function loadDevices() {
  try {
    const data = await api("/api/devices");
    S.devices = data.devices;
    const mkOptions = (sel, preferFn) => {
      const cur = sel.value;
      sel.innerHTML = `<option value="">— выберите устройство —</option>` +
        S.devices.map((d) =>
          `<option value="${d.index}">${esc(d.name)} (${esc(d.hostapi)})</option>`).join("");
      if (cur && S.devices.some((d) => String(d.index) === cur)) sel.value = cur;
      else {
        const pref = S.devices.find(preferFn);
        if (pref) sel.value = String(pref.index);
      }
    };
    mkOptions($("sel-mic"), (d) => d.is_default_input);
    mkOptions($("sel-sys"), (d) => /blackhole|cable|vb-audio|virtual/i.test(d.name));
    if (data.error) setupMsg(data.error);
  } catch (e) {
    setupMsg("Не удалось получить список устройств: " + e.message);
  }
  renderHeader();
  syncMonitor(true);
}

async function checkLLM() {
  const el = $("llm-status");
  el.textContent = "проверка…";
  try {
    const st = await api("/api/llm/status");
    S.llmOk = st.ok;
    el.innerHTML = st.ok
      ? `<span class="ok">✓ Ollama ${esc(st.version || "")}, модель ${esc(st.model)}</span>`
      : `<span style="color:var(--high)">✗ ${esc(st.error || "недоступна")}</span>`;
  } catch (e) {
    S.llmOk = false;
    el.textContent = "✗ сервер недоступен";
  }
  renderHeader();
}

function setupMsg(text, ok = false) {
  const el = $("setup-msg");
  el.textContent = text || "";
  el.style.color = ok ? "var(--covered)" : "var(--high)";
}

/* ---- монитор уровней (до старта сессии) ---- */

function renderLevels(p) {
  const scale = (v) => Math.min(100, Math.round(Math.sqrt(v || 0) * 100));
  if ("interviewer" in p) $("vu-mic").style.width = scale(p.interviewer) + "%";
  if ("respondent" in p) $("vu-sys").style.width = scale(p.respondent) + "%";
}

function renderChannelHealth() {
  $("vu-mic").parentElement.classList.toggle("dead", S.channels.interviewer === false);
  $("vu-sys").parentElement.classList.toggle("dead", S.channels.respondent === false);
}

let monitorTimer = null;

function syncMonitor(force = false) {
  // Монитор нужен, когда сессии нет, открыта «Настройка» и выбраны устройства.
  clearTimeout(monitorTimer);
  monitorTimer = setTimeout(async () => {
    const setupActive = $("tab-setup").classList.contains("active");
    const mic = $("sel-mic").value, sys = $("sel-sys").value;
    const want = S.state === "idle" && setupActive && (mic !== "" || sys !== "");
    try {
      if (want && (!S.monitorOn || force)) {
        S.monitorOn = true;
        await api("/api/monitor/start", {
          method: "POST",
          body: JSON.stringify({
            mic_index: mic === "" ? null : +mic,
            system_index: sys === "" ? null : +sys,
          }),
        });
      } else if (!want && S.monitorOn) {
        S.monitorOn = false;
        await api("/api/monitor/stop", { method: "POST" });
        $("vu-mic").style.width = "0%";
        $("vu-sys").style.width = "0%";
      }
    } catch (e) { /* монитор — вспомогательный, не шумим */ }
  }, 200);
}

/* ---- библиотека гайдов ---- */

async function loadLibrary() {
  try {
    const data = await api("/api/guides");
    S.libGuides = data.guides || [];
    const sel = $("sel-guide-lib");
    sel.innerHTML = S.libGuides.length
      ? S.libGuides.map((g) =>
          `<option value="${esc(g.file_id)}">${esc(g.title)} (${g.topics_count} тем)</option>`).join("")
      : `<option value="">— пусто —</option>`;
  } catch (e) { /* библиотека не критична */ }
}

async function libLoad() {
  const id = $("sel-guide-lib").value;
  if (!id) return;
  try {
    const data = await api(`/api/guides/${encodeURIComponent(id)}`);
    S.guideConfirmed = data.guide;
    if (data.source_text) $("guide-text").value = data.source_text;
    $("guide-preview").classList.add("hidden");
    $("guide-editor").classList.remove("hidden");
    renderGuideBadge();
    renderTopics();
    renderHeader();
    setupMsg(`Гайд «${data.guide.title}» загружен из библиотеки.`, true);
  } catch (e) {
    setupMsg("Не удалось загрузить гайд: " + e.message);
  }
}

async function libDelete() {
  const id = $("sel-guide-lib").value;
  if (!id) return;
  try {
    await api(`/api/guides/${encodeURIComponent(id)}`, { method: "DELETE" });
    await loadLibrary();
  } catch (e) {
    setupMsg("Не удалось удалить гайд: " + e.message);
  }
}

/* ---- парсинг гайда ---- */

function showGuidePreview(guide) {
  $("guide-tree").innerHTML = guide.sections.map((sec) => `
    <div class="g-sec">${esc(sec.title)}</div>
    ${sec.topics.map((t) => `<div class="g-topic">• ${esc(t.question)}</div>`).join("")}`).join("");
  $("guide-editor").classList.add("hidden");
  $("guide-preview").classList.remove("hidden");
}

async function parseGuide() {
  const text = $("guide-text").value.trim();
  if (!text) { setupMsg("Вставьте текст гайда."); return; }
  const btn = $("btn-parse");
  btn.disabled = true;
  btn.textContent = "Парсинг (LLM)…";
  setupMsg("");
  try {
    S.guideParsed = await api("/api/guide/parse", { method: "POST", body: JSON.stringify({ text }) });
    showGuidePreview(S.guideParsed);
  } catch (e) {
    setupMsg("Не удалось распарсить гайд: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Распарсить гайд";
  }
}

async function confirmGuide() {
  S.guideConfirmed = S.guideParsed;
  renderGuideBadge();
  renderTopics();
  renderHeader();
  try {
    await api("/api/guides", {
      method: "POST",
      body: JSON.stringify({ guide: S.guideConfirmed, source_text: $("guide-text").value }),
    });
    await loadLibrary();
    setupMsg("Структура подтверждена и сохранена в библиотеку. Выберите устройства и нажмите «Старт».", true);
  } catch (e) {
    setupMsg("Структура подтверждена (в библиотеку не сохранилась: " + e.message + ")", true);
  }
}

/* ------------------------------------------------------------- действия */

async function startStop() {
  const btn = $("btn-startstop");
  btn.disabled = true;
  try {
    if (S.state === "idle") {
      await api("/api/session/start", {
        method: "POST",
        body: JSON.stringify({
          mic_index: +$("sel-mic").value,
          system_index: +$("sel-sys").value,
          guide: S.guideConfirmed,
          duration_min: +$("duration").value || null,
        }),
      });
      S.monitorOn = false; // сервер сам остановил монитор
    } else {
      await api("/api/session/stop", { method: "POST" });
    }
  } catch (e) {
    setStatus(e.message, true);
    setupMsg(e.message);
  } finally {
    btn.disabled = false;
    renderHeader();
  }
}

async function sendFlag() {
  try { await api("/api/session/flag", { method: "POST", body: JSON.stringify({ note: "" }) }); }
  catch (e) { setStatus(e.message, true); }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

/* ---------------------------------------------------------------- init */

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
  document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
  tab.classList.add("active");
  $("tab-" + tab.dataset.tab).classList.add("active");
  syncMonitor();
}));

$("btn-startstop").addEventListener("click", startStop);
$("btn-analyze").addEventListener("click", async () => {
  try { await api("/api/session/analyze", { method: "POST" }); }
  catch (e) { setStatus(e.message, true); }
});
$("btn-flag").addEventListener("click", sendFlag);
$("btn-parse").addEventListener("click", parseGuide);
$("btn-confirm-guide").addEventListener("click", confirmGuide);
$("btn-edit-guide").addEventListener("click", () => {
  $("guide-preview").classList.add("hidden");
  $("guide-editor").classList.remove("hidden");
});
$("btn-dev-refresh").addEventListener("click", loadDevices);
$("btn-llm-refresh").addEventListener("click", checkLLM);
$("btn-lib-load").addEventListener("click", libLoad);
$("btn-lib-del").addEventListener("click", libDelete);
$("sel-mic").addEventListener("change", () => { renderHeader(); syncMonitor(true); });
$("sel-sys").addEventListener("change", () => { renderHeader(); syncMonitor(true); });
$("duration").addEventListener("change", (e) => { e.target.dataset.touched = "1"; });

document.addEventListener("keydown", (e) => {
  if (e.code !== "KeyF" || e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  if (S.state === "running") { e.preventDefault(); sendFlag(); }
});

connectWS();
loadDevices();
checkLLM();
loadLibrary();
