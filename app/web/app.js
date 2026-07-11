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
  segments: [],
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
      appendSegment(p);
      break;
    case "coverage":
      S.coverage = p;
      renderTopics();
      renderHeader();
      break;
    case "recommendations":
      S.recommendations = p.items || [];
      renderRecs();
      break;
    case "timer":
      S.nextAt = p.next_analysis_at;
      if (p.interval_s) S.intervalS = p.interval_s;
      break;
    case "analysis":
      S.analyzing = p.phase === "started";
      renderHeader();
      if (p.phase === "done") setStatus(`Анализ #${p.iteration} готов (${p.duration_s} с)`);
      if (p.phase === "started") setStatus(p.manual ? "Ручной анализ…" : "Анализ…");
      break;
    case "session":
      S.state = p.state;
      if (p.state === "idle") { S.nextAt = null; S.analyzing = false; }
      renderHeader();
      break;
    case "status":
      if (p.state) S.state = p.state;
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
  S.coverage = p.coverage;
  S.recommendations = (p.recommendations && p.recommendations.items) || [];
  if (p.guide) S.guideConfirmed = p.guide;
  renderAll();
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
}

function canStart() {
  const mic = $("sel-mic").value, sys = $("sel-sys").value;
  return S.llmOk && S.guideConfirmed && mic !== "" && sys !== "" && mic !== sys;
}

function renderHeader() {
  const dot = $("status-dot");
  dot.className = "dot " + (S.statusIsError ? "error" : S.state);
  const btn = $("btn-startstop");
  if (S.state === "running" || S.state === "starting") {
    btn.textContent = "Стоп";
    btn.className = "btn small danger";
    btn.disabled = S.state === "starting";
  } else {
    btn.textContent = "Старт";
    btn.className = "btn small primary";
    btn.disabled = S.state !== "idle" || !canStart();
  }
  $("btn-analyze").disabled = S.state !== "running" || S.analyzing;
  const c = S.coverage && S.coverage.counts;
  $("cov-counter").textContent = c ? `${c.covered}/${c.total}` : "";
}

function renderTimer() {
  const el = $("timer");
  if (S.analyzing) { el.textContent = "анализ…"; return; }
  if (S.state !== "running" || !S.nextAt) { el.textContent = "–:––"; return; }
  const left = Math.max(0, Math.round(S.nextAt - Date.now() / 1000));
  el.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
}
setInterval(renderTimer, 500);

const STATUS_ICON = { covered: "●", partial: "◐", not_covered: "○" };
const STATUS_RU = { covered: "покрыто", partial: "частично", not_covered: "не покрыто" };

function renderRecs() {
  const box = $("recs-list");
  if (!S.recommendations.length) {
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
  box.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", () => {
    const r = S.recommendations[+b.dataset.i];
    copyText(r.suggested_question || r.topic_question || "");
    b.textContent = "✓";
    setTimeout(() => { b.textContent = "⧉"; }, 1200);
  }));
}

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

function segmentHtml(s) {
  const mm = Math.floor(s.t0 / 60), ss = Math.floor(s.t0 % 60);
  const isI = s.speaker === "INTERVIEWER";
  return `<div class="seg">
    <span class="spk ${isI ? "i" : "r"}">${isI ? "И" : "Р"}</span>
    <span class="seg-t">${mm}:${String(ss).padStart(2, "0")}</span>
    <span class="seg-x">${esc(s.text)}</span>
  </div>`;
}

function renderTranscript() {
  const box = $("transcript-list");
  if (!S.segments.length) {
    box.innerHTML = `<div class="placeholder">Транскрипт появится после старта сессии.</div>`;
    return;
  }
  box.innerHTML = [...S.segments].sort((a, b) => a.t0 - b.t0).map(segmentHtml).join("");
  box.scrollTop = box.scrollHeight;
}

function appendSegment(s) {
  const box = $("transcript-list");
  const pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  if (S.segments.length === 1) box.innerHTML = "";
  box.insertAdjacentHTML("beforeend", segmentHtml(s));
  if (pinned) box.scrollTop = box.scrollHeight;
}

function renderGuideBadge() {
  const badge = $("guide-badge");
  if (S.guideConfirmed) {
    const n = S.guideConfirmed.sections.reduce((a, s) => a + s.topics.length, 0);
    badge.innerHTML = `<span class="ok">✓ «${esc(S.guideConfirmed.title)}», тем: ${n}</span>`;
  } else {
    badge.textContent = "не задан";
  }
}

/* ------------------------------------------------------------ настройка */

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
        }),
      });
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
}));

$("btn-startstop").addEventListener("click", startStop);
$("btn-analyze").addEventListener("click", async () => {
  try { await api("/api/session/analyze", { method: "POST" }); }
  catch (e) { setStatus(e.message, true); }
});
$("btn-parse").addEventListener("click", parseGuide);
$("btn-confirm-guide").addEventListener("click", () => {
  S.guideConfirmed = S.guideParsed;
  renderGuideBadge();
  renderTopics();
  renderHeader();
  setupMsg("Структура гайда подтверждена. Выберите устройства и нажмите «Старт».", true);
});
$("btn-edit-guide").addEventListener("click", () => {
  $("guide-preview").classList.add("hidden");
  $("guide-editor").classList.remove("hidden");
});
$("btn-dev-refresh").addEventListener("click", loadDevices);
$("btn-llm-refresh").addEventListener("click", checkLLM);
$("sel-mic").addEventListener("change", renderHeader);
$("sel-sys").addEventListener("change", renderHeader);

connectWS();
loadDevices();
checkLLM();
