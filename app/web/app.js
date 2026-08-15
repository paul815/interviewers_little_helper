/* Interviewer's Little Helper — front end. Vanilla JS, all data over
   REST + WebSocket from the local server, no external requests. */
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
  guideParsed: null,     // the parse result (awaiting confirmation)
  guideConfirmed: null,  // the confirmed structure
  coverage: null,        // {topics, counts, iteration}
  recommendations: [],
  probes: [],
  segments: [],
  flags: [],
  questions: [],         // "ask as well": {id, t, text, done}
  guideText: "",         // the original guide — what was pasted in "Setup"
  guideBlocks: [],       // the same text, split into sections and questions
  curBlock: null,        // the selected question: comments attach to it
  recFresh: new Set(),   // hints that arrived in the last analysis
  sessionId: null,       // id of the running interview
  startedAt: null,       // epoch, seconds
  durationMin: null,
  channels: {},          // interviewer/respondent -> alive
  setup: null,           // {running, items} — state of the model downloads
  wizardHidden: false,   // "Later" in the wizard: do not show until a reload
  libGuides: [],
  projects: [],
  projectId: "",         // "" — an interview outside a project
  projectSessions: [],
  projectCoverage: null,
  openSession: null,     // id of the interview expanded in the list — its notes show
  sessionNotes: {},      // session_id -> {flags, questions, ...} | "loading" | {error}
  monitorOn: false,
  statusMsg: "",
  statusIsError: false,
  gfs: +localStorage.getItem("ilh_gfs") || 20,  // guide font size, the A−/A+ buttons
  theme: localStorage.getItem("ilh_theme") || "system",  // system | light | dark
  screen: "start",       // start | work — the project list or work on one project
  projectQuery: "",      // the search string on the start screen
  prompts: null,         // the prompt editor, null — never opened yet
  viewer: null,          // the interview open in the viewer: {sessionId, rows}
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
let recoveredShown = false;  // a recovered session is announced once per window

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { wsRetry = 1000; };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    handleMessage(msg.type, msg.payload || {});
  };
  ws.onclose = () => {
    setStatus("No connection to the server — reconnecting…", true);
    setTimeout(connectWS, wsRetry);
    wsRetry = Math.min(wsRetry * 2, 8000);
  };
}

function handleMessage(type, p) {
  switch (type) {
    case "snapshot":
      applySnapshot(p);
      break;
    case "setup_progress":
      renderSetup(p);
      break;
    case "segment":
      S.segments.push(p);
      appendTranscriptRow(segmentHtml(p));
      break;
    case "coverage":
      S.coverage = p;
      renderTopics();
      renderLiveGuide();
      renderHeader();
      renderPace();
      break;
    case "recommendations":
      S.recommendations = p.items || [];
      S.probes = p.probes || [];
      markFreshRecs();
      renderLiveRecs();
      break;
    case "timer":
      S.nextAt = p.next_analysis_at;
      if (p.interval_s) S.intervalS = p.interval_s;
      break;
    case "analysis":
      S.analyzing = p.phase === "started";
      renderHeader();
      if (p.phase === "done") {
        setStatus(`Analysis #${p.iteration}${p.mode === "reconcile" ? " (reconcile)" : ""} done (${p.duration_s} s)`);
      }
      if (p.phase === "started" && p.mode !== "final") {
        setStatus(p.mode === "reconcile" ? "Reconciliation analysis…" : (p.manual ? "Manual analysis…" : "Analysing…"));
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
      renderComments();
      renderLiveGuide();
      setStatus(p.note ? `💬 Comment saved (${fmtT(p.t)})`
                       : `🚩 Moment flagged (${fmtT(p.t)})`);
      break;
    case "question": {
      const i = S.questions.findIndex((q) => q.id === p.id);
      if (i < 0) S.questions.push(p); else S.questions[i] = p;
      renderQuestions();
      break;
    }
    case "session":
      S.state = p.state;
      if (p.state === "running") {
        // A new interview in the same window: the previous one's notes must not linger.
        if (p.session_id && p.session_id !== S.sessionId) {
          S.segments = []; S.flags = []; S.questions = [];
          recSeen = new Set(); S.recFresh = new Set();
        }
        S.startedAt = p.started_at || null;
        S.durationMin = p.duration_min || null;
        S.sessionId = p.session_id || null;
      }
      if (p.state === "idle") {
        S.nextAt = null; S.analyzing = false;
        S.startedAt = null; S.channels = {};
        S.sessionId = null;
      }
      renderHeader();
      renderTopics();  // rebind the clicks when the mode changes
      renderLive();
      renderTranscript();
      renderPace();
      syncMonitor();
      // An interview appeared in the series (or ended and gained coverage).
      loadProjectData();
      break;
    case "status":
      if (p.state) { S.state = p.state; syncMonitor(); }
      if (p.message) setStatus(p.message, false);
      renderHeader();
      break;
    case "error":
      setStatus(p.message || "Error", true);
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
  S.questions = p.questions || [];
  S.coverage = p.coverage;
  S.recommendations = (p.recommendations && p.recommendations.items) || [];
  S.probes = (p.recommendations && p.recommendations.probes) || [];
  // A window reconnect is no reason to highlight everything as "just arrived".
  recSeen = new Set(liveRecItems().map(recKey));
  S.recFresh = new Set();
  S.sessionId = p.session_id || null;
  S.startedAt = p.started_at || null;
  S.durationMin = p.duration_min || null;
  S.channels = p.channels || {};
  if (p.guide) S.guideConfirmed = p.guide;
  // The session may have started from another window — the original guide comes with it.
  if (p.guide_text) {
    if (!$("guide-text").value.trim()) $("guide-text").value = p.guide_text;
    setGuideText(p.guide_text);
  } else if (!S.guideText) {
    setGuideText($("guide-text").value);
  }
  const dur = $("duration");
  if (!dur.dataset.touched && p.default_duration_min) dur.value = p.default_duration_min;
  // The session may have started before this window opened: pick up its project
  // so the selector and the "Project" tab show what is actually being recorded.
  if (p.project_id && p.project_id !== S.projectId) {
    S.projectId = p.project_id;
    renderWorkTitle();
    loadProjectData();
  }
  // The window was opened mid-interview — the project list is of no use here.
  if (S.state === "running" && S.screen === "start") { showScreen("work"); showTab("live"); }
  renderAll();
  syncMonitor();
  if (S.state === "running") setStatus("Session running");
  else if (!S.statusMsg) setStatus("Ready");
  // The previous run was cut short — the server has already repaired the folder,
  // and a human needs to hear it once, not on every websocket reconnect.
  if (!recoveredShown && p.recovered && p.recovered.length) {
    recoveredShown = true;
    const r = p.recovered[p.recovered.length - 1];
    const mins = Math.round((r.audio_seconds || 0) / 60);
    setStatus(
      `The previous interview crashed: recovered ${r.session_id}` +
      (r.audio_seconds ? ` — ${mins} min of audio` : "") +
      `, ${r.segments || 0} utterances. No report was built`,
      true,
    );
  }
}

/* -------------------------------------------------------------- render */

function setStatus(msg, isError = false) {
  S.statusMsg = msg;
  S.statusIsError = isError;
  const el = $("status-text");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

function renderAll() {
  renderHeader();
  renderTopics();
  renderLive();
  renderTranscript();
  renderGuideBadge();
  renderPace();
  renderChannelHealth();
  renderProject();
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
    btn.textContent = S.state === "stopping" ? "Stopping…" : "Stop";
    btn.className = "btn small danger";
    btn.disabled = S.state !== "running";
  } else {
    btn.textContent = "Start";
    btn.className = "btn small primary";
    btn.disabled = !canStart();
  }
  $("btn-analyze").disabled = S.state !== "running" || S.analyzing;
  $("btn-flag").classList.toggle("hidden", S.state !== "running");
  const c = S.coverage && S.coverage.counts;
  $("cov-counter").textContent = c ? `${c.covered}/${c.total}` : "";
  // "Can we start" is the same question the readiness line on the start screen asks.
  renderReady();
}

function fmtT(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function renderTimer() {
  const el = $("timer");
  if (S.analyzing) { el.textContent = "analysing…"; return; }
  if (S.state !== "running" || !S.nextAt) { el.textContent = "–:––"; return; }
  const left = Math.max(0, Math.round(S.nextAt - Date.now() / 1000));
  el.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
}
setInterval(() => { renderTimer(); renderPace(); renderLiveClock(); }, 500);

/* ---- interview pace ---- */

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
  let cls = "ontrack", word = "on track";
  if (actual < expected - 0.15) { cls = "behind"; word = "behind"; }
  else if (actual > expected + 0.15) { cls = "ahead"; word = "ahead"; }
  el.className = "pace " + cls;
  el.innerHTML =
    `<span>⏱ ${Math.floor(elapsedMin)}/${S.durationMin} min</span>` +
    `<span>· covered ${c.covered}${c.partial ? `+${c.partial}◐` : ""} of ${c.total}</span>` +
    `<span class="pace-word">· ${word}</span>`;
  el.classList.remove("hidden");
}

/* ---- hints and probes ---- */

const STATUS_ICON = { covered: "●", partial: "◐", not_covered: "○" };
const STATUS_LABEL = { covered: "covered", partial: "partial", not_covered: "not covered" };

/* The cards themselves are drawn by renderLiveRecs — hints live only in the
   column of the interview screen; there is no separate tab for them any more. */

/* ---- topics ---- */

function renderTopics() {
  const box = $("topics-list");
  const guide = S.guideConfirmed;
  if (!guide) {
    box.innerHTML = `<div class="placeholder">First parse and confirm the guide on the «Project settings» tab.</div>`;
    return;
  }
  const topics = (S.coverage && S.coverage.topics) || {};
  const running = S.state === "running";
  box.innerHTML = guide.sections.map((sec) => `
    <div class="sec-title">${esc(sec.title)}</div>
    ${sec.topics.map((t) => {
      const ts = topics[t.id] || {};
      const st = ts.status || "not_covered";
      const manual = !!ts.manual;
      const tip = manual
        ? "Marked by hand — click to hand it back to the LLM"
        : (running ? "Click to mark it covered by hand. " : "") + (ts.evidence || "");
      return `<div class="topic-row ${running ? "clickable" : ""}" data-tid="${esc(t.id)}"
                   data-manual="${manual ? 1 : 0}" title="${esc(tip)}">
        <span class="st-ico ${st} ${manual ? "manual" : ""}">${manual ? "✔" : STATUS_ICON[st]}</span>
        <span class="topic-q ${st}">${esc(t.question)}</span>
      </div>`;
    }).join("")}`).join("");
  if (running) {
    box.querySelectorAll(".topic-row").forEach((row) => row.addEventListener("click", async () => {
      const manual = row.dataset.manual === "1";
      try {
        await api("/api/topics/status", {
          method: "POST",
          body: JSON.stringify({ topic_id: row.dataset.tid, status: manual ? null : "covered" }),
        });
      } catch (e) { setStatus(e.message, true); }
    }));
  }
}

/* ---- interview screen: the guide, large, plus comments and extra questions ---- */

/* Cyrillic stays in the character class on purpose: the interface is English,
   but a guide may be written in any language and still has to match. */
const normQ = (s) => String(s || "").toLowerCase().replace(/ё/g, "е")
  .replace(/[^0-9a-zа-я]+/g, " ").trim();

/** The original guide text → sections and questions.
 *
 *  We show the pasted text, not the LLM's parse: the parse rewrites the wording
 *  and sometimes loses questions, while in the interview the original is read
 *  out loud. Section titles come from the parse — the LLM does not touch those;
 *  without a parse we fall back to the crude heuristic "a short line with no
 *  punctuation at the end". */
function buildGuideBlocks(text) {
  const secTitles = new Map();
  const topics = [];
  for (const sec of (S.guideConfirmed || {}).sections || []) {
    secTitles.set(normQ(sec.title), sec.title);
    for (const t of sec.topics) topics.push({ id: t.id, q: normQ(t.question) });
  }
  const out = [];
  let section = "", n = 0, buf = [];
  const flush = () => {
    const lines = buf.map((l) => l.trim()).filter(Boolean);
    buf = [];
    if (!lines.length) return;
    // Lines in brackets are "probes" under a question, not a question of their own.
    let body = lines.filter((l) => !l.startsWith("("));
    const probe = lines.filter((l) => l.startsWith("("));
    if (!body.length) { body = probe.splice(0); }
    const raw = body.join("\n");
    const nrm = normQ(raw);
    // There are usually more topics than blocks: the parse splits a paragraph of
    // three questions into three topics. Keep them all — otherwise a hint about
    // the second topic has nowhere to land.
    const hits = topics.filter((t) => t.q && (nrm.includes(t.q) || t.q.includes(nrm)));
    out.push({
      kind: "blk", id: "b" + n++, section, text: raw,
      probe: probe.join("\n"), topicIds: hits.map((t) => t.id),
    });
  };
  for (const line of String(text || "").replace(/\r/g, "").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) { flush(); continue; }
    const known = secTitles.get(normQ(trimmed));
    const guessed = !secTitles.size && trimmed.length < 70 && !/[?.!:;,]$/.test(trimmed);
    if (known || guessed) {
      flush();
      section = known || trimmed;
      out.push({ kind: "sec", id: "s" + n++, title: section });
      continue;
    }
    buf.push(line);
  }
  flush();
  return out;
}

function setGuideText(text) {
  S.guideText = text || "";
  S.guideBlocks = buildGuideBlocks(S.guideText);
  if (!S.guideBlocks.some((b) => b.id === S.curBlock)) {
    const first = S.guideBlocks.find((b) => b.kind === "blk");
    S.curBlock = first ? first.id : null;
  }
  renderLive();
}

const curBlock = () => S.guideBlocks.find((b) => b.id === S.curBlock) || null;

/** A short caption for the question: it travels into flags.jsonl and transcript.md. */
function blockAnchor(b) {
  if (!b) return "";
  const first = b.text.split("\n")[0].trim();
  return first.length > 160 ? first.slice(0, 157) + "…" : first;
}

function blockComments(b) {
  const a = normQ(blockAnchor(b));
  return a ? S.flags.filter((f) => f.note && normQ(f.anchor || "") === a) : [];
}

/** A block's status covers all of its topics at once: "partial" until all are closed. */
function blockStatus(b) {
  const topics = (S.coverage && S.coverage.topics) || {};
  const list = (b.topicIds || []).map((id) => topics[id]).filter(Boolean);
  if (!list.length) return "";
  const st = list.map((t) => t.status || "not_covered");
  if (st.every((s) => s === "covered")) return "covered";
  if (st.every((s) => s === "not_covered")) return "not_covered";
  return "partial";
}

const blockForTopic = (topicId) => (topicId
  ? S.guideBlocks.find((b) => b.kind === "blk" && (b.topicIds || []).includes(topicId))
  : null) || null;

function fmtWall(iso) {
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

function renderLiveGuide() {
  const box = $("live-guide");
  if (!S.guideBlocks.length) {
    box.innerHTML = `<div class="placeholder">No guide set.<br>
      The «Project settings» tab → «Research guide»: paste the text and it will be
      visible here in full, word for word.</div>`;
    return;
  }
  box.innerHTML = S.guideBlocks.map((b) => {
    if (b.kind === "sec") return `<div class="lv-sec">${esc(b.title)}</div>`;
    const st = blockStatus(b);
    const n = blockComments(b).length;
    return `<div class="lv-blk ${b.id === S.curBlock ? "cur" : ""} ${st}" data-blk="${b.id}">
      ${st ? `<span class="lv-ico ${st}">${STATUS_ICON[st]}</span>` : ""}
      <span class="lv-q">${esc(b.text)}</span>
      ${n ? `<span class="lv-n">💬 ${n}</span>` : ""}
      ${b.probe ? `<span class="lv-probe">${esc(b.probe)}</span>` : ""}
    </div>`;
  }).join("");
  box.querySelectorAll(".lv-blk").forEach((el) => el.addEventListener("click", () => {
    S.curBlock = el.dataset.blk;
    renderLiveGuide();
    renderAnchorHint();
  }));
}

function renderAnchorHint() {
  const el = $("cmt-anchor");
  const b = curBlock();
  el.innerHTML = b
    ? `on question: <b>${esc(blockAnchor(b))}</b>`
    : `select a question in the guide and the comment will attach to it`;
  el.title = b ? blockAnchor(b) : "";
}

function renderComments() {
  const box = $("cmt-list");
  $("cmt-n").textContent = S.flags.length;
  if (!S.flags.length) {
    box.innerHTML = `<div class="placeholder">${S.state === "running"
      ? "Type in the field below — the time and the selected question fill themselves in."
      : "Comments are written during the interview."}</div>`;
    return;
  }
  box.innerHTML = [...S.flags].reverse().map((f) => `
    <div class="cmt ${f.note ? "" : "bare"}">
      <div class="cmt-meta"><b>${fmtT(f.t)}</b><span>${fmtWall(f.ts)}</span></div>
      ${esc(f.note || "flagged moment (F5)")}
      ${f.anchor ? `<span class="cmt-anchor" title="${esc(f.anchor)}">↳ ${esc(f.anchor)}</span>` : ""}
    </div>`).join("");
}

function renderQuestions() {
  const box = $("q-list");
  $("q-n").textContent = S.questions.filter((q) => !q.done).length;
  if (!S.questions.length) {
    box.innerHTML = `<div class="placeholder">Whatever comes to mind along the way
      goes here, so you do not interrupt the other person.</div>`;
    return;
  }
  // Asked ones move to the bottom but do not disappear: they show up in the report.
  const sorted = [...S.questions].sort((a, b) => (a.done - b.done) || (a.t - b.t));
  box.innerHTML = sorted.map((q) => `
    <label class="qrow ${q.done ? "done" : ""}">
      <input type="checkbox" data-qid="${q.id}" ${q.done ? "checked" : ""}>
      <span class="qtext">${esc(q.text)}</span>
      <span class="qt">${fmtT(q.t)}</span>
    </label>`).join("");
  box.querySelectorAll("input[data-qid]").forEach((cb) => cb.addEventListener("change", async () => {
    try {
      await api("/api/session/question/done", {
        method: "POST",
        body: JSON.stringify({ question_id: +cb.dataset.qid, done: cb.checked }),
      });
    } catch (e) {
      cb.checked = !cb.checked;
      setStatus(e.message, true);
    }
  }));
}

/* LLM hints on the interview screen.
   The analysis arrives once every few minutes — by then the researcher has
   already moved on through the guide. So a card lives in its own corner, carries
   the "where" (section → question) and on click sends the guide back to that spot. */

const liveRecItems = () => [
  ...S.probes.map((r) => ({ ...r, kind: "probe" })),
  ...S.recommendations.map((r) => ({ ...r, kind: "rec" })),
];
const recKey = (r) => `${r.kind}|${r.topic_id || ""}|${r.suggested_question || ""}`;

let recSeen = new Set();
let recFreshTimer = null;

/** Mark the cards that arrived in this cycle — they get highlighted and counted. */
function markFreshRecs() {
  const keys = liveRecItems().map(recKey);
  S.recFresh = new Set(keys.filter((k) => !recSeen.has(k)));
  recSeen = new Set(keys);
  clearTimeout(recFreshTimer);
  if (S.recFresh.size) {
    recFreshTimer = setTimeout(() => { S.recFresh = new Set(); renderLiveRecs(); }, 15000);
  }
}

function recCardHtml(r, i) {
  const where = r.kind === "probe"
    ? (r.topic_question || "on a fresh utterance")
    : [r.section_title, r.topic_question].filter(Boolean).join(" → ");
  const b = blockForTopic(r.topic_id);
  const cls = ["rc", r.kind, r.urgency === "high" ? "high" : "",
    S.recFresh.has(recKey(r)) ? "fresh" : ""].filter(Boolean).join(" ");
  return `<div class="${cls}" ${b ? `data-goto="${b.id}"` : ""}>
    <div class="rc-head">
      <span class="chip">${r.kind === "probe" ? "🔍 dig in" : esc(STATUS_LABEL[r.status] || r.type || "")}</span>
      <span class="rc-where" title="${esc(where)}">${esc(where)}</span>
      <button class="copy" data-reccopy="${i}" title="Copy the question">⧉</button>
      ${r.kind === "rec" && r.topic_id
        ? `<button class="copy" data-rechide="${esc(r.topic_id)}"
             title="Hide the hint for this topic">✕</button>` : ""}
    </div>
    ${r.quote ? `<div class="rc-quote">«${esc(r.quote)}»</div>` : ""}
    <div class="rc-q">${esc(r.suggested_question || r.topic_question || "")}</div>
    ${r.note ? `<div class="rc-note">${esc(r.note)}</div>` : ""}
    ${b ? `<div class="rc-goto">↑ show in the guide</div>` : ""}
  </div>`;
}

function renderLiveRecs() {
  const box = $("rec-live");
  const items = liveRecItems();
  $("rec-n").textContent = items.length;
  const fresh = $("rec-fresh");
  fresh.textContent = S.recFresh.size ? `+${S.recFresh.size}` : "";
  fresh.classList.toggle("hidden", !S.recFresh.size);

  if (!items.length) {
    const txt = S.state === "running"
      ? (S.coverage && S.coverage.iteration > 0 ? "All topics covered 🎉" : "Waiting for the first analysis…")
      : "They appear after the first analysis.";
    box.innerHTML = `<div class="placeholder">${txt}</div>`;
    return;
  }
  box.innerHTML = items.map(recCardHtml).join("");

  box.querySelectorAll("[data-reccopy]").forEach((b) => b.addEventListener("click", (e) => {
    e.stopPropagation();
    const r = items[+b.dataset.reccopy];
    copyText(r.suggested_question || r.topic_question || "");
    b.textContent = "✓";
    setTimeout(() => { b.textContent = "⧉"; }, 1200);
  }));
  box.querySelectorAll("[data-rechide]").forEach((b) => b.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await api("/api/recommendations/dismiss", {
        method: "POST", body: JSON.stringify({ topic_id: b.dataset.rechide }),
      });
    } catch (err) { setStatus(err.message, true); }
  }));
  box.querySelectorAll("[data-goto]").forEach((el) => el.addEventListener("click", () => {
    S.curBlock = el.dataset.goto;
    renderLiveGuide();
    renderAnchorHint();
    const target = $("live-guide").querySelector(`[data-blk="${S.curBlock}"]`);
    // No smooth: in a conversation, landing at once beats watching an animation —
    // and smooth scrolling silently does nothing under prefers-reduced-motion.
    if (target) target.scrollIntoView({ block: "center" });
  }));
}

function renderLiveClock() {
  $("live-el").textContent = (S.state === "running" && S.startedAt)
    ? fmtT(Date.now() / 1000 - S.startedAt) : "";
}

function renderLive() {
  renderLiveGuide();
  renderLiveRecs();
  renderComments();
  renderQuestions();
  renderAnchorHint();
  renderLiveClock();
  const running = S.state === "running";
  $("cmt-text").disabled = !running;
  $("q-text").disabled = !running;
  $("live-title").textContent = S.guideConfirmed
    ? S.guideConfirmed.title : (S.guideText ? "guide not parsed" : "no guide set");
}

async function sendComment() {
  const ta = $("cmt-text");
  const note = ta.value.trim();
  if (!note) return;
  const b = curBlock();
  ta.value = "";
  try {
    await api("/api/session/flag", {
      method: "POST",
      body: JSON.stringify({
        note,
        anchor: blockAnchor(b),
        anchor_section: b ? b.section : "",
        topic_id: b ? (b.topicIds[0] || null) : null,
      }),
    });
  } catch (e) {
    ta.value = note;   // do not lose what was typed: the session may not have started
    setStatus(e.message, true);
  }
}

async function sendQuestion() {
  const ta = $("q-text");
  const text = ta.value.trim();
  if (!text) return;
  ta.value = "";
  try {
    await api("/api/session/question", { method: "POST", body: JSON.stringify({ text }) });
  } catch (e) {
    ta.value = text;
    setStatus(e.message, true);
  }
}

function applyGuideFontSize() {
  document.documentElement.style.setProperty("--gfs", S.gfs + "px");
  localStorage.setItem("ilh_gfs", String(S.gfs));
}

/* ---- colour theme ----
   "system" — set nothing and leave the choice to the media query in app.css:
   people often have the dark theme switch on a schedule, and the window should
   ride along with it. "light"/"dark" — an explicit choice, which survives a restart. */
function applyTheme() {
  const root = document.documentElement;
  if (S.theme === "system") delete root.dataset.theme;
  else root.dataset.theme = S.theme;
  try { localStorage.setItem("ilh_theme", S.theme); } catch (e) { /* private mode */ }
  for (const b of $("theme-switch").children) {
    b.classList.toggle("on", b.dataset.theme === S.theme);
  }
}

/* ---- transcript (segments plus flags) ---- */

function segmentHtml(s) {
  const isI = s.speaker === "INTERVIEWER";
  return `<div class="seg">
    <span class="spk ${isI ? "i" : "r"}">${isI ? "I" : "R"}</span>
    <span class="seg-t">${fmtT(s.t0)}</span>
    <span class="seg-x">${esc(s.text)}</span>
  </div>`;
}

function flagHtml(f) {
  return `<div class="seg flag-row">
    <span class="spk">🚩</span>
    <span class="seg-t">${fmtT(f.t)}</span>
    <span class="seg-x">${esc(f.note || "flagged moment")}</span>
  </div>`;
}

function renderTranscript() {
  const box = $("transcript-list");
  if (!S.segments.length && !S.flags.length) {
    box.innerHTML = `<div class="placeholder">The transcript appears once the session starts.</div>`;
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

/* ---------------------------------------------------------------- setup */

function renderGuideBadge() {
  const badge = $("guide-badge");
  if (S.guideConfirmed) {
    const n = S.guideConfirmed.sections.reduce((a, s) => a + s.topics.length, 0);
    badge.innerHTML = `<span class="ok">✓ «${esc(S.guideConfirmed.title)}», topics: ${n}</span>`;
  } else {
    badge.textContent = "not set";
  }
}

async function loadDevices() {
  try {
    const data = await api("/api/devices");
    S.devices = data.devices;
    // preferFns are tried in order: the first one that matches makes the choice.
    const mkOptions = (sel, preferFns) => {
      const cur = sel.value;
      sel.innerHTML = `<option value="">— choose a device —</option>` +
        S.devices.map((d) =>
          `<option value="${d.index}">${esc(d.name)} (${esc(d.hostapi)})</option>`).join("");
      if (cur && S.devices.some((d) => String(d.index) === cur)) sel.value = cur;
      else {
        for (const fn of preferFns) {
          const pref = S.devices.find(fn);
          if (pref) { sel.value = String(pref.index); break; }
        }
      }
    };
    mkOptions($("sel-mic"), [(d) => d.is_default_input && !d.is_loopback]);
    // Loopback of the current output device is the cable-free route, so it comes first.
    mkOptions($("sel-sys"), [
      (d) => d.is_default_loopback,
      (d) => d.is_loopback,
      (d) => /blackhole|cable|vb-audio|virtual/i.test(d.name),
    ]);
    const hasLoopback = S.devices.some((d) => d.is_loopback);
    $("sys-lbl").textContent = hasLoopback
      ? "System audio (speakers / headphones) → respondent"
      : "System audio (BlackHole / VB-Cable) → respondent";
    $("vu-hint").textContent = hasLoopback
      ? "Say something and let Zoom play sound — both bars should move. System audio is taken straight from the output device, no cable needed."
      : "Say something and let Zoom play sound — both bars should move.";
    if (data.error) appMsg(data.error);
  } catch (e) {
    appMsg("Could not get the device list: " + e.message);
  }
  renderHeader();
  syncMonitor(true);
}

async function checkLLM() {
  const el = $("llm-status");
  el.textContent = "checking…";
  try {
    const st = await api("/api/llm/status");
    S.llmOk = st.ok;
    el.innerHTML = st.ok
      ? `<span class="ok">✓ Ollama ${esc(st.version || "")}, model ${esc(st.model)}</span>`
      : `<span style="color:var(--high)">✗ ${esc(st.error || "unavailable")}</span>`;
  } catch (e) {
    S.llmOk = false;
    el.textContent = "✗ the server is unavailable";
  }
  renderHeader();
}

/* ---- the model-download wizard ---- */

const MB = 1024 * 1024;
const fmtMB = (b) => (b >= 1024 * MB ? (b / 1024 / MB).toFixed(1) + " GB"
                                     : Math.round(b / MB) + " MB");

async function loadSetupStatus() {
  try {
    renderSetup(await api("/api/setup/status"));
  } catch (e) {
    $("models-status").textContent = "the check failed";
  }
}

function renderSetup(p) {
  S.setup = p;
  const items = p.items || {};
  const order = ["asr", "llm"];
  const list = order.filter((k) => items[k]).map((k) => items[k]);

  $("wiz-items").innerHTML = list.map((it) => {
    const pct = it.total_bytes ? Math.min(100, Math.round(100 * it.done_bytes / it.total_bytes)) : null;
    let right = esc(it.message);
    if (it.state === "ok") right = `<span class="ok">✓ ${esc(it.message)}</span>`;
    if (it.state === "error") right = `<span class="err">✗ ${esc(it.message)}</span>`;
    if (it.state === "blocked") right = `<span class="err">${esc(it.message)}</span>`;
    if (it.state === "downloading") {
      // Ollama gives exact byte counts, HuggingFace only the amount downloaded.
      right = pct !== null ? `${pct}% — ${fmtMB(it.done_bytes)} of ${fmtMB(it.total_bytes)}`
                           : (it.done_bytes ? fmtMB(it.done_bytes) : esc(it.message));
    }
    const bar = it.state === "downloading"
      ? `<div class="wiz-bar"><div class="wiz-fill${pct === null ? " pulse" : ""}"
           style="width:${pct === null ? 100 : pct}%"></div></div>`
      : "";
    return `<div class="wiz-item"><div class="wiz-row"><span>${esc(it.title)}</span>
      <span class="wiz-right">${right}</span></div>${bar}</div>`;
  }).join("");

  const missing = list.filter((it) => it.state === "missing");
  const blocked = list.filter((it) => it.state === "blocked");
  const busy = !!p.running;
  const getBtn = $("btn-wiz-get");
  getBtn.disabled = busy || missing.length === 0;
  getBtn.textContent = busy ? "Downloading…" : "Download";
  $("btn-wiz-close").textContent = busy ? "Minimise" : (missing.length || blocked.length ? "Later" : "Done");

  const short = list.every((it) => it.state === "ok") ? "all present"
    : busy ? "downloading…"
    : `missing: ${[...missing, ...blocked].map((it) => it.title.toLowerCase()).join(", ")}`;
  $("models-status").textContent = short;
  $("models-status").classList.toggle("dim", list.every((it) => it.state === "ok"));

  if (!S.wizardHidden && (missing.length || blocked.length || busy)) showWizard(true);
  if (list.every((it) => it.state === "ok") && !busy) showWizard(false);
}

function showWizard(on) {
  $("wizard").classList.toggle("hidden", !on);
}

/** A message about the project's own settings: the guide, the length, the vocabulary. */
function setupMsg(text, ok = false) {
  const el = $("setup-msg");
  el.textContent = text || "";
  el.style.color = ok ? "var(--covered)" : "var(--high)";
}

/** A message about the machine: devices, models, the LLM. It belongs in the ⚙ dialog —
 *  the project settings tab may not even be open when the device list fails. */
function appMsg(text, ok = false) {
  const el = $("app-msg");
  el.textContent = text || "";
  el.style.color = ok ? "var(--covered)" : "var(--high)";
}

/* ---- application settings: everything that belongs to the machine ----
   Two settings menus, and the split is by owner. Here — the theme, the audio
   devices, the models, the shared analysis prompts: one machine, one answer.
   In the «Project settings» tab — the length, the guide, the extra instructions:
   they change with the study. */

const appSettingsOpen = () => !$("app-settings").classList.contains("hidden");

function showAppSettings(on) {
  $("app-settings").classList.toggle("hidden", !on);
  if (on) appMsg("");
  // The VU meters live in this dialog: the monitor follows it, not a tab.
  syncMonitor(true);
}

/* ---- "are you sure?" ---- */

let confirmResolve = null;

/** A modal question with one dangerous answer. Returns a promise of true/false. */
function askConfirm({ title, body = "", yes = "Delete", danger = true }) {
  closeConfirm(false);   // a second question replaces the first, it does not stack
  $("confirm-title").textContent = title;
  $("confirm-body").textContent = body;
  const btn = $("btn-confirm-yes");
  btn.textContent = yes;
  btn.classList.toggle("danger", danger);
  btn.classList.toggle("primary", !danger);
  $("confirm").classList.remove("hidden");
  btn.focus();
  return new Promise((resolve) => { confirmResolve = resolve; });
}

function closeConfirm(answer) {
  if (!confirmResolve) return;
  const resolve = confirmResolve;
  confirmResolve = null;
  $("confirm").classList.add("hidden");
  resolve(answer);
}

/* ---- level monitor (before the session starts) ---- */

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
  // The monitor is wanted when there is no session, the ⚙ dialog with the VU
  // meters is open, and at least one device is chosen.
  clearTimeout(monitorTimer);
  monitorTimer = setTimeout(async () => {
    const mic = $("sel-mic").value, sys = $("sel-sys").value;
    const want = S.state === "idle" && appSettingsOpen() && (mic !== "" || sys !== "");
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
    } catch (e) { /* the monitor is auxiliary — stay quiet */ }
  }, 200);
}

/* ---- projects ---- */

async function loadProjects() {
  try {
    const data = await api("/api/projects");
    S.projects = data.projects || [];
  } catch (e) {
    S.projects = [];
  }
  renderProjectGrid();
  renderWorkTitle();
}

/* ---- screens: the project list ↔ work on a single project ---- */

function showScreen(name) {
  S.screen = name;
  $("screen-start").classList.toggle("hidden", name !== "start");
  $("screen-work").classList.toggle("hidden", name !== "work");
  if (name === "start") { renderReady(); renderProjectGrid(); }
  else renderWorkTitle();
  syncMonitor();
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  syncMonitor();
}

function renderWorkTitle() {
  const p = S.projects.find((x) => x.project_id === S.projectId);
  $("work-title").textContent = p ? p.title : (S.projectId || "No project");
}

/** Open a project and move to the work screen. Without a guide there is nothing
 *  to work with, so we go straight to its settings rather than an empty interview screen. */
async function openProject(id) {
  startMsg("");
  await selectProject(id);
  showScreen("work");
  showTab(S.guideConfirmed ? "live" : "setup");
}

function startMsg(text, ok = false) {
  const el = $("start-msg");
  el.textContent = text || "";
  el.classList.toggle("ok", ok);
}

/** The readiness line: audio should be fixed before the call, not a minute before it. */
function renderReady() {
  const items = (S.setup || {}).items || {};
  const missing = Object.values(items).filter((it) => it.state === "missing").length;
  const checks = [
    { name: "Microphone", ok: $("sel-mic").value !== "", why: "no microphone selected" },
    { name: "System audio", ok: $("sel-sys").value !== "",
      why: "no system audio selected — the other person's voice will not be recorded" },
    { name: "Recognition", ok: !S.setup || missing === 0, why: "recognition models are missing" },
    { name: "Hints", ok: S.llmOk, why: "the LLM is not answering — there will be no hints" },
  ];
  const bad = checks.filter((c) => !c.ok);
  const el = $("ready-line");
  el.classList.toggle("alarm", bad.length > 0);
  el.innerHTML = checks.map((c) =>
    `<span class="chk${c.ok ? "" : " bad"}"><i class="dot"></i>${esc(c.name)}</span>`).join("")
    + (bad.length
      ? `<button class="btn tiny" id="btn-fix">Fix</button>
         <span class="note">${esc(bad.map((c) => c.why).join("; "))}.</span>`
      : "");
  // Everything on this line — devices, models, the LLM — is fixed in the ⚙ dialog.
  if (bad.length) $("btn-fix").addEventListener("click", () => showAppSettings(true));
}

/** A session folder is named 2026-08-08_10-15-00. Show it the human way. */
function fmtSessionDay(name) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(name || "");
  if (!m) return null;
  const d = new Date(+m[1], +m[2] - 1, +m[3]);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const days = Math.round((today - d) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  return d.toLocaleDateString("en-GB", d.getFullYear() === today.getFullYear()
    ? { day: "numeric", month: "short" }
    : { day: "numeric", month: "short", year: "numeric" });
}

function projectCardHtml(p) {
  const day = fmtSessionDay(p.last_session_at);
  const meta = p.sessions_count
    ? `<b>${p.sessions_count} interviews</b> · last one ${esc(day || "—")}`
    : "No interviews yet";
  return `<div class="card" data-id="${esc(p.project_id)}">
    <button class="card-more" data-more title="More">⋯</button>
    <div class="card-title">${esc(p.title)}</div>
    <div class="card-meta${p.sessions_count ? "" : " empty"}">${meta}</div>
    ${p.duration_min ? `<div class="card-foot"><span class="chip">${p.duration_min} min</span></div>` : ""}
  </div>`;
}

function renderProjectGrid() {
  const box = $("project-grid");
  const q = S.projectQuery.trim().toLowerCase();
  const list = q ? S.projects.filter((p) => (p.title || "").toLowerCase().includes(q)) : S.projects;
  $("start-count").textContent = !S.projects.length ? ""
    : (q ? `Found — ${list.length}` : `Projects — ${S.projects.length}`);

  const empty = !S.projects.length
    ? `<div class="placeholder">No projects yet.<br>
        A project is a series of interviews run off one guide: it remembers the
        guide, the vocabulary and the length, and keeps the recordings in one folder.</div>`
    : "";
  box.innerHTML = empty + list.map(projectCardHtml).join("")
    + `<div class="card new" id="card-new">＋ New project</div>`;

  box.querySelectorAll(".card[data-id]").forEach((card) => {
    card.addEventListener("click", (e) => {
      if (e.target.closest(".card-menu, [data-more], .card-rename")) return;
      openProject(card.dataset.id);
    });
    card.querySelector("[data-more]").addEventListener("click", (e) => {
      e.stopPropagation();
      toggleCardMenu(card);
    });
  });
  $("card-new").addEventListener("click", () => {
    $("proj-new").classList.remove("hidden");
    $("proj-title").focus();
  });
}

function closeCardMenus() {
  document.querySelectorAll(".card-menu").forEach((m) => m.remove());
}

function toggleCardMenu(card) {
  const open = card.querySelector(".card-menu");
  closeCardMenus();
  if (open) return;
  const id = card.dataset.id;
  const menu = document.createElement("div");
  menu.className = "card-menu";
  menu.innerHTML = `<button data-act="open">Open the project folder</button>
    <button data-act="rename">Rename</button>
    <hr>
    <button class="danger" data-act="delete">Delete</button>`;
  card.appendChild(menu);
  menu.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    e.stopPropagation();
    if (btn.dataset.act === "open") { closeCardMenus(); await openProjectFolder(null, id); }
    else if (btn.dataset.act === "rename") { closeCardMenus(); startRename(card, id); }
    // Deletion takes two clicks: the item sits right next to "open the folder".
    else if (btn.dataset.sure) { closeCardMenus(); await deleteProject(id); }
    else { btn.dataset.sure = "1"; btn.textContent = "Really delete?"; }
  });
}

function startRename(card, id) {
  const title = card.querySelector(".card-title");
  const was = title.textContent;
  const input = document.createElement("input");
  input.type = "text";
  input.className = "card-rename";
  input.value = was;
  title.replaceWith(input);
  input.focus();
  input.select();
  input.addEventListener("click", (e) => e.stopPropagation());
  input.addEventListener("blur", () => renderProjectGrid());
  input.addEventListener("keydown", async (e) => {
    e.stopPropagation();
    if (e.key === "Escape") { renderProjectGrid(); return; }
    if (e.key !== "Enter") return;
    const value = input.value.trim();
    if (!value || value === was) { renderProjectGrid(); return; }
    try {
      await api(`/api/projects/${encodeURIComponent(id)}`, {
        method: "PATCH", body: JSON.stringify({ title: value }),
      });
      await loadProjects();
    } catch (err) {
      startMsg("Could not rename: " + err.message);
      renderProjectGrid();
    }
  });
}

async function deleteProject(id) {
  try {
    await api(`/api/projects/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (S.projectId === id) await selectProject("");
    startMsg("");
    await loadProjects();
  } catch (e) {
    // The store refuses to delete a project with recordings — show its refusal verbatim.
    startMsg(e.message);
  }
}

/** The preset catches up with edits: the guide is confirmed inside an already
 *  open project, and the project must remember it for the next interview. */
async function saveProjectPreset(fields) {
  if (!S.projectId) return false;
  try {
    await api(`/api/projects/${encodeURIComponent(S.projectId)}`, {
      method: "PATCH", body: JSON.stringify(fields),
    });
    return true;
  } catch (e) {
    setupMsg("The project did not remember the setting: " + e.message);
    return false;
  }
}

/** Apply the project preset to the setup. Empty project fields do not wipe what
 *  the researcher has already typed by hand. */
function applyPreset(project) {
  if (project.guide) {
    S.guideConfirmed = project.guide;
    renderGuideBadge();
    renderTopics();
    renderHeader();
  }
  if (project.guide_text) {
    $("guide-text").value = project.guide_text;
    setGuideText(project.guide_text);
  } else if (project.guide) {
    setGuideText(S.guideText);  // the sections are known — rebuild the split
  }
  if (project.asr_vocabulary) $("vocab").value = project.asr_vocabulary;
  // The instructions belong to the project entirely: empty means "there are
  // none", not "leave alone", or they would leak into the next project opened.
  setInstructions(project.llm_instructions || "");
  if (project.duration_min) {
    const dur = $("duration");
    dur.value = project.duration_min;
    // The project value is a deliberate choice: a snapshot must not undo it.
    dur.dataset.touched = "1";
  }
}

/** The project's extra instructions: the field plus its status caption. Outside
 *  a project there is nowhere to write them — disable it so text is not lost silently. */
function setInstructions(text, state = "") {
  const ta = $("proj-instr");
  ta.value = text || "";
  ta.disabled = !S.projectId;
  ta.dataset.saved = ta.value;
  $("proj-instr-state").textContent =
    S.projectId ? state : "available inside a project";
}

async function selectProject(id, { preset = true } = {}) {
  S.projectId = id;
  localStorage.setItem("ilh_project", id);
  if (id && preset) {
    try {
      applyPreset(await api(`/api/projects/${encodeURIComponent(id)}`));
    } catch (e) {
      setupMsg("Could not open the project: " + e.message);
    }
  } else if (!id) {
    setInstructions("");
  }
  await loadProjectData();
}

async function loadProjectData() {
  if (!S.projectId) {
    S.projectSessions = [];
    S.projectCoverage = null;
    S.openSession = null;
    renderProject();
    return;
  }
  try {
    const [sessions, coverage] = await Promise.all([
      api(`/api/projects/${encodeURIComponent(S.projectId)}/sessions`),
      api(`/api/projects/${encodeURIComponent(S.projectId)}/coverage`),
    ]);
    S.projectSessions = sessions.sessions || [];
    S.projectCoverage = coverage;
    // The project changed — the expanded row belongs elsewhere, so close it.
    if (!S.projectSessions.some((s) => s.session_id === S.openSession)) S.openSession = null;
  } catch (e) {
    S.projectSessions = [];
    S.projectCoverage = null;
    S.openSession = null;
  }
  renderProject();
}

async function createProject() {
  const title = $("proj-title").value.trim();
  if (!title) { startMsg("Enter a project name."); return; }
  try {
    const project = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({
        title,
        guide: S.guideConfirmed,
        guide_text: $("guide-text").value,
        asr_vocabulary: $("vocab").value.trim(),
        duration_min: +$("duration").value || null,
      }),
    });
    $("proj-title").value = "";
    $("proj-new").classList.add("hidden");
    await loadProjects();
    // Created means they intend to work: open it at once, without a second click.
    await openProject(project.project_id);
  } catch (e) {
    startMsg("Could not create the project: " + e.message);
  }
}

async function openProjectFolder(sessionId = null, projectId = null) {
  const id = projectId || S.projectId;
  if (!id) { setupMsg("Choose a project first."); return; }
  try {
    await api(`/api/projects/${encodeURIComponent(id)}/open`, {
      method: "POST", body: JSON.stringify({ session_id: sessionId }),
    });
  } catch (e) {
    startMsg("Could not open the folder: " + e.message);
    setupMsg("Could not open the folder: " + e.message);
  }
}

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? String(iso).slice(0, 16).replace("T", " ")
    : d.toLocaleString("en-GB", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function sessionRowHtml(s) {
  const c = s.counts;
  const cov = c ? `${c.covered}/${c.total}` : "—";
  const mins = s.duration_s ? `${Math.round(s.duration_s / 60)} min` : "—";
  const live = S.state === "running" && s.session_id === S.sessionId;
  const open = S.openSession === s.session_id;
  return `<div class="sess-row${open ? " open" : ""}" data-session="${esc(s.session_id)}"
      title="Show the interview notes">
    <span class="sess-caret">${open ? "▾" : "▸"}</span>
    <span class="sess-t">${esc(fmtDate(s.started_at))}</span>
    <span class="grow">${live ? "<b>running now</b>" : esc(mins)}
      <span class="dim">· ${s.segments == null ? "—" : s.segments} utterances</span></span>
    <span class="sess-cov" title="Topics covered">${cov}</span>
    <button class="copy" data-act="view"
      title="Open the transcript${s.has_audio ? " and the recording" : ""}">${s.has_audio ? "🎧" : "📄"}</button>
    <button class="copy" data-act="folder" title="Open the interview folder">📂</button>
  </div>` + (open ? sessionNotesHtml(s.session_id) : "");
}

/** Notes of the expanded interview: timecoded flags and the "ask" questions. */
function sessionNotesHtml(sessionId) {
  const d = S.sessionNotes[sessionId];
  if (!d) return `<div class="sess-notes"><div class="hint">Loading…</div></div>`;
  if (d.error) return `<div class="sess-notes"><div class="hint">${esc(d.error)}</div></div>`;

  const flags = (d.flags || []).map((f) => {
    const note = (f.note || "").trim();
    const where = [f.anchor_section, f.anchor].filter(Boolean).join(" → ");
    return `<div class="note-row">
      <span class="note-t">${fmtT(f.t || 0)}</span>
      <span class="grow">${note ? esc(note) : "<span class='dim'>bookmark with no text</span>"}
        ${where ? `<div class="note-where">on question: ${esc(where)}</div>` : ""}</span>
    </div>`;
  }).join("");

  const questions = (d.questions || []).map((q) => `<div class="note-row${q.done ? " done" : ""}">
      <span class="note-t">${fmtT(q.t || 0)}</span>
      <span class="grow">${q.done ? "✓ " : ""}${esc(q.text || "")}</span>
    </div>`).join("");

  if (!flags && !questions) {
    return `<div class="sess-notes"><div class="hint">No notes were left in this interview —
      the transcript and the report are in the folder (📂).</div></div>`;
  }
  return `<div class="sess-notes">
    ${flags ? `<div class="notes-title">Flags and comments</div>${flags}` : ""}
    ${questions ? `<div class="notes-title">Ask</div>${questions}` : ""}
  </div>`;
}

async function toggleSessionNotes(sessionId) {
  if (S.openSession === sessionId) { S.openSession = null; renderProject(); return; }
  S.openSession = sessionId;
  renderProject();
  // Re-read on every expand: a running interview accumulates notes as it goes.
  try {
    S.sessionNotes[sessionId] = await api(
      `/api/projects/${encodeURIComponent(S.projectId)}/sessions/${encodeURIComponent(sessionId)}`);
  } catch (e) {
    S.sessionNotes[sessionId] = { error: "Could not read the notes: " + e.message };
  }
  if (S.openSession === sessionId) renderProject();
}

/** A row of the aggregate coverage: in how many interviews of the series a topic was covered. */
function coverageRowHtml(t) {
  const seen = t.sessions_with_topic;
  const pct = (n) => (seen ? (n / seen) * 100 : 0);
  const cold = seen && t.covered === 0;
  return `<div class="pcov-row">
    <div class="pcov-head">
      <span class="topic-q grow${cold ? " cold" : ""}">${esc(t.question)}</span>
      <span class="pcov-n${cold ? " cold" : ""}">${t.covered}/${seen}</span>
    </div>
    <div class="pbar">
      <span class="pb covered" style="width:${pct(t.covered)}%"></span>
      <span class="pb partial" style="width:${pct(t.partial)}%"></span>
      <span class="pb notcov" style="width:${pct(t.not_covered)}%"></span>
    </div>
  </div>`;
}

function renderProject() {
  const head = $("proj-head"), sessions = $("proj-sessions"), coverage = $("proj-coverage");
  if (!S.projectId) {
    head.innerHTML = `<div class="placeholder">This interview runs outside a project —
      the recording goes into the shared sessions folder.<br>
      The «←» button in the header takes you back to the project list.</div>`;
    sessions.innerHTML = coverage.innerHTML = "";
    return;
  }
  const meta = S.projects.find((p) => p.project_id === S.projectId) || {};
  const cov = S.projectCoverage;
  head.innerHTML = `<div class="proj-title">${esc(meta.title || S.projectId)}</div>
    <div class="hint">${esc(meta.guide_title || "no guide set")} ·
      ${S.projectSessions.length} interviews in the series</div>`;

  sessions.innerHTML = S.projectSessions.length
    ? `<div class="sec-title">Interviews in the series</div>` + S.projectSessions.map(sessionRowHtml).join("")
    : `<div class="hint">No interviews recorded yet — set the guide up and press «Start».</div>`;
  sessions.querySelectorAll(".sess-row .copy").forEach((btn) => btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const id = btn.closest(".sess-row").dataset.session;
    if (btn.dataset.act === "view") openViewer(id);
    else openProjectFolder(id);
  }));
  sessions.querySelectorAll(".sess-row").forEach((row) => row.addEventListener("click", () => {
    toggleSessionNotes(row.dataset.session);
  }));

  // Until the series has a single interview, the summary is a column of zeros: noise.
  if (!cov || !cov.sections.length || !cov.sessions_count) {
    coverage.innerHTML = "";
    return;
  }
  coverage.innerHTML = `<div class="sec-title">Coverage across the series
      <span class="dim">(in how many interviews a topic was covered)</span></div>` +
    cov.sections.map((sec) => `<div class="pcov-sec">${esc(sec.title)}</div>` +
      sec.topics.map(coverageRowHtml).join("")).join("");
}

/* ---- a past interview: the transcript next to the recording ----
   The folder holds transcript.md, but reading it there loses the timecodes.
   Here a line is a control: click it and the recording moves to that moment. */

const sessionUrl = (sessionId, tail) =>
  `/api/projects/${encodeURIComponent(S.projectId)}/sessions/${encodeURIComponent(sessionId)}/${tail}`;

/** Utterances and flags in one timeline, the way the interview actually went. */
function viewerRows(data) {
  const rows = (data.segments || []).map((s) => {
    const isI = s.speaker === "INTERVIEWER";
    return {
      t: s.t0 || 0,
      flag: false,
      html: `<span class="spk ${isI ? "i" : "r"}">${isI ? "I" : "R"}</span>
        <span class="seg-t">${fmtT(s.t0 || 0)}</span>
        <span class="seg-x">${esc(s.text || "")}</span>`,
    };
  }).concat((data.flags || []).map((f) => ({
    t: f.t || 0,
    flag: true,
    html: `<span class="spk">🚩</span>
      <span class="seg-t">${fmtT(f.t || 0)}</span>
      <span class="seg-x">${esc((f.note || "").trim() || "flagged moment")}</span>`,
  })));
  rows.sort((a, b) => a.t - b.t);
  return rows;
}

function renderViewer() {
  const d = S.viewer;
  if (!d) return;
  const box = $("viewer-body");
  if (d.error) { box.innerHTML = `<div class="placeholder">${esc(d.error)}</div>`; return; }
  if (!d.rows) { box.innerHTML = `<div class="placeholder">Loading…</div>`; return; }
  if (!d.rows.length) {
    box.innerHTML = `<div class="placeholder">Nothing was recognised in this interview.
      ${d.hasAudio ? "The recording is above — it can still be listened to." : ""}</div>`;
    return;
  }
  box.innerHTML = d.rows.map((r, i) =>
    `<div class="seg${r.flag ? " flag-row" : ""}${d.hasAudio ? " seekable" : ""}"
       data-i="${i}"${d.hasAudio ? ' title="Play from here"' : ""}>${r.html}</div>`).join("");
}

/** Follow the playhead: the line being spoken is highlighted and kept in view. */
function highlightViewerRow() {
  const d = S.viewer;
  if (!d || !d.rows || !d.rows.length) return;
  const t = $("viewer-player").currentTime || 0;
  let idx = -1;
  for (let i = 0; i < d.rows.length && d.rows[i].t <= t + 0.05; i++) idx = i;
  if (idx === d.cur) return;
  d.cur = idx;
  const box = $("viewer-body");
  box.querySelectorAll(".seg.cur").forEach((el) => el.classList.remove("cur"));
  const el = idx < 0 ? null : box.querySelector(`.seg[data-i="${idx}"]`);
  if (!el) return;
  el.classList.add("cur");
  el.scrollIntoView({ block: "nearest" });
}

async function openViewer(sessionId) {
  if (!S.projectId) return;
  const meta = S.projectSessions.find((s) => s.session_id === sessionId) || {};
  S.viewer = { sessionId, rows: null, hasAudio: false, cur: -1 };
  $("viewer-title").textContent = `Interview of ${fmtDate(meta.started_at)}`;
  $("viewer-sub").textContent = "";
  $("viewer-audio-note").textContent = "";
  $("viewer-player").classList.add("hidden");
  stopViewerAudio();
  $("viewer").classList.remove("hidden");
  renderViewer();

  let data;
  try {
    data = await api(sessionUrl(sessionId, "transcript"));
  } catch (e) {
    if (S.viewer && S.viewer.sessionId === sessionId) {
      S.viewer.error = "Could not read the transcript: " + e.message;
      renderViewer();
    }
    return;
  }
  // While the request was in flight the viewer may have been closed or moved on.
  if (!S.viewer || S.viewer.sessionId !== sessionId) return;

  S.viewer.rows = viewerRows(data);
  S.viewer.hasAudio = !!data.has_audio;
  $("viewer-sub").textContent = [
    fmtDate(data.started_at),
    data.duration_s ? `${Math.round(data.duration_s / 60)} min` : null,
    `${(data.segments || []).length} utterances`,
  ].filter(Boolean).join(" · ");

  const player = $("viewer-player");
  player.classList.toggle("hidden", !data.has_audio);
  $("viewer-audio-note").textContent = data.has_audio
    ? "Left channel — you, right — the respondent. Click a line to play from it."
    : "This interview has no recording: audio saving was off at the time.";
  if (data.has_audio) player.src = sessionUrl(sessionId, "audio");
  renderViewer();
}

/** Let go of the file: during a live interview it is still being written to. */
function stopViewerAudio() {
  const player = $("viewer-player");
  player.pause();
  player.removeAttribute("src");
  player.load();
}

function closeViewer() {
  if (!S.viewer) return;
  S.viewer = null;
  stopViewerAudio();
  $("viewer").classList.add("hidden");
}

/* ---- guide library ---- */

async function loadLibrary() {
  try {
    const data = await api("/api/guides");
    S.libGuides = data.guides || [];
    const sel = $("sel-guide-lib");
    sel.innerHTML = S.libGuides.length
      ? S.libGuides.map((g) =>
          `<option value="${esc(g.file_id)}">${esc(g.title)} (${g.topics_count} topics)</option>`).join("")
      : `<option value="">— empty —</option>`;
  } catch (e) { /* the library is not critical */ }
}

async function libLoad() {
  const id = $("sel-guide-lib").value;
  if (!id) return;
  try {
    const data = await api(`/api/guides/${encodeURIComponent(id)}`);
    S.guideConfirmed = data.guide;
    if (data.source_text) $("guide-text").value = data.source_text;
    setGuideText($("guide-text").value);
    $("guide-preview").classList.add("hidden");
    $("guide-editor").classList.remove("hidden");
    renderGuideBadge();
    renderTopics();
    renderHeader();
    setupMsg(`Guide «${data.guide.title}» loaded from the library.`, true);
  } catch (e) {
    setupMsg("Could not load the guide: " + e.message);
  }
}

/** Deleting a guide is irreversible and the ✕ sits next to «Load» — ask first. */
async function libDelete() {
  const id = $("sel-guide-lib").value;
  if (!id) return;
  const guide = S.libGuides.find((g) => g.file_id === id);
  const title = guide ? guide.title : id;
  const ok = await askConfirm({
    title: `Delete the guide «${title}»?`,
    body: "It goes from the library for good. Projects and interviews that already "
      + "use this guide keep their own copy of it — only the library entry disappears.",
  });
  if (!ok) return;
  try {
    await api(`/api/guides/${encodeURIComponent(id)}`, { method: "DELETE" });
    await loadLibrary();
    setupMsg(`Guide «${title}» deleted from the library.`, true);
  } catch (e) {
    setupMsg("Could not delete the guide: " + e.message);
  }
}

/* ---- guide parsing ---- */

function showGuidePreview(guide) {
  $("guide-tree").innerHTML = guide.sections.map((sec) => `
    <div class="g-sec">${esc(sec.title)}</div>
    ${sec.topics.map((t) => `<div class="g-topic">• ${esc(t.question)}</div>`).join("")}`).join("");
  $("guide-editor").classList.add("hidden");
  $("guide-preview").classList.remove("hidden");
}

async function parseGuide() {
  const text = $("guide-text").value.trim();
  if (!text) { setupMsg("Paste the guide text."); return; }
  const btn = $("btn-parse");
  btn.disabled = true;
  btn.textContent = "Parsing (LLM)…";
  setupMsg("");
  try {
    S.guideParsed = await api("/api/guide/parse", { method: "POST", body: JSON.stringify({ text }) });
    showGuidePreview(S.guideParsed);
  } catch (e) {
    setupMsg("Could not parse the guide: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Parse the guide";
  }
}

async function confirmGuide() {
  S.guideConfirmed = S.guideParsed;
  renderGuideBadge();
  renderTopics();
  renderHeader();
  setGuideText($("guide-text").value);  // the section titles are known now
  try {
    await api("/api/guides", {
      method: "POST",
      body: JSON.stringify({ guide: S.guideConfirmed, source_text: $("guide-text").value }),
    });
    await loadLibrary();
    setupMsg("Structure confirmed and saved to the library. Choose the devices and press «Start».", true);
  } catch (e) {
    setupMsg("Structure confirmed (it was not saved to the library: " + e.message + ")", true);
  }
  await saveProjectPreset({ guide: S.guideConfirmed, guide_text: $("guide-text").value });
}

/* -------------------------------------------------------------- actions */

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
          asr_vocabulary: $("vocab").value.trim(),
          project_id: S.projectId || null,
          guide_text: S.guideText || $("guide-text").value,
        }),
      });
      S.monitorOn = false; // the server stopped the monitor itself
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
  const b = curBlock();
  try {
    await api("/api/session/flag", {
      method: "POST",
      body: JSON.stringify({
        note: "",
        anchor: blockAnchor(b),
        anchor_section: b ? b.section : "",
        topic_id: b ? (b.topicIds[0] || null) : null,
      }),
    });
  } catch (e) { setStatus(e.message, true); }
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

/* ------------------------------------------------------ analysis prompts */

/** Loaded on first expand: the texts are long and rarely opened. */
async function loadPrompts() {
  if (S.prompts) return;
  const box = $("prompts-list");
  box.textContent = "Loading…";
  try {
    S.prompts = (await api("/api/prompts")).prompts || [];
    box.innerHTML = S.prompts.map(promptCardHtml).join("");
  } catch (e) {
    box.textContent = "Could not load the prompts: " + e.message;
  }
}

function promptCardHtml(p) {
  const chips = (p.placeholders || [])
    .map((ph) => `<code title="${esc(ph.hint)}">{${esc(ph.name)}}</code>`)
    .join(" ");
  return `<div class="prompt" data-key="${esc(p.key)}">
  <div class="row"><span class="lbl">${esc(p.title)}</span><span class="grow"></span>
    <span class="dim js-badge">${p.customized ? "edited" : ""}</span></div>
  <div class="hint">${esc(p.hint)}</div>
  ${chips ? `<div class="hint">Substitutions: ${chips} — these must not be removed.</div>` : ""}
  <textarea class="mono" rows="12" spellcheck="false">${esc(p.text)}</textarea>
  <div class="btnrow">
    <button class="btn tiny" data-act="save">Save</button>
    <button class="btn tiny" data-act="reset"${p.customized ? "" : " disabled"}>Restore the default</button>
    <span class="js-msg hint"></span>
  </div>
</div>`;
}

function promptMsg(card, text, ok) {
  const el = card.querySelector(".js-msg");
  el.textContent = text || "";
  el.style.color = ok ? "var(--covered)" : "var(--high)";
}

/** The card after the server responds: the "edited" badge and whether reset is available. */
function syncPromptCard(card, item) {
  card.querySelector(".js-badge").textContent = item.customized ? "edited" : "";
  card.querySelector('button[data-act="reset"]').disabled = !item.customized;
}

async function onPromptAction(btn) {
  const card = btn.closest(".prompt");
  const key = card.dataset.key;
  const ta = card.querySelector("textarea");
  const item = S.prompts.find((p) => p.key === key);
  if (!item) return;
  promptMsg(card, "", true);
  try {
    if (btn.dataset.act === "save") {
      const res = await api(`/api/prompts/${encodeURIComponent(key)}`, {
        method: "PUT", body: JSON.stringify({ text: ta.value }),
      });
      item.text = res.text;
      item.customized = res.customized;
      ta.value = res.text;
      promptMsg(card, "Saved — from the next session on", true);
    } else {
      await api(`/api/prompts/${encodeURIComponent(key)}`, { method: "DELETE" });
      item.text = item.default;
      item.customized = false;
      ta.value = item.default;
      promptMsg(card, "Restored the default text", true);
    }
    syncPromptCard(card, item);
  } catch (e) {
    promptMsg(card, e.message, false);
  }
}

/* ---------------------------------------------------------------- init */

document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => showTab(tab.dataset.tab)));

/* ---- start screen ---- */
$("btn-back").addEventListener("click", async () => {
  showScreen("start");
  await loadProjects();   // the interview counters may have changed during the session
});
$("btn-settings").addEventListener("click", () => showAppSettings(true));
$("btn-app-settings").addEventListener("click", () => showAppSettings(true));
$("btn-app-close").addEventListener("click", () => showAppSettings(false));
$("btn-no-project").addEventListener("click", () => openProject(""));
$("proj-search").addEventListener("input", (e) => {
  S.projectQuery = e.target.value;
  renderProjectGrid();
});
document.addEventListener("click", closeCardMenus);

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
$("btn-models").addEventListener("click", async () => {
  S.wizardHidden = false;
  showWizard(true);
  await loadSetupStatus();
});
$("btn-wiz-get").addEventListener("click", async () => {
  const missing = Object.entries((S.setup || {}).items || {})
    .filter(([, it]) => it.state === "missing").map(([k]) => k);
  try {
    renderSetup(await api("/api/setup/download", {
      method: "POST", body: JSON.stringify({ items: missing }),
    }));
  } catch (e) {
    setStatus(e.message, true);
  }
});
$("btn-wiz-close").addEventListener("click", async () => {
  S.wizardHidden = true;
  showWizard(false);
  await checkLLM();   // the model may have arrived — recompute the header status
});
$("btn-lib-load").addEventListener("click", libLoad);
$("btn-lib-del").addEventListener("click", libDelete);
$("btn-proj-new").addEventListener("click", () => {
  $("proj-new").classList.toggle("hidden");
  $("proj-title").focus();
});
$("btn-proj-create").addEventListener("click", createProject);
$("btn-proj-cancel").addEventListener("click", () => {
  $("proj-title").value = "";
  $("proj-new").classList.add("hidden");
});
$("proj-title").addEventListener("keydown", (e) => { if (e.key === "Enter") createProject(); });
$("btn-proj-open").addEventListener("click", () => openProjectFolder());
$("sel-mic").addEventListener("change", () => { renderHeader(); syncMonitor(true); });
$("sel-sys").addEventListener("change", () => { renderHeader(); syncMonitor(true); });
$("duration").addEventListener("change", (e) => {
  e.target.dataset.touched = "1";
  saveProjectPreset({ duration_min: +e.target.value || null });
});
$("vocab").value = localStorage.getItem("ilh_vocab") || "";
$("vocab").addEventListener("change", (e) => {
  localStorage.setItem("ilh_vocab", e.target.value);
  saveProjectPreset({ asr_vocabulary: e.target.value.trim() });
});
$("proj-instr").addEventListener("blur", async (e) => {
  const ta = e.target;
  const text = ta.value.trim();
  if (!S.projectId || text === (ta.dataset.saved || "").trim()) return;
  const ok = await saveProjectPreset({ llm_instructions: text });
  if (!ok) {
    // The edit is not lost: the text stays in the field and the next blur retries.
    $("proj-instr-state").textContent = "not saved — try again";
    return;
  }
  ta.dataset.saved = ta.value;
  $("proj-instr-state").textContent = text ? "saved" : "no instructions";
});
$("prompts-box").addEventListener("toggle", (e) => {
  if (e.target.open) loadPrompts();
});
$("prompts-list").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-act]");
  if (btn) onPromptAction(btn);
});
$("theme-switch").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-theme]");
  if (!btn) return;
  S.theme = btn.dataset.theme;
  applyTheme();
});

/* ---- dialogs ---- */
/* Clicking the darkened area outside a sheet closes it — but only if the press
   started there too, or a text selection dragged out of the sheet would close it. */
for (const [id, close] of [["app-settings", () => showAppSettings(false)],
                           ["viewer", closeViewer],
                           ["confirm", () => closeConfirm(false)]]) {
  const modal = $(id);
  let onBackdrop = false;
  modal.addEventListener("mousedown", (e) => { onBackdrop = e.target === modal; });
  modal.addEventListener("click", (e) => { if (onBackdrop && e.target === modal) close(); });
}
$("btn-viewer-close").addEventListener("click", closeViewer);
$("btn-viewer-folder").addEventListener("click", () => {
  if (S.viewer) openProjectFolder(S.viewer.sessionId);
});
$("viewer-body").addEventListener("click", (e) => {
  const row = e.target.closest(".seg.seekable");
  if (!row || !S.viewer || !S.viewer.rows) return;
  const item = S.viewer.rows[+row.dataset.i];
  const player = $("viewer-player");
  if (!item || !player.src) return;
  player.currentTime = item.t;
  player.play().catch(() => { /* the browser may want a gesture first — the bar is there */ });
});
$("viewer-player").addEventListener("timeupdate", highlightViewerRow);
$("viewer-player").addEventListener("seeked", highlightViewerRow);
$("btn-confirm-yes").addEventListener("click", () => closeConfirm(true));
$("btn-confirm-no").addEventListener("click", () => closeConfirm(false));
/* Escape closes the topmost dialog: the question first, then the viewer, then ⚙. */
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (confirmResolve) closeConfirm(false);
  else if (S.viewer) closeViewer();
  else if (appSettingsOpen()) showAppSettings(false);
});

/* ---- interview screen ---- */
$("cmt-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendComment(); }
});
$("q-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
});
$("btn-gfs-p").addEventListener("click", () => { S.gfs = Math.min(40, S.gfs + 2); applyGuideFontSize(); });
$("btn-gfs-m").addEventListener("click", () => { S.gfs = Math.max(13, S.gfs - 2); applyGuideFontSize(); });
// The guide is edited in the project settings and read here — keep this screen up to date.
$("guide-text").addEventListener("input", (e) => setGuideText(e.target.value));
applyGuideFontSize();
applyTheme();
setInstructions("");   // until a project is chosen the field is disabled and captioned
renderLive();   // before the first snapshot the screen must already say something
showScreen("start");

/* F5 works from an input field too: the hand reaches for it without leaving the
   keyboard. The reload is intercepted only during a running session — outside
   one, let F5 refresh the page as usual. */
document.addEventListener("keydown", (e) => {
  if (e.key !== "F5" || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
  if (S.state !== "running") return;
  e.preventDefault();
  sendFlag();
});

connectWS();
loadDevices();
checkLLM();
loadSetupStatus();
loadLibrary();
loadProjects();
