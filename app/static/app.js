/* =========================================================
   Book RecSys frontend logic
   ========================================================= */

const API = '/api/v1';

/* ---------- DOM utilities ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function setBusy(btn, busy) {
  btn.disabled = busy;
  btn.querySelector('.btn-spinner').hidden = !busy;
}

function toast({ type = 'info', title, body = '' }) {
  const host = $('#toasts');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `
    <div class="t-title"></div>
    <div class="t-body"></div>`;
  el.querySelector('.t-title').textContent = title;
  el.querySelector('.t-body').textContent = body;
  host.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(8px)';
    setTimeout(() => el.remove(), 220);
  }, type === 'err' ? 6500 : 3500);
}

async function apiCall(method, path, { query, signal } = {}) {
  let url = API + path;
  if (query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === '') continue;
      qs.append(k, v);
    }
    if ([...qs].length) url += '?' + qs.toString();
  }
  const res = await fetch(url, { method, signal });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; }
  catch { data = text; }
  if (!res.ok) {
    const detail = (data && data.detail) || res.statusText || 'unknown error';
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.payload = data;
    throw err;
  }
  return data;
}

/* ---------- Tab switching ---------- */
const panels = {
  train:  $('#panel-train'),
  global: $('#panel-global'),
  user:   $('#panel-user'),
};
$$('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab').forEach(b => b.classList.toggle('active', b === btn));
    const key = btn.dataset.tab;
    for (const [k, el] of Object.entries(panels)) {
      el.hidden = k !== key;
    }
    onTabChange(key);
  });
});
function onTabChange(key) {
  if (key === 'train') connectLogs({ silent: true });
  else                 disconnectLogs({ silent: true });
}

/* ---------- API status check ---------- */
async function refreshStatus() {
  const chip = $('#apiStatus');
  const label = chip.querySelector('.label');
  try {
    const data = await apiCall('GET', '/status');
    chip.classList.remove('bad'); chip.classList.add('ok');
    const bits = [];
    if (data.model_loaded_in_memory) bits.push('model ✓');
    else if (data.model_file_exists) bits.push('model on disk');
    else bits.push('no model');
    if (data.candidates_count) bits.push(`${data.candidates_count} users w/ candidates`);
    label.textContent = bits.join(' · ');
  } catch (e) {
    chip.classList.remove('ok'); chip.classList.add('bad');
    label.textContent = 'API unreachable';
  }
}
refreshStatus();
setInterval(refreshStatus, 12000);

/* ============================================================
   1) /train_full
   ============================================================ */
const trainForm = $('#trainForm');
const trainBtn  = $('#trainBtn');
const trainResultEl = $('#trainResult');

trainForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(trainForm);
  const params = {};
  for (const [k, v] of fd.entries()) params[k] = v;
  // checkboxes — FormData omits unchecked ones; we need explicit booleans
  for (const name of ['use_existing_model', 'skip_if_exists']) {
    params[name] = fd.get(name) !== null ? 'true' : 'false';
  }

  setBusy(trainBtn, true);
  trainResultEl.hidden = true;
  // Open the live log stream so user can watch progress
  connectLogs();
  toast({ type: 'info', title: 'Training started',
           body: 'Логи стримятся снизу. Окно можно не трогать.' });

  const start = Date.now();
  try {
    const data = await apiCall('POST', '/train_full', { query: params });
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    renderTrainResult({ ok: true, data, elapsed });
    toast({ type: 'ok', title: 'Training finished', body: `за ${elapsed}s` });
    refreshStatus();
  } catch (err) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    renderTrainResult({ ok: false, error: err, elapsed });
    toast({ type: 'err', title: `Failed (${err.status || '—'})`, body: err.message });
  } finally {
    setBusy(trainBtn, false);
  }
});

function renderTrainResult({ ok, data, error, elapsed }) {
  trainResultEl.hidden = false;
  const badge = trainResultEl.querySelector('.badge');
  const title = trainResultEl.querySelector('.result-title');
  const body  = trainResultEl.querySelector('.result-body');
  if (ok) {
    badge.className = 'badge ok'; badge.textContent = 'success';
    title.textContent = `Pipeline complete · ${elapsed}s`;
    body.textContent = JSON.stringify(data, null, 2);
  } else {
    badge.className = 'badge err'; badge.textContent = 'error';
    title.textContent = `Failed after ${elapsed}s`;
    body.textContent = (error.payload ? JSON.stringify(error.payload, null, 2) : error.message);
  }
}

/* ============================================================
   2) GET /shap/global
   ============================================================ */
const globalForm = $('#globalForm');
const globalBtn  = $('#globalBtn');
const globalResultEl = $('#globalResult');
let globalChart = null;

globalForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(globalForm);
  const params = Object.fromEntries(fd.entries());

  setBusy(globalBtn, true);
  globalResultEl.hidden = true;
  try {
    const data = await apiCall('GET', '/shap/global', { query: params });
    renderGlobal(data);
    toast({ type: 'ok', title: 'Global SHAP ready',
             body: `${data.n_pairs_analyzed} pairs · ${data.n_features_total} features` });
  } catch (err) {
    toast({ type: 'err', title: `SHAP failed (${err.status || '—'})`, body: err.message });
  } finally {
    setBusy(globalBtn, false);
  }
});

function renderGlobal(data) {
  globalResultEl.hidden = false;

  // metrics row
  const metricsEl = $('#globalMetrics');
  metricsEl.innerHTML = '';
  const items = [
    { label: 'Pairs analyzed', value: data.n_pairs_analyzed ?? '—' },
    { label: 'Features total', value: data.n_features_total ?? '—' },
    { label: 'Top feature',
      value: data.feature_importance?.[0]?.feature ?? '—' },
    { label: 'Max |SHAP|',
      value: (data.feature_importance?.[0]?.mean_abs_shap ?? 0).toFixed(4) },
  ];
  for (const m of items) {
    const el = document.createElement('div');
    el.className = 'metric';
    el.innerHTML = `<div class="metric-label">${m.label}</div>
                    <div class="metric-value">${m.value}</div>`;
    metricsEl.appendChild(el);
  }

  // chart
  const fi = (data.feature_importance || []).slice(0, 15);
  const labels = fi.map(d => d.feature);
  const vals   = fi.map(d => d.mean_abs_shap);
  const signed = fi.map(d => d.mean_shap);

  const ctx = $('#globalChart').getContext('2d');
  if (globalChart) globalChart.destroy();
  globalChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'mean |SHAP|',
        data: vals,
        backgroundColor: signed.map(v => v >= 0
          ? 'rgba(52,211,153,.78)'
          : 'rgba(248,113,113,.78)'),
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      maintainAspectRatio: false,
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17,20,27,.96)',
          borderColor: 'rgba(60,68,86,1)', borderWidth: 1,
          padding: 12,
          callbacks: {
            afterLabel: (ctx) => {
              const s = signed[ctx.dataIndex];
              return `mean SHAP: ${s.toFixed(4)} (${s >= 0 ? 'pushes up' : 'pushes down'})`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,.05)' },
          ticks: { color: '#a3a8b8', font: { family: 'JetBrains Mono' } },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#e6e8ee', font: { size: 12 } },
        }
      }
    }
  });

  $('#globalRaw').textContent = JSON.stringify(data, null, 2);
}

/* ============================================================
   3) POST /shap/explain/recommendations/{user_id}
   ============================================================ */
const userForm = $('#userForm');
const userBtn  = $('#userBtn');
const userResultEl = $('#userResult');

userForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(userForm);
  const userId = fd.get('user_id');
  if (!userId) return;
  const params = { n_recs: fd.get('n_recs'), nsamples: fd.get('nsamples') };

  setBusy(userBtn, true);
  userResultEl.hidden = true;

  try {
    const data = await apiCall('POST',
      `/shap/explain/recommendations/${encodeURIComponent(userId)}`,
      { query: params });
    renderUser(data);
    toast({ type: 'ok', title: 'Explanations ready',
             body: `${data.recommendations.length} recommendations` });
  } catch (err) {
    toast({ type: 'err', title: `Explanation failed (${err.status || '—'})`, body: err.message });
  } finally {
    setBusy(userBtn, false);
  }
});

function renderUser(data) {
  userResultEl.hidden = false;
  const recs = data.recommendations || [];

  // summary
  const summary = $('#userSummary');
  summary.innerHTML = `
    <div class="who">User <b>${data.user_id}</b> · ${recs.length} recommendation${recs.length === 1 ? '' : 's'}</div>
    <button type="button" id="llmExplainAll" class="btn-llm-all">✨ Объяснить все через LLM</button>
  `;
  $('#llmExplainAll').addEventListener('click', () => explainAllWithLLM(data.user_id));

  // recommendations list
  const list = $('#userRecs');
  list.innerHTML = '';

  if (!recs.length) {
    list.innerHTML = `<div class="rec-card"><div class="rec-title">Нет рекомендаций для этого пользователя.</div></div>`;
    return;
  }

  // Build factor scale (use max |val| across all recs to compare visually)
  let scale = 0;
  for (const r of recs) {
    for (const [, v] of [...(r.top_positive || []), ...(r.top_negative || [])]) {
      if (Math.abs(v) > scale) scale = Math.abs(v);
    }
  }
  if (!scale) scale = 1;

  // user_id is captured from the form when /shap/explain was called;
  // store it on the result element for the LLM button to pick up.
  const ctxUserId = $('#userForm input[name="user_id"]').value;

  for (const rec of recs) {
    const card = document.createElement('div');
    card.className = 'rec-card';
    const title = rec.book_title || `Book ${rec.book_id}`;
    const score = (rec.score != null) ? rec.score.toFixed(4) : '—';
    card.innerHTML = `
      <div class="rec-head">
        <div>
          <div class="rec-title"></div>
          <div class="rec-meta">ISBN: ${rec.book_id}</div>
        </div>
        <span class="rec-score">score · ${score}</span>
      </div>
      <div class="genre-tags"></div>
      <div class="rec-factors"></div>
      <div class="llm-block" hidden>
        <div class="llm-text"></div>
      </div>`;
    card.querySelector('.rec-title').textContent = title;
    card.dataset.bookId = rec.book_id;
    card.dataset.userId = ctxUserId;

    // Genre tags (max 5, "+N more" if longer)
    const tagsEl = card.querySelector('.genre-tags');
    const allGenres = Array.isArray(rec.genres) ? rec.genres : [];
    const shown = allGenres.slice(0, 5);
    for (const g of shown) {
      const tag = document.createElement('span');
      tag.className = 'genre-tag';
      tag.textContent = g;
      tagsEl.appendChild(tag);
    }
    if (allGenres.length > shown.length) {
      const more = document.createElement('span');
      more.className = 'genre-tag muted';
      more.textContent = `+${allGenres.length - shown.length}`;
      tagsEl.appendChild(more);
    }
    if (!allGenres.length) tagsEl.remove();

    const factors = card.querySelector('.rec-factors');
    const pos = (rec.top_positive || []).slice(0, 5);
    const neg = (rec.top_negative || []).slice(0, 5);
    // interleave: positives first (sorted desc), then negatives (sorted by magnitude desc)
    const ordered = [...pos, ...neg];
    for (const [name, value] of ordered) {
      const row = document.createElement('div');
      row.className = 'factor-row';
      const isPos = value >= 0;
      const pct = Math.min(100, Math.abs(value) / scale * 100);
      row.innerHTML = `
        <div class="factor-name" title="${name}"></div>
        <div class="factor-bar"><div class="fill ${isPos ? 'pos' : 'neg'}" style="width:${pct/2}%"></div></div>
        <div class="factor-val ${isPos ? 'pos' : 'neg'}">${value.toFixed(4)}</div>`;
      row.querySelector('.factor-name').textContent = name;
      factors.appendChild(row);
    }
    list.appendChild(card);
  }
}

/* ============================================================
   LLM-powered explanations (calls /api/v1/explain/llm/{user_id})
   ============================================================ */
async function explainAllWithLLM(userId) {
  const btn = $('#llmExplainAll');
  if (!btn) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = '✨ LLM думает…';

  // Show pending state on each card
  for (const card of $$('.rec-card')) {
    const block = card.querySelector('.llm-block');
    const text  = card.querySelector('.llm-text');
    if (block && text) {
      block.hidden = false;
      text.classList.add('pending');
      text.textContent = '… генерирую объяснение …';
    }
  }

  const fd = new FormData(userForm);
  const params = { n_recs: fd.get('n_recs'), nsamples: fd.get('nsamples') };

  try {
    const data = await apiCall('POST',
      `/explain/llm/${encodeURIComponent(userId)}`,
      { query: params });

    const recs = data.recommendations || [];
    let any = 0;
    for (const rec of recs) {
      const card = $$('.rec-card').find(c => c.dataset.bookId === String(rec.book_id));
      if (!card) continue;
      const block = card.querySelector('.llm-block');
      const text  = card.querySelector('.llm-text');
      text.classList.remove('pending');
      if (rec.llm_explanation) {
        text.textContent = rec.llm_explanation;
        block.hidden = false;
        any++;
      } else {
        block.hidden = true;
      }
    }
    const tok = data.tokens || {};
    toast({
      type: 'ok', title: 'LLM ready',
      body: `${any} объяснений · model ${data.model || ''}`
              + (tok.total_tokens ? ` · ${tok.total_tokens} tok` : ''),
    });
  } catch (err) {
    // Restore card state on failure
    for (const card of $$('.rec-card')) {
      const block = card.querySelector('.llm-block');
      const text  = card.querySelector('.llm-text');
      if (block && text) {
        block.hidden = true;
        text.classList.remove('pending');
      }
    }
    toast({
      type: 'err',
      title: `LLM failed (${err.status || '—'})`,
      body: err.message,
    });
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

/* ============================================================
   Live server logs (Server-Sent Events from /api/v1/logs/stream)
   ============================================================ */
const logPanel   = $('#logPanel');
const logBody    = $('#logBody');
const logStateEl = logPanel.querySelector('.log-state');
const logConnBtn = $('#logConnBtn');
const logScrollBtn = $('#logScrollBtn');
const logClearBtn  = $('#logClearBtn');

let logSrc = null;
let autoScroll = true;
let logLines = 0;
const MAX_LINES = 2500;

function setLogState(state) {
  logStateEl.textContent = state;
  logPanel.classList.remove('ok', 'bad', 'pending');
  if (state === 'live')         logPanel.classList.add('ok');
  else if (state === 'connecting') logPanel.classList.add('pending');
  else if (state === 'error')   logPanel.classList.add('bad');
  // button label
  if (state === 'live' || state === 'connecting') {
    logConnBtn.textContent = 'Отключить';
    logConnBtn.classList.add('primary');
  } else {
    logConnBtn.textContent = 'Подключиться';
    logConnBtn.classList.remove('primary');
  }
}

function classifyLogLine(text) {
  if (/❌|⚠|🚨|error|ошибка|fail|exception|traceback/i.test(text)) return 'err';
  if (/✅|✓|готов|complete|saved|сохранен|loaded|загружен/i.test(text)) return 'ok';
  if (/^={3,}|🚀|📊|📚|🔧|🎭|🤖|🧹|🔄|💾|🔍|📈|👤|⭐|📐|📋|🎯/.test(text)) return 'info';
  if (/^\s*===\s*session|^\s*$/.test(text)) return 'muted';
  return null;
}

function appendLog(text) {
  const el = document.createElement('div');
  el.className = 'ln';
  const k = classifyLogLine(text);
  if (k) el.classList.add(k);
  el.textContent = text;
  logBody.appendChild(el);
  logLines++;
  if (logLines > MAX_LINES) {
    while (logBody.children.length > MAX_LINES) {
      logBody.removeChild(logBody.firstChild);
      logLines--;
    }
  }
  if (autoScroll) logBody.scrollTop = logBody.scrollHeight;
}

function connectLogs({ silent = false } = {}) {
  if (logSrc && logSrc.readyState !== 2 /* CLOSED */) return;
  setLogState('connecting');
  try {
    logSrc = new EventSource('/api/v1/logs/stream?tail=300');
  } catch (e) {
    setLogState('error');
    if (!silent) toast({ type: 'err', title: 'Logs failed', body: e.message });
    return;
  }
  logSrc.onopen = () => setLogState('live');
  logSrc.onmessage = (ev) => appendLog(ev.data);
  logSrc.onerror = () => {
    // EventSource will auto-reconnect; show 'error' until it does
    if (logSrc && logSrc.readyState === 0) setLogState('connecting');
    else setLogState('error');
  };
}

function disconnectLogs({ silent = false } = {}) {
  if (logSrc) {
    logSrc.close();
    logSrc = null;
  }
  setLogState('disconnected');
}

logConnBtn.addEventListener('click', () => {
  if (logSrc && logSrc.readyState !== 2) disconnectLogs();
  else connectLogs();
});
logScrollBtn.addEventListener('click', () => {
  autoScroll = !autoScroll;
  logScrollBtn.textContent = `Auto-scroll · ${autoScroll ? 'on' : 'off'}`;
  if (autoScroll) logBody.scrollTop = logBody.scrollHeight;
});
logClearBtn.addEventListener('click', () => {
  logBody.innerHTML = '';
  logLines = 0;
});

// Pause auto-scroll if the user manually scrolls up
logBody.addEventListener('scroll', () => {
  const atBottom =
    logBody.scrollTop + logBody.clientHeight >= logBody.scrollHeight - 20;
  if (!atBottom && autoScroll) {
    autoScroll = false;
    logScrollBtn.textContent = 'Auto-scroll · off';
  } else if (atBottom && !autoScroll) {
    // Don't auto-flip back on; user explicitly turned it off.
  }
});

// Auto-connect on first page load (we start on the train tab)
setLogState('disconnected');
connectLogs({ silent: true });

