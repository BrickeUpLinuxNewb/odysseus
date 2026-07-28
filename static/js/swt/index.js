// static/js/swt/index.js
// SWT — the automated generator → critic → analyzer loop.
//
// A self-contained tool window (built the same way as the Deep Research panel:
// a `.modal` overlay + `.modal-content` pane, draggable by its header) that
// streams a loop run over SSE and renders each round live: the generated
// answer, the critic's verdict, the analyzer's diagnosis category, and the
// cognitive model's predicted user-satisfaction. The final answer carries
// Accept / Reject buttons that train the cognitive model.
//
// The panel reuses Odysseus's model list (/api/models) and existing modal CSS,
// so it looks native without touching style.css.

let _open = false;
let _onDocKeydown = null;
let _running = false;
let _lastRun = null; // { prompt, answer } for feedback

const CATEGORY_LABELS = {
  retrieval_failure: 'Retrieval failure',
  knowledge_gap: 'Knowledge gap',
  orchestration_failure: 'Orchestration failure',
  none: 'Accepted',
};

function _esc(s) {
  // Escapes for both text and double/single-quoted attribute contexts, so a
  // crafted model name (e.g. from a rogue/poisoned model server on the LAN)
  // cannot break out of value="…" / class="…" and inject attributes.
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function _injectStyle() {
  if (document.getElementById('swt-style')) return;
  const style = document.createElement('style');
  style.id = 'swt-style';
  style.textContent = `
    #swt-overlay .swt-pane-body { padding: 14px 16px; overflow-y: auto; }
    #swt-overlay .swt-desc { opacity: .65; font-size: 12px; margin: 2px 0 12px; }
    #swt-overlay .swt-field { display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }
    #swt-overlay .swt-field > label { font-size: 11px; opacity: .7; text-transform: uppercase; letter-spacing: .04em; }
    #swt-overlay textarea, #swt-overlay input, #swt-overlay select {
      background: var(--input-bg, var(--bg)); color: var(--fg, inherit);
      border: 1px solid var(--border, #3a3a3a); border-radius: 8px; padding: 8px 10px; font: inherit;
    }
    #swt-overlay .swt-models { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    #swt-overlay .swt-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    #swt-overlay .swt-row label { font-size: 12px; opacity: .8; display: inline-flex; gap: 5px; align-items: center; }
    #swt-overlay .swt-run-btn {
      background: var(--accent, var(--red, #d33)); color: #fff; border: none; border-radius: 8px;
      padding: 9px 18px; font-weight: 600; cursor: pointer;
    }
    #swt-overlay .swt-run-btn[disabled] { opacity: .5; cursor: default; }
    #swt-overlay .swt-rounds { margin-top: 14px; display: flex; flex-direction: column; gap: 10px; }
    #swt-overlay .swt-round {
      border: 1px solid var(--border, #3a3a3a); border-radius: 10px; padding: 10px 12px;
      background: var(--panel, rgba(127,127,127,.06));
    }
    #swt-overlay .swt-round h5 { margin: 0 0 6px; font-size: 12px; opacity: .8; }
    #swt-overlay .swt-answer { white-space: pre-wrap; font-size: 13px; line-height: 1.45; }
    #swt-overlay .swt-sub { font-size: 12px; margin-top: 8px; opacity: .9; }
    #swt-overlay .swt-sub b { opacity: .7; font-weight: 600; }
    #swt-overlay .swt-badge {
      display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
    }
    #swt-overlay .swt-badge.retrieval_failure { background: #8a5a00; color: #fff; }
    #swt-overlay .swt-badge.knowledge_gap { background: #7a2f8a; color: #fff; }
    #swt-overlay .swt-badge.orchestration_failure { background: #1f6f8b; color: #fff; }
    #swt-overlay .swt-badge.none { background: #2e7d32; color: #fff; }
    #swt-overlay .swt-badge.accepted { background: #2e7d32; color: #fff; }
    #swt-overlay .swt-badge.rejected { background: #b23b3b; color: #fff; }
    #swt-overlay .swt-final { margin-top: 14px; border-top: 1px solid var(--border, #3a3a3a); padding-top: 12px; }
    #swt-overlay .swt-final .swt-answer { border-left: 3px solid var(--accent, var(--red, #d33)); padding-left: 10px; }
    #swt-overlay .swt-feedback { display: flex; gap: 8px; margin-top: 10px; align-items: center; }
    #swt-overlay .swt-fb-btn { border: 1px solid var(--border, #3a3a3a); background: transparent; color: inherit;
      border-radius: 8px; padding: 6px 14px; cursor: pointer; }
    #swt-overlay .swt-fb-btn.accept:hover { background: #2e7d32; color: #fff; }
    #swt-overlay .swt-fb-btn.reject:hover { background: #b23b3b; color: #fff; }
    #swt-overlay .swt-status { font-size: 12px; opacity: .7; margin-top: 8px; min-height: 16px; }
    #swt-overlay .swt-issues { margin: 4px 0 0 16px; font-size: 12px; }
  `;
  document.head.appendChild(style);
}

async function _loadModels(pane) {
  const dl = pane.querySelector('#swt-model-list');
  if (!dl) return;
  try {
    const r = await fetch('/api/models?background=false', { credentials: 'same-origin' });
    if (!r.ok) return;
    const data = await r.json();
    // Defensive: accept several shapes ({models:[...]}, [...], {data:[...]}).
    let items = [];
    if (Array.isArray(data)) items = data;
    else if (Array.isArray(data.models)) items = data.models;
    else if (Array.isArray(data.data)) items = data.data;
    const names = new Set();
    for (const it of items) {
      if (typeof it === 'string') names.add(it);
      else if (it && (it.id || it.name || it.model)) names.add(it.id || it.name || it.model);
    }
    dl.innerHTML = [...names].map((n) => `<option value="${_esc(n)}"></option>`).join('');
  } catch { /* free-text fallback is fine */ }
}

function _buildHTML() {
  return `
    <div class="modal-header swt-pane-header" style="cursor:move;">
      <h4 style="margin:0;display:inline-flex;align-items:center;gap:6px;">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/></svg>
        SWT Loop
      </h4>
      <button id="swt-close" class="close-btn" title="Close">&#x2716;</button>
    </div>
    <div class="modal-body swt-pane-body" data-no-swipe-dismiss>
      <p class="swt-desc">Generator answers → Critic finds problems → Analyzer diagnoses the root cause and adjusts the next round. Converges when the critic accepts and the cognitive model predicts you will too.</p>

      <div class="swt-field">
        <label>Request</label>
        <textarea id="swt-prompt" rows="3" placeholder="Ask something you want to get right…"></textarea>
      </div>

      <div class="swt-field">
        <label>Reference material (optional — condensed recursively if long)</label>
        <textarea id="swt-context" rows="2" placeholder="Paste notes, docs, or context the answer should draw on…"></textarea>
      </div>

      <div class="swt-field">
        <label>Models</label>
        <div class="swt-models">
          <input id="swt-gen" list="swt-model-list" placeholder="Generator" />
          <input id="swt-crit" list="swt-model-list" placeholder="Critic" />
          <input id="swt-anal" list="swt-model-list" placeholder="Analyzer (optional)" />
        </div>
        <datalist id="swt-model-list"></datalist>
      </div>

      <div class="swt-row" style="margin-bottom:12px;">
        <label>Max rounds
          <select id="swt-rounds">
            <option>2</option><option selected>4</option><option>6</option><option>8</option>
          </select>
        </label>
        <label>Satisfaction ≥
          <select id="swt-thresh">
            <option value="0.5">0.5</option><option value="0.7" selected>0.7</option><option value="0.9">0.9</option>
          </select>
        </label>
        <label><input type="checkbox" id="swt-cog" checked> Cognitive model</label>
        <label><input type="checkbox" id="swt-rec" checked> Recursive context</label>
      </div>

      <button id="swt-run" class="swt-run-btn">Run loop</button>
      <div class="swt-status" id="swt-status"></div>

      <div class="swt-rounds" id="swt-rounds-out"></div>
      <div id="swt-final-out"></div>
    </div>
  `;
}

function _roundEl(idx, gen) {
  const el = document.createElement('div');
  el.className = 'swt-round';
  el.id = `swt-round-${idx}`;
  el.innerHTML = `<h5>Round ${idx + 1} · <span style="opacity:.7">${_esc(gen)}</span></h5><div class="swt-round-content"></div>`;
  return el;
}

function _renderCritique(c) {
  const badge = c.accepted
    ? '<span class="swt-badge accepted">Accepted</span>'
    : '<span class="swt-badge rejected">Needs work</span>';
  let issues = '';
  if (c.issues && c.issues.length) {
    issues = '<ul class="swt-issues">' + c.issues.map((i) => `<li>${_esc(i)}</li>`).join('') + '</ul>';
  }
  return `<div class="swt-sub"><b>Critic:</b> ${badge} ${_esc(c.summary)}${issues}</div>`;
}

function _renderDiagnosis(d) {
  const cat = d.category || 'none';
  const label = CATEGORY_LABELS[cat] || cat;
  return `<div class="swt-sub"><b>Analyzer:</b> <span class="swt-badge ${_esc(cat)}">${_esc(label)}</span> ${_esc(d.rationale)}</div>`;
}

function _renderSatisfaction(s) {
  if (!s) return '';
  const pct = Math.round((s.score || 0) * 100);
  const reasons = (s.reasons || []).map(_esc).join(' ');
  return `<div class="swt-sub"><b>Predicted satisfaction:</b> ${pct}% <span style="opacity:.6">(${_esc(s.basis)})</span> ${reasons}</div>`;
}

async function _run(pane) {
  if (_running) return;
  const prompt = pane.querySelector('#swt-prompt').value.trim();
  const gen = pane.querySelector('#swt-gen').value.trim();
  const crit = pane.querySelector('#swt-crit').value.trim();
  const anal = pane.querySelector('#swt-anal').value.trim();
  const status = pane.querySelector('#swt-status');
  const roundsOut = pane.querySelector('#swt-rounds-out');
  const finalOut = pane.querySelector('#swt-final-out');

  if (!prompt || !gen || !crit) {
    status.textContent = 'Need a request, a generator model, and a critic model.';
    return;
  }
  _running = true;
  roundsOut.innerHTML = '';
  finalOut.innerHTML = '';
  status.textContent = 'Starting…';
  const runBtn = pane.querySelector('#swt-run');
  runBtn.disabled = true;

  const body = {
    prompt,
    generator_model: gen,
    critic_model: crit,
    analyzer_model: anal,
    max_rounds: parseInt(pane.querySelector('#swt-rounds').value, 10),
    satisfaction_threshold: parseFloat(pane.querySelector('#swt-thresh').value),
    context: pane.querySelector('#swt-context').value,
    use_cognitive_model: pane.querySelector('#swt-cog').checked,
    use_recursive_context: pane.querySelector('#swt-rec').checked,
  };

  try {
    const resp = await fetch('/api/swt/run', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      status.textContent = `Failed to start (HTTP ${resp.status}).`;
      _running = false; runBtn.disabled = false; return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
        _handleEvent(ev, { status, roundsOut, finalOut, prompt });
      }
    }
  } catch (e) {
    status.textContent = 'Stream error: ' + (e && e.message ? e.message : e);
  } finally {
    _running = false;
    runBtn.disabled = false;
  }
}

function _handleEvent(ev, ctx) {
  const { status, roundsOut, finalOut, prompt } = ctx;
  switch (ev.type) {
    case 'loop_start':
      status.textContent = 'Running loop…';
      break;
    case 'context_condensing':
      status.textContent = `Condensing ${ev.original_chars} chars of context…`;
      break;
    case 'context_condensed':
      status.textContent = `Context condensed to ${ev.condensed_chars} chars.`;
      break;
    case 'round_start': {
      const el = _roundEl(ev.round, ev.generator_model);
      roundsOut.appendChild(el);
      status.textContent = `Round ${ev.round + 1}: generating…`;
      break;
    }
    case 'generation': {
      const c = document.querySelector(`#swt-round-${ev.round} .swt-round-content`);
      if (c) c.innerHTML = `<div class="swt-answer">${_esc(ev.answer)}</div>`;
      break;
    }
    case 'critique': {
      const c = document.querySelector(`#swt-round-${ev.round} .swt-round-content`);
      if (c) c.insertAdjacentHTML('beforeend', _renderCritique(ev.critique));
      break;
    }
    case 'diagnosis': {
      const c = document.querySelector(`#swt-round-${ev.round} .swt-round-content`);
      if (c) c.insertAdjacentHTML('beforeend', _renderDiagnosis(ev.diagnosis));
      break;
    }
    case 'satisfaction': {
      const c = document.querySelector(`#swt-round-${ev.round} .swt-round-content`);
      if (c) c.insertAdjacentHTML('beforeend', _renderSatisfaction(ev.satisfaction));
      break;
    }
    case 'model_switch':
      status.textContent = `Analyzer switched generator: ${ev.from} → ${ev.to}`;
      break;
    case 'error':
      status.textContent = `Error (${ev.stage}): ${ev.message}`;
      break;
    case 'loop_complete':
      _renderFinal(ev.result, finalOut, status, prompt);
      break;
    default:
      break;
  }
}

function _renderFinal(result, finalOut, status, prompt) {
  if (!result) { status.textContent = 'Loop ended without a result.'; return; }
  status.textContent = result.converged
    ? `Converged: ${result.converged_reason}`
    : result.converged_reason || 'Finished.';
  _lastRun = { loop_id: result.loop_id, prompt, answer: result.final_answer };
  finalOut.innerHTML = `
    <div class="swt-final">
      <div class="swt-sub"><b>Final answer</b> (${result.rounds.length} round${result.rounds.length === 1 ? '' : 's'})</div>
      <div class="swt-answer">${_esc(result.final_answer)}</div>
      <div class="swt-feedback">
        <span style="font-size:12px;opacity:.7;">Train the cognitive model:</span>
        <button class="swt-fb-btn accept" id="swt-accept">Accept</button>
        <button class="swt-fb-btn reject" id="swt-reject">Reject</button>
      </div>
    </div>`;
  finalOut.querySelector('#swt-accept').addEventListener('click', () => _feedback(true, finalOut));
  finalOut.querySelector('#swt-reject').addEventListener('click', () => _feedback(false, finalOut));
}

async function _feedback(accepted, finalOut) {
  if (!_lastRun) return;
  const fb = finalOut.querySelector('.swt-feedback');
  try {
    const r = await fetch('/api/swt/feedback', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: _lastRun.prompt,
        answer: _lastRun.answer,
        accepted,
        loop_id: _lastRun.loop_id,
      }),
    });
    const data = await r.json().catch(() => ({}));
    const stats = data.stats || {};
    if (fb) {
      fb.innerHTML = `<span style="font-size:12px;opacity:.75;">Recorded ${accepted ? 'accept' : 'reject'}. ` +
        `Cognitive model now has ${stats.total || 0} example(s) (${stats.accepted || 0} accepted).</span>`;
    }
  } catch {
    if (fb) fb.innerHTML = '<span style="font-size:12px;opacity:.75;">Could not save feedback.</span>';
  }
}

export function openPanel() {
  if (_open) return;
  _open = true;
  _injectStyle();

  const btn = document.getElementById('tool-swt-btn');
  if (btn) btn.classList.add('active');

  const overlay = document.createElement('div');
  overlay.id = 'swt-overlay';
  overlay.className = 'modal swt-overlay';

  const pane = document.createElement('div');
  pane.id = 'swt-pane';
  pane.className = 'modal-content swt-pane';
  pane.style.cssText = (window.innerWidth <= 768)
    ? 'width:100vw;max-width:100vw;height:90dvh;max-height:90dvh;border-radius:14px 14px 0 0;background:var(--bg);'
    : 'width:min(680px, 94vw);max-height:88vh;background:var(--bg);';
  pane.innerHTML = _buildHTML();

  overlay.appendChild(pane);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', (e) => { if (e.target === overlay) closePanel(); });
  _onDocKeydown = (e) => { if (e.key === 'Escape' && _open) { e.preventDefault(); closePanel(); } };
  document.addEventListener('keydown', _onDocKeydown);

  pane.querySelector('#swt-close').addEventListener('click', closePanel);
  pane.querySelector('#swt-run').addEventListener('click', () => _run(pane));

  // Draggable header, matching the rest of the modal family when available.
  import('/static/js/theme.js').then((m) => {
    const header = pane.querySelector('.swt-pane-header');
    if (m && m.makeDraggable && header) m.makeDraggable(pane, header);
  }).catch(() => {});

  _loadModels(pane);
}

export function closePanel() {
  if (!_open) return;
  _open = false;
  if (_onDocKeydown) { document.removeEventListener('keydown', _onDocKeydown); _onDocKeydown = null; }
  const btn = document.getElementById('tool-swt-btn');
  if (btn) btn.classList.remove('active');
  const overlay = document.getElementById('swt-overlay');
  if (overlay) overlay.remove();
}

function _wireLaunchers() {
  const ids = ['tool-swt-btn', 'rail-swt'];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el && !el._swtWired) {
      el._swtWired = true;
      el.addEventListener('click', (e) => { e.preventDefault(); openPanel(); });
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _wireLaunchers);
} else {
  _wireLaunchers();
}

// Expose for other modules / keyboard shortcuts, mirroring existing tools.
window.SWT = { open: openPanel, close: closePanel };
