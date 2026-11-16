import { db, settings } from './db.js';
import { fetchArchiveList, fetchArchive, gameFromApi, gameFromPgn, timeClassLabel, timeControlLabel } from './games.js';
import { looksLikePgn } from './pgn.js';
import { Engine, DEPTHS } from './engine.js';
import { analyseGame, deriveMoves, summarize, wpFor, evalText, CLASSES, ANALYSIS_VERSION } from './analysis.js';
import { commentFor, introFor, setNotation, fmtSan, fmtClock, fmtPct, esc, moveLabel } from './coach.js';
import { Board } from './board.js';
import { AVATAR_SVG, setMood, talk } from './avatar.js';

const $ = s => document.querySelector(s);
const bridge = window.AndroidBridge || null;
if (!bridge) document.body.classList.add('no-android');

let cfg = settings.get();
setNotation(cfg.notation);

const engine = new Engine();
const cache = new Map();          // id -> game (the same object the analyser updates)
const view = { game: null, derived: [], summary: null, idx: 0, flip: false };
const board = new Board($('#board'));

$('#avatar').innerHTML = AVATAR_SVG;
document.querySelector('.brand-owl').innerHTML = AVATAR_SVG;
document.querySelector('.ob-owl').innerHTML = AVATAR_SVG;
setMood(document.querySelector('.ob-owl'), 'happy');

// ---------------------------------------------------------------- helpers

let toastTimer;
function toast(msg, ms = 2800) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
}

async function getGame(id) {
  if (cache.has(id)) return cache.get(id);
  const g = await db.get(id);
  if (g) cache.set(id, g);
  return g;
}

async function allGames() {
  const list = await db.all();
  return list.map(g => {
    if (!cache.has(g.id)) cache.set(g.id, g);
    return cache.get(g.id);
  });
}

function isAnalysed(g) {
  return !!(g.analysis && g.analysis.done && g.analysis.version === ANALYSIS_VERSION);
}

function fmtDate(ms) {
  if (!ms) return '';
  const d = new Date(ms), now = new Date();
  const time = d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
  const days = Math.floor((new Date(now.toDateString()) - new Date(d.toDateString())) / 86400000);
  if (days === 0) return 'Heute, ' + time;
  if (days === 1) return 'Gestern, ' + time;
  return d.toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', year: d.getFullYear() === now.getFullYear() ? undefined : '2-digit' }) + ', ' + time;
}

function isCharging() {
  if (bridge && bridge.isCharging) { try { return Promise.resolve(!!bridge.isCharging()); } catch { /* ignore */ } }
  if (navigator.getBattery) return navigator.getBattery().then(b => b.charging).catch(() => false);
  return Promise.resolve(false);
}

// ---------------------------------------------------------------- analysis queue

const analyser = { queue: [], running: false, priority: null, progress: new Map() };

function enqueue(id, priority = false) {
  if (priority) analyser.priority = id;
  else if (!analyser.queue.includes(id)) analyser.queue.push(id);
  runQueue();
}

async function runQueue() {
  if (analyser.running) return;
  analyser.running = true;
  try {
    for (;;) {
      let id = null;
      if (analyser.priority) {
        const pg = await getGame(analyser.priority);
        if (pg && !isAnalysed(pg)) id = analyser.priority; else analyser.priority = null;
      }
      if (!id) id = analyser.queue.shift();
      if (!id) break;
      const game = await getGame(id);
      if (!game || isAnalysed(game)) continue;
      const opts = DEPTHS[cfg.depth] || DEPTHS.normal;
      let lastSave = Date.now();
      const finished = await analyseGame(game, engine, opts, async (i, n) => {
        analyser.progress.set(id, (i + 1) / n);
        onAnalysisProgress(game);
        if (Date.now() - lastSave > 4000) { lastSave = Date.now(); await db.put(game); }
      }, () => analyser.priority && analyser.priority !== id);
      if (finished) {
        game.summary = summarize(game, deriveMoves(game));
        analyser.progress.delete(id);
      } else if (!analyser.queue.includes(id)) analyser.queue.unshift(id);
      await db.put(game);
      onAnalysisProgress(game, finished);
    }
  } catch (e) {
    console.error(e);
    toast('Analyse-Fehler: ' + (e.message || e));
  } finally {
    analyser.running = false;
  }
}

let progressRaf = 0;
function onAnalysisProgress(game, finished) {
  if (view.game && view.game.id === game.id) {
    if (finished) refreshGameData();
    else {
      cancelAnimationFrame(progressRaf);
      progressRaf = requestAnimationFrame(() => { refreshGameData(); });
    }
  }
  const card = document.querySelector(`.gcard[data-id="${CSS.escape(game.id)}"]`);
  if (card) card.querySelector('.acc').innerHTML = accHtml(game);
  if (finished && $('#home').classList.contains('active')) renderStats();
}

// ---------------------------------------------------------------- chess.com sync

let syncing = false;
async function sync({ older = false, quiet = false } = {}) {
  const user = cfg.username.trim();
  if (!user || syncing) return;
  syncing = true;
  $('#btn-sync').classList.add('spin');
  const status = $('#sync-status');
  if (!quiet) { status.textContent = 'Lade Partien von Chess.com …'; status.classList.remove('hidden'); }
  try {
    const archives = await fetchArchiveList(user);
    if (!archives.length) { toast('Für diesen Namen gibt es noch keine Partien.'); return; }
    let months;
    if (older) {
      const idx = archives.length - 1 - cfg.monthsLoaded;
      if (idx < 0) { toast('Keine älteren Partien mehr.'); return; }
      months = [archives[idx]];
    } else {
      const lastSyncMonth = cfg.lastSync ? new Date(cfg.lastSync).getUTCMonth() : -1;
      const needPrev = !cfg.lastSync || lastSyncMonth !== new Date().getUTCMonth();
      months = archives.slice(needPrev ? -2 : -1);
    }
    const known = new Set(await db.keys());
    const fresh = [];
    for (const url of months) {
      const games = await fetchArchive(url);
      for (const g of games) {
        if (g.rules && g.rules !== 'chess') continue;
        if (!g.pgn) continue;
        let rec;
        try { rec = gameFromApi(g, user); } catch (e) { console.warn('skip game', e); continue; }
        if (known.has(rec.id) || !rec.moves.length) continue;
        known.add(rec.id);
        fresh.push(rec);
      }
    }
    if (fresh.length) await db.putMany(fresh);
    fresh.forEach(g => cache.set(g.id, g));
    cfg = settings.set({
      lastSync: older ? cfg.lastSync : Date.now(),
      monthsLoaded: older ? cfg.monthsLoaded + 1 : Math.max(cfg.monthsLoaded, months.length),
    });
    const newest = Math.max(0, ...(await allGames()).filter(g => g.source === 'chess.com').map(g => g.endTime || 0));
    if (bridge && bridge.setSyncState) bridge.setSyncState(user, newest, !!cfg.notify);
    status.textContent = fresh.length
      ? `${fresh.length} neue ${fresh.length === 1 ? 'Partie' : 'Partien'} geladen.`
      : 'Alles aktuell – keine neuen Partien.';
    status.classList.remove('hidden');
    setTimeout(() => status.classList.add('hidden'), 4000);
    await renderHome();
    if (fresh.length && !older) autoAnalyse(fresh);
  } catch (e) {
    console.warn(e);
    status.classList.add('hidden');
    if (e.status === 404) toast(`Chess.com kennt den Namen „${user}“ nicht.`, 4000);
    else toast('Chess.com ist gerade nicht erreichbar. Bist du online?', 4000);
  } finally {
    syncing = false;
    $('#btn-sync').classList.remove('spin');
  }
}

async function autoAnalyse(fresh) {
  if (cfg.autoAnalyse === 'open') return;
  if (cfg.autoAnalyse === 'charging' && !(await isCharging())) return;
  // newest first, at most 5 per sync (an old month full of games is analysed when you open them)
  fresh.sort((a, b) => (b.endTime || 0) - (a.endTime || 0)).slice(0, 5).forEach(g => enqueue(g.id));
}

// ---------------------------------------------------------------- home

function accHtml(g) {
  const p = analyser.progress.get(g.id);
  if (!isAnalysed(g) && p != null) return `<span>Analyse</span><div class="mini"><div style="width:${Math.round(p * 100)}%"></div></div>`;
  if (g.summary && g.summary.analysed && g.myColor) return `<b>${fmtPct(g.summary[g.myColor]).replace(' %', '')}</b><span>Genauigkeit</span>`;
  if (g.summary && g.summary.analysed) return `<b>${fmtPct(g.summary.w).replace(' %', '')}</b><span>Weiß</span>`;
  return '<span>noch nicht<br>analysiert</span>';
}

function gameCard(g) {
  const me = g.myColor;
  const opp = me === 'b' ? g.white : g.black;
  const res = g.outcome === 'win' ? 'S' : g.outcome === 'loss' ? 'N' : g.outcome === 'draw' ? '½' : g.result === '1-0' ? '1-0' : g.result === '0-1' ? '0-1' : '½';
  const oppTxt = me ? `<span class="dot ${me === 'w' ? 'b' : 'w'}"></span>${esc(opp.name)} <span class="sub">${opp.rating || ''}</span>`
    : `${esc(g.white.name)} – ${esc(g.black.name)}`;
  const meta = [timeClassLabel(g.timeClass), timeControlLabel(g), fmtDate(g.endTime)].filter(Boolean).join(' · ');
  const meta2 = [g.termination, Math.ceil(g.moves.length / 2) + ' Züge'].filter(Boolean).join(' · ');
  return `<div class="gcard ${g.outcome || ''}" data-id="${esc(g.id)}">
    <div class="res">${res}</div>
    <div class="mid"><div class="opp">${oppTxt}</div><div class="meta">${esc(meta)}</div><div class="meta">${esc(meta2)}</div></div>
    <div class="acc">${accHtml(g)}</div></div>`;
}

async function renderStats() {
  const games = (await allGames()).filter(g => g.myColor);
  const el = $('#stats');
  if (!games.length) { el.classList.add('hidden'); return; }
  const w = games.filter(g => g.outcome === 'win').length, l = games.filter(g => g.outcome === 'loss').length, d = games.filter(g => g.outcome === 'draw').length;
  const accs = games.filter(g => g.summary && g.summary.analysed).map(g => g.summary[g.myColor]).filter(x => x != null);
  const avg = accs.length ? accs.reduce((s, x) => s + x, 0) / accs.length : null;
  el.innerHTML = `<div class="stat"><div class="v">${games.length}</div><div class="l">Partien</div></div>
    <div class="stat"><div class="v">${w} / ${d} / ${l}</div><div class="l">S / R / N</div></div>
    <div class="stat"><div class="v">${fmtPct(avg)}</div><div class="l">Ø Genauigkeit</div></div>`;
  el.classList.remove('hidden');
}

async function renderHome() {
  const games = await allGames();
  $('#onboarding').classList.toggle('hidden', !!cfg.username);
  $('#btn-older').classList.toggle('hidden', !cfg.username || !cfg.lastSync);
  const list = $('#game-list');
  if (!games.length) {
    list.innerHTML = cfg.username ? '<div class="empty">Noch keine Partien geladen. Tippe oben auf ⟳.</div>' : '';
  } else {
    list.innerHTML = games.map(gameCard).join('');
  }
  renderStats();
}

$('#game-list').addEventListener('click', e => {
  const c = e.target.closest('.gcard');
  if (c) openGame(c.dataset.id);
});

// ---------------------------------------------------------------- game view

async function openGame(id, { idx = 0, push = true } = {}) {
  const game = await getGame(id);
  if (!game) { toast('Partie nicht gefunden.'); return; }
  view.game = game;
  view.flip = game.myColor === 'b';
  view.idx = idx;
  stopSpeaking();
  refreshGameData(true);
  $('#home').classList.remove('active');
  $('#game').classList.add('active');
  $('#game-scroll').scrollTop = 0;
  if (push) history.pushState({ game: id }, '');
  if (!isAnalysed(game)) enqueue(id, true);
  renderPosition();
}

function closeGame() {
  view.game = null;
  stopSpeaking();
  if (analyser.priority) analyser.priority = null;
  $('#game').classList.remove('active');
  $('#home').classList.add('active');
  renderHome();
}

function refreshGameData(first) {
  const g = view.game;
  if (!g) return;
  const prevCount = view.derived.filter(d => d.cls).length;
  view.derived = deriveMoves(g);
  view.summary = summarize(g, view.derived);
  const p = analyser.progress.get(g.id);
  const done = isAnalysed(g);
  $('#g-progress').classList.toggle('hidden', done || p == null);
  if (!done && p != null) {
    $('#g-progress-bar').style.width = Math.round(p * 100) + '%';
    $('#g-progress-text').textContent = `Stockfish analysiert … ${Math.round(p * 100)} %`;
  }
  const me = g.myColor;
  const opp = me === 'b' ? g.white : g.black;
  $('#g-title').textContent = me ? `gegen ${opp.name}` : `${g.white.name} – ${g.black.name}`;
  const res = g.outcome === 'win' ? 'Sieg' : g.outcome === 'loss' ? 'Niederlage' : g.outcome === 'draw' ? 'Remis' : g.result;
  $('#g-sub').textContent = [res, timeControlLabel(g), fmtDate(g.endTime)].filter(Boolean).join(' · ');
  const nowCount = view.derived.filter(d => d.cls).length;
  if (first || nowCount !== prevCount || done) {
    renderMoveStrip();
    renderGraph();
    renderSummary();
    // refresh the text only if the shown move just got its evaluation (do not interrupt reading otherwise)
    const shownDerived = view.idx > 0 ? view.derived[view.idx - 1] : null;
    if (first || view.idx === 0 || (shownDerived && shownDerived.cls && view.shownCls !== shownDerived.cls)) renderPosition({ speak: false });
  }
}

function renderMoveStrip() {
  const el = $('#moves');
  let html = `<button class="mv start" data-i="0">Start</button>`;
  view.derived.forEach((d, i) => {
    const c = d.cls ? CLASSES[d.cls] : null;
    const num = d.color === 'w' ? `<span class="n">${Math.ceil(d.ply / 2)}.</span>` : (i === 0 ? `<span class="n">${Math.ceil(d.ply / 2)}…</span>` : '');
    const sym = c && (d.score !== 0 || d.cls === 'forced') && d.cls !== 'best' && d.cls !== 'excellent' ? `<span class="s" style="color:${c.color}">${c.sym}</span>` : '';
    const border = c && d.mine && d.score <= -1 ? `style="border-bottom-color:${c.color}"` : (c && d.mine && d.score >= 2 ? `style="border-bottom-color:${c.color}"` : '');
    html += `<button class="mv" data-i="${i + 1}" ${border}>${num}${esc(fmtSan(d.san))}${sym}</button>`;
  });
  el.innerHTML = html;
  markCurrentMove();
}

function markCurrentMove() {
  const el = $('#moves');
  el.querySelectorAll('.mv.cur').forEach(b => b.classList.remove('cur'));
  const cur = el.querySelector(`.mv[data-i="${view.idx}"]`);
  if (cur) {
    cur.classList.add('cur');
    const left = cur.offsetLeft - el.clientWidth / 2 + cur.offsetWidth / 2;
    el.scrollTo({ left, behavior: 'smooth' });
  }
}

$('#moves').addEventListener('click', e => {
  const b = e.target.closest('.mv');
  if (b) go(+b.dataset.i);
});

function positionEval(i) {
  const a = view.game.analysis;
  return a && a.positions ? a.positions[i] : null;
}

function clockAt(color) {
  const g = view.game;
  let clock = g.tc && !g.tc.daily ? g.tc.base : null;
  let spent = null;
  for (let i = 0; i < view.idx; i++) {
    const d = view.derived[i];
    if (d.color === color && d.clock != null) { clock = d.clock; spent = d.spent; }
  }
  return { clock, spent };
}

function playerHtml(color) {
  const g = view.game;
  const p = color === 'w' ? g.white : g.black;
  const { clock, spent } = clockAt(color);
  const isMe = g.myColor === color;
  const low = clock != null && g.tc && clock < Math.max(10, g.tc.base * 0.1);
  const last = view.idx > 0 ? view.derived[view.idx - 1] : null;
  const spentTxt = last && last.color === color && spent != null ? `<span class="spent">${spent < 10 ? spent.toFixed(1).replace('.', ',') : Math.round(spent)} s</span>` : '';
  return `<span class="dot ${color}"></span><span class="name">${esc(p.name)}${isMe ? ' (du)' : ''} <span class="rating">${p.rating || ''}</span></span>${spentTxt}${clock != null && !(g.tc && g.tc.daily) ? `<span class="clock ${low ? 'low' : ''}">${fmtClock(clock)}</span>` : ''}`;
}

function renderPosition({ speak = true } = {}) {
  const g = view.game;
  if (!g) return;
  const i = view.idx;
  const d = i > 0 ? view.derived[i - 1] : null;
  const fen = i > 0 ? g.moves[i - 1].fen : g.initialFen;
  const arrows = [];
  if (d && d.cls && d.before && d.after) {
    const bad = d.score != null && d.score <= -1;
    if (d.mine) {
      if (!d.isBest && d.cls !== 'forced' && d.before.best) arrows.push({ from: d.before.best.slice(0, 2), to: d.before.best.slice(2, 4), color: '#5fb236' });
      if (bad && d.after.best) arrows.push({ from: d.after.best.slice(0, 2), to: d.after.best.slice(2, 4), color: '#e0574f', dashed: true });
    } else if (bad && d.after.best) {
      arrows.push({ from: d.after.best.slice(0, 2), to: d.after.best.slice(2, 4), color: '#5fb236' });
    }
  }
  let check = null;
  if (d && d.inCheck) {
    const turn = fen.split(' ')[1];
    const rows = fen.split(' ')[0].split('/');
    rows.forEach((row, r) => { let f = 0; for (const ch of row) { if (/\d/.test(ch)) f += +ch; else { if (ch === (turn === 'w' ? 'K' : 'k')) check = 'abcdefgh'[f] + (8 - r); f++; } } });
  }
  board.render(fen, { flip: view.flip, lastMove: d ? { from: d.move.from, to: d.move.to } : null, cls: d && d.cls, arrows, check });

  // players: opponent on top
  const bottomColor = view.flip ? 'b' : 'w';
  $('#p-bottom').innerHTML = playerHtml(bottomColor);
  $('#p-top').innerHTML = playerHtml(bottomColor === 'w' ? 'b' : 'w');

  // eval bar
  const ev = positionEval(i);
  const wpW = ev ? wpFor(ev, 'w') : 50;
  const bar = $('#evalbar');
  bar.classList.toggle('flip', view.flip);
  $('#evalfill').style.height = (ev ? wpW : 50) + '%';
  const et = $('#evaltext');
  et.textContent = ev ? evalText(ev).replace('+', '') : '';
  const whiteAhead = wpW >= 50;
  et.className = (whiteAhead !== view.flip) ? '' : 'top';
  et.style.color = whiteAhead ? '#222' : '#eee';
  if (view.flip) et.className = whiteAhead ? 'top' : '';

  // coach
  let html, mood, head = '';
  if (!d) {
    ({ html, mood } = introFor(g, view.summary, !isAnalysed(g)));
  } else {
    ({ html, mood } = commentFor(g, view.derived, i - 1));
    if (d.cls) {
      const c = CLASSES[d.cls];
      const sc = d.score > 0 ? '+' + d.score : String(d.score);
      head = `<span class="chip" style="background:${c.color}"><span class="sym">${c.sym}</span>${c.label}</span>
        <span class="score" title="Bewertung von -3 bis +3" style="color:${c.color}">${sc}</span>
        <span class="evalchip">${d.mine ? 'dein Zug' : 'Gegner'} · ${evalText(d.after)}</span>`;
    }
  }
  view.shownCls = d ? d.cls : null;
  $('#bubble-head').innerHTML = head;
  $('#bubble-text').innerHTML = html;
  setMood($('#avatar'), mood);
  markCurrentMove();
  renderGraphCursor();
  if (speak && cfg.voice) say(html);
}

function go(i) {
  if (!view.game) return;
  const n = view.game.moves.length;
  i = Math.max(0, Math.min(n, i));
  if (i === view.idx) return;
  view.idx = i;
  renderPosition();
}

function nextMistake() {
  const ds = view.derived;
  if (!ds.some(d => d.cls)) { toast('Die Analyse läuft noch …'); return; }
  for (let k = view.idx; k < ds.length; k++) {
    const d = ds[k];
    if ((view.game.myColor ? d.mine : true) && d.score != null && d.score <= -1) { go(k + 1); return; }
  }
  for (let k = 0; k < Math.min(view.idx, ds.length); k++) {
    const d = ds[k];
    if ((view.game.myColor ? d.mine : true) && d.score != null && d.score <= -1) { toast('Von vorne: erster Fehler'); go(k + 1); return; }
  }
  toast('Keine Fehler gefunden 🎉');
}

$('#nav-first').onclick = () => go(0);
$('#nav-prev').onclick = () => go(view.idx - 1);
$('#nav-next').onclick = () => go(view.idx + 1);
$('#nav-last').onclick = () => go(view.game ? view.game.moves.length : 0);
$('#nav-mistake').onclick = nextMistake;
$('#btn-flip').onclick = () => { view.flip = !view.flip; renderPosition({ speak: false }); };
$('#btn-back').onclick = () => history.back();

document.addEventListener('keydown', e => {
  if (!view.game || e.target.matches('input, textarea')) return;
  if (e.key === 'ArrowRight') go(view.idx + 1);
  else if (e.key === 'ArrowLeft') go(view.idx - 1);
  else if (e.key === 'Home') go(0);
  else if (e.key === 'End') go(view.game.moves.length);
});

// swipe on the board: left = next move, right = previous
(() => {
  let x0 = null, y0 = null, t0 = 0;
  const el = $('#board');
  el.addEventListener('touchstart', e => { x0 = e.touches[0].clientX; y0 = e.touches[0].clientY; t0 = Date.now(); }, { passive: true });
  el.addEventListener('touchend', e => {
    if (x0 == null) return;
    const dx = e.changedTouches[0].clientX - x0, dy = e.changedTouches[0].clientY - y0;
    x0 = null;
    if (Math.abs(dx) > 35 && Math.abs(dx) > Math.abs(dy) * 1.3 && Date.now() - t0 < 800) go(view.idx + (dx < 0 ? 1 : -1));
    else if (Math.abs(dx) < 8 && Math.abs(dy) < 8) {
      // tap: right half = next, left half = previous
      const r = el.getBoundingClientRect();
      go(view.idx + (e.changedTouches[0].clientX > r.left + r.width / 2 ? 1 : -1));
    }
  });
  el.addEventListener('click', e => {
    if (e.pointerType === 'touch' || 'ontouchstart' in window) return;
    const r = el.getBoundingClientRect();
    go(view.idx + (e.clientX > r.left + r.width / 2 ? 1 : -1));
  });
})();

// ---------------------------------------------------------------- graph + summary

function renderGraph() {
  const svg = $('#graph');
  const g = view.game;
  const n = g.moves.length;
  const a = g.analysis;
  const W = Math.max(n, 1), H = 100;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  if (!a || !a.positions.some(Boolean)) { svg.innerHTML = `<line x1="0" y1="50" x2="${W}" y2="50" stroke="#666" stroke-width="0.5" vector-effect="non-scaling-stroke"/><g id="gcur"></g>`; return; }
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const ev = a.positions[i];
    if (!ev) break;
    const wp = wpFor(ev, 'w');
    pts.push([i, view.flip ? wp : 100 - wp]);
  }
  const line = pts.map(p => p[0] + ',' + p[1].toFixed(2)).join(' ');
  const fillColor = view.flip ? '#1f1e1b' : '#f2f0eb';
  const lastX = pts[pts.length - 1][0];
  let dots = '';
  view.derived.forEach((d, i) => {
    if (!d.cls || !(d.cls === 'blunder' || d.cls === 'mistake' || d.cls === 'brilliant' || d.cls === 'great')) return;
    const p = pts[i + 1];
    if (!p) return;
    dots += `<ellipse cx="${p[0]}" cy="${p[1]}" rx="${Math.max(0.35, W / 120)}" ry="3.2" fill="${CLASSES[d.cls].color}" stroke="#000" stroke-width="0.6" vector-effect="non-scaling-stroke"/>`;
  });
  svg.innerHTML = `<polygon points="0,${H} ${line} ${lastX},${H}" fill="${fillColor}"/>
    <line x1="0" y1="50" x2="${W}" y2="50" stroke="#888" stroke-width="0.6" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>
    <g id="gcur"></g>${dots}`;
  svg.style.background = view.flip ? '#f2f0eb' : '#403e3a';
  renderGraphCursor();
}

function renderGraphCursor() {
  const c = document.getElementById('gcur');
  if (c) c.innerHTML = `<line x1="${view.idx}" y1="0" x2="${view.idx}" y2="100" stroke="#e8a33d" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
}

$('#graph').addEventListener('click', e => {
  if (!view.game) return;
  const r = e.currentTarget.getBoundingClientRect();
  go(Math.round((e.clientX - r.left) / r.width * view.game.moves.length));
});

function renderSummary() {
  const g = view.game, s = view.summary, el = $('#summary');
  if (!s || !s.analysed) {
    el.innerHTML = `<h3>Auswertung</h3><div class="sum-note">Sobald Stockfish fertig ist, siehst du hier die Genauigkeit und alle Zugbewertungen.</div>`;
    return;
  }
  const me = g.myColor || 'w', opp = me === 'w' ? 'b' : 'w';
  const meName = g.myColor ? 'Du' : g.white.name, oppName = (me === 'w' ? g.black : g.white).name;
  let rows = '';
  for (const [k, c] of Object.entries(CLASSES)) {
    const a = s.counts[me][k] || 0, b2 = s.counts[opp][k] || 0;
    if (!a && !b2 && (k === 'forced' || k === 'brilliant')) continue;
    rows += `<div class="lbl"><i style="background:${c.color}">${c.sym}</i>${c.label} <span class="sub">(${c.score > 0 ? '+' : ''}${c.score})</span></div><div class="c">${a}</div><div class="c">${b2}</div>`;
  }
  // time usage
  let timeNote = '';
  const mine = view.derived.filter(d => d.color === me && d.spent != null);
  if (mine.length && g.tc && !g.tc.daily) {
    const avg = mine.reduce((x, d) => x + d.spent, 0) / mine.length;
    const longest = mine.reduce((m, d) => (d.spent > m.spent ? d : m), mine[0]);
    const quickBad = mine.filter(d => d.spent <= 2.5 && d.score != null && d.score <= -2).length;
    timeNote = `Bedenkzeit: im Schnitt ${avg.toFixed(1).replace('.', ',')} s pro Zug, am längsten bei ${esc(moveLabel(longest))} (${Math.round(longest.spent)} s).`;
    if (quickBad) timeNote += ` ${quickBad} ${quickBad === 1 ? 'Fehler kam' : 'Fehler kamen'} nach weniger als 3 Sekunden – da lohnt sich ein kurzer Kontrollblick.`;
  }
  const cc = g.chesscomAccuracy && g.myColor ? ` <span class="sub">(Chess.com: ${g.chesscomAccuracy[me === 'w' ? 'white' : 'black']?.toFixed?.(1) ?? '–'})</span>` : '';
  el.innerHTML = `<h3>Auswertung</h3>
    <div class="sum-acc"><div><b>${fmtPct(s[me])}</b><span>Genauigkeit ${esc(meName)}${cc}</span></div><div><b>${fmtPct(s[opp])}</b><span>${esc(oppName)}</span></div></div>
    <div class="sum-grid"><div></div><div class="h">${esc(meName)}</div><div class="h">Gegner</div>${rows}</div>
    ${timeNote ? `<div class="sum-note">${timeNote}</div>` : ''}
    <div class="sum-note">Skala: +3 brillant … 0 gut … −3 Patzer. Grundlage ist, wie stark sich deine Gewinnchance durch den Zug verändert (laut Stockfish, Tiefe ${g.analysis.depth}).</div>`;
}

// ---------------------------------------------------------------- voice

function speakable(html) {
  const tmp = document.createElement('div');
  tmp.innerHTML = html.replace(/<b>(.*?)<\/b>/g, (m, inner) => ' ' + inner.split(' ').map(sanToSpeech).join(' ') + ' ');
  return tmp.textContent.replace(/\s+/g, ' ').replace(/[🎉👍]/gu, '').trim();
}

const SPEAK_PIECE = { K: 'König', D: 'Dame', T: 'Turm', L: 'Läufer', S: 'Springer', N: 'Springer', B: 'Läufer', R: 'Turm', Q: 'Dame' };
function sanToSpeech(tok) {
  if (/^\d+\.$/.test(tok) || /^\d+…$/.test(tok)) return '';
  if (/^O-O-O/.test(tok)) return 'lange Rochade';
  if (/^O-O/.test(tok)) return 'kurze Rochade';
  const m = tok.match(/^([KDTLSNBRQ])?([a-h]?[1-8]?)(x)?([a-h][1-8])(=([DTLSQRBN]))?([+#])?$/);
  if (!m) return tok;
  let s = m[1] ? SPEAK_PIECE[m[1]] + ' ' : '';
  if (m[2] && m[1]) s += m[2] + ' ';
  if (m[3]) s += (m[1] ? '' : m[2] + ' ') + 'schlägt ';
  s += m[4];
  if (m[6]) s += ' wird ' + SPEAK_PIECE[m[6]];
  if (m[7] === '+') s += ' Schach';
  if (m[7] === '#') s += ' Matt';
  return s;
}

function say(html) {
  const text = speakable(html);
  if (!text) return;
  talk($('#avatar'), text.length * 65);
  if (bridge && bridge.speak) { bridge.speak(text); return; }
  if (window.speechSynthesis) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'de-DE';
    speechSynthesis.speak(u);
  }
}

function stopSpeaking() {
  if (bridge && bridge.stopSpeaking) bridge.stopSpeaking();
  else if (window.speechSynthesis) speechSynthesis.cancel();
  $('#avatar').classList.remove('talking');
}

function syncVoiceButton() { $('#nav-voice').classList.toggle('on', !!cfg.voice); }
$('#nav-voice').onclick = () => {
  cfg = settings.set({ voice: !cfg.voice });
  syncVoiceButton();
  if (cfg.voice) say($('#bubble-text').innerHTML); else stopSpeaking();
};

// ---------------------------------------------------------------- game menu

$('#btn-gmenu').onclick = e => { e.stopPropagation(); $('#gmenu').classList.toggle('hidden'); };
document.addEventListener('click', e => { if (!e.target.closest('#gmenu')) $('#gmenu').classList.add('hidden'); });
$('#gmenu').addEventListener('click', async e => {
  const act = e.target.dataset.act;
  $('#gmenu').classList.add('hidden');
  const g = view.game;
  if (!g || !act) return;
  if (act === 'chesscom') {
    if (!g.url) { toast('Kein Chess.com-Link vorhanden.'); return; }
    if (bridge && bridge.openUrl) bridge.openUrl(g.url); else window.open(g.url, '_blank');
  } else if (act === 'reanalyse') {
    g.analysis = null; g.summary = null;
    await db.put(g);
    refreshGameData(true);
    enqueue(g.id, true);
  } else if (act === 'copy') {
    try { await navigator.clipboard.writeText(g.pgn); toast('PGN kopiert'); } catch { toast('Kopieren nicht möglich'); }
  } else if (act === 'delete') {
    if (!confirm('Diese Partie löschen?')) return;
    await db.remove(g.id);
    cache.delete(g.id);
    history.back();
  }
});

// ---------------------------------------------------------------- settings, import

function openSheet(id) { $(id).classList.remove('hidden'); history.pushState({ sheet: id }, ''); }
function closeSheets() { document.querySelectorAll('.sheet').forEach(s => s.classList.add('hidden')); }
document.querySelectorAll('.sheet').forEach(s => s.addEventListener('click', e => {
  if (e.target === s || e.target.closest('[data-close]')) history.back();
}));

function fillSettings() {
  $('#set-user').value = cfg.username;
  for (const [sel, key] of [['#set-depth', 'depth'], ['#set-auto', 'autoAnalyse'], ['#set-notation', 'notation']])
    $(sel).querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === cfg[key]));
  $('#set-voice').checked = !!cfg.voice;
  $('#set-notify').checked = !!cfg.notify;
}

$('#btn-settings').onclick = () => { fillSettings(); openSheet('#settings'); };
for (const [sel, key] of [['#set-depth', 'depth'], ['#set-auto', 'autoAnalyse'], ['#set-notation', 'notation']]) {
  $(sel).addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    cfg = settings.set({ [key]: b.dataset.v });
    if (key === 'notation') setNotation(cfg.notation);
    fillSettings();
  });
}
$('#set-user').addEventListener('change', async () => {
  const u = $('#set-user').value.trim();
  if (u.toLowerCase() === cfg.username.toLowerCase()) return;
  cfg = settings.set({ username: u, lastSync: 0, monthsLoaded: 0 });
  // games were stored with "my color" for the old name: recompute it
  for (const g of await allGames()) {
    const lu = u.toLowerCase();
    g.myColor = g.white.name.toLowerCase() === lu ? 'w' : g.black.name.toLowerCase() === lu ? 'b' : null;
    g.outcome = !g.myColor ? null : g.result === '1/2-1/2' ? 'draw' : (g.result === '1-0') === (g.myColor === 'w') ? 'win' : 'loss';
    await db.put(g);
  }
  renderHome();
  sync();
});
$('#set-voice').onchange = e => { cfg = settings.set({ voice: e.target.checked }); syncVoiceButton(); };
$('#set-notify').onchange = async e => {
  cfg = settings.set({ notify: e.target.checked });
  if (bridge && bridge.setSyncState) {
    const newest = Math.max(0, ...(await allGames()).filter(g => g.source === 'chess.com').map(g => g.endTime || 0));
    bridge.setSyncState(cfg.username, newest, cfg.notify);
  }
};
$('#set-clear').onclick = async () => {
  if (!confirm('Wirklich alle gespeicherten Partien und Analysen löschen?')) return;
  await db.clear();
  cache.clear();
  cfg = settings.set({ lastSync: 0, monthsLoaded: 0 });
  history.back();
  renderHome();
};
$('#set-paste').onclick = () => { closeSheets(); history.replaceState({ sheet: '#paste' }, ''); $('#paste').classList.remove('hidden'); };
$('#btn-paste').onclick = () => openSheet('#paste');
$('#paste-go').onclick = async () => {
  const text = $('#paste-text').value;
  if (!text.trim()) return;
  try {
    const id = await importPgn(text);
    $('#paste-text').value = '';
    closeSheets();
    history.replaceState({ game: id }, '');
    openGame(id, { push: false });
  } catch (e) { toast(e.message || 'Diese PGN konnte ich nicht lesen.'); }
};

async function importPgn(text) {
  const rec = gameFromPgn(text, cfg.username);
  const existing = await getGame(rec.id);
  if (existing) return existing.id;
  await db.put(rec);
  cache.set(rec.id, rec);
  return rec.id;
}

$('#ob-go').onclick = () => {
  const u = $('#ob-user').value.trim();
  if (!u) { $('#ob-user').focus(); return; }
  cfg = settings.set({ username: u, lastSync: 0, monthsLoaded: 0 });
  renderHome();
  sync();
};
$('#ob-user').addEventListener('keydown', e => { if (e.key === 'Enter') $('#ob-go').click(); });
$('#btn-sync').onclick = () => cfg.username ? sync() : $('#ob-user').focus();
$('#btn-older').onclick = () => sync({ older: true });

// ---------------------------------------------------------------- shared from the Chess.com app (Android)

async function handleShared(text) {
  if (!text) return;
  if (looksLikePgn(text)) {
    try { const id = await importPgn(text); openGame(id); } catch (e) { toast(e.message); }
    return;
  }
  const m = text.match(/chess\.com\/(?:[a-z]{2}\/)?game\/(?:live\/|daily\/)?(\d+)/i);
  if (!m) { toast('Damit kann ich nichts anfangen – teile die Partie bitte als PGN oder Link.'); return; }
  const id = 'cc-' + m[1];
  if (!(await getGame(id)) && cfg.username) await sync({ quiet: true });
  if (await getGame(id)) openGame(id);
  else toast('Die Partie ist bei Chess.com noch nicht abrufbar – versuch es in ein, zwei Minuten nochmal oder teile sie als PGN.', 5000);
}

window.__checkShared = () => {
  if (bridge && bridge.takeSharedText) {
    const t = bridge.takeSharedText();
    if (t) handleShared(t);
  }
};

// ---------------------------------------------------------------- navigation (Android back button = history.back)

window.addEventListener('popstate', e => {
  const st = e.state || {};
  closeSheets();
  $('#gmenu').classList.add('hidden');
  if (st.game) { if (!view.game || view.game.id !== st.game) openGame(st.game, { push: false }); }
  else if (st.sheet) $(st.sheet).classList.remove('hidden');
  else if (view.game) closeGame();
});

// Returns true if the app handled the back button itself.
window.__onBack = () => {
  if (history.state && (history.state.game || history.state.sheet)) { history.back(); return true; }
  return false;
};

// ---------------------------------------------------------------- start

history.replaceState({}, '');
syncVoiceButton();
renderHome().then(() => {
  window.__checkShared();
  if (cfg.username && Date.now() - cfg.lastSync > 5 * 60 * 1000) sync({ quiet: !!cfg.lastSync });
  // continue analyses that were interrupted
  allGames().then(gs => gs.filter(g => g.analysis && !isAnalysed(g)).forEach(g => enqueue(g.id)));
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && cfg.username && Date.now() - cfg.lastSync > 10 * 60 * 1000) sync({ quiet: true });
  if (document.visibilityState === 'hidden') stopSpeaking();
});

// test hook (used by the automated browser test)
window.__cc = { importPgn, openGame, go, view, analyser, settings };
