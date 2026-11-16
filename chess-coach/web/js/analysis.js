// Game analysis: Stockfish looks at every position once (two best lines), everything else - move classes,
// the -3..+3 score, accuracy, the coach's text - is derived from those evaluations without the engine.
import { Chess } from '../vendor/chess.js';

export const ANALYSIS_VERSION = 1;

// ---------- evaluations (always stored from White's point of view) ----------

function terminalEval(chess) {
  if (chess.isCheckmate()) return { mated: chess.turn() };          // side to move is mated
  if (chess.isStalemate() || chess.isInsufficientMaterial() || chess.isThreefoldRepetition() || chess.isDrawByFiftyMoves()) return { draw: true };
  return null;
}

function toWhite(line, turn) {
  const s = turn === 'w' ? 1 : -1;
  return { cp: line.cp == null ? null : line.cp * s, mate: line.mate == null ? null : line.mate * s };
}

// Analyses positions[0..n] of the game. Saves progress through onPosition so a stopped analysis can resume.
export async function analyseGame(game, engine, opts, onPosition, shouldStop) {
  const fens = [game.initialFen, ...game.moves.map(m => m.fen)];
  const a = game.analysis && game.analysis.version === ANALYSIS_VERSION && game.analysis.depth === opts.depth
    ? game.analysis
    : { version: ANALYSIS_VERSION, depth: opts.depth, positions: new Array(fens.length).fill(null), done: false };
  if (a.positions.length !== fens.length) a.positions = new Array(fens.length).fill(null);
  game.analysis = a;
  const chess = new Chess();
  for (let i = 0; i < fens.length; i++) {
    if (a.positions[i]) continue;
    if (shouldStop && shouldStop()) return false;
    chess.load(fens[i]);
    const term = terminalEval(chess);
    if (term) {
      a.positions[i] = { ...term, depth: 0 };
    } else {
      const r = await engine.analyse(fens[i], { depth: opts.depth, movetime: opts.movetime, multipv: 2 });
      const turn = chess.turn();
      const l1 = r.lines[0], l2 = r.lines[1];
      a.positions[i] = l1 ? {
        ...toWhite(l1, turn), best: l1.pv[0] || r.best, pv: l1.pv,
        second: l2 ? { ...toWhite(l2, turn), move: l2.pv[0] } : null,
        depth: r.depth,
      } : { cp: 0, mate: null, best: r.best, pv: [r.best], second: null, depth: r.depth };
    }
    if (onPosition) await onPosition(i, fens.length);
  }
  a.done = true;
  return true;
}

// ---------- win chance and classes ----------

// Lichess' conversion of centipawns to a win chance (0..100).
export function winPct(cp) {
  return 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * cp)) - 1);
}

// Win chance of `color` in an evaluated position.
export function wpFor(ev, color) {
  if (!ev) return 50;
  let w;
  if (ev.mated) w = ev.mated === 'w' ? 0 : 100;
  else if (ev.draw) w = 50;
  else if (ev.mate != null) w = ev.mate > 0 ? 100 : ev.mate < 0 ? 0 : 50;
  else w = winPct(Math.max(-2000, Math.min(2000, ev.cp || 0)));
  return color === 'w' ? w : 100 - w;
}

export function secondWp(ev, color) {
  return ev && ev.second ? wpFor(ev.second, color) : null;
}

export const CLASSES = {
  brilliant:  { score: 3,  label: 'Brillant',   sym: '!!', color: '#1baca6' },
  great:      { score: 2,  label: 'Stark',      sym: '!',  color: '#5c8bb0' },
  best:       { score: 1,  label: 'Bester Zug', sym: '★',  color: '#81b64c' },
  excellent:  { score: 1,  label: 'Exzellent',  sym: '👍', color: '#81b64c' },
  good:       { score: 0,  label: 'Gut',        sym: '✓',  color: '#95b776' },
  forced:     { score: 0,  label: 'Erzwungen',  sym: '→',  color: '#97958f' },
  inaccuracy: { score: -1, label: 'Ungenau',    sym: '?!', color: '#f7c045' },
  mistake:    { score: -2, label: 'Fehler',     sym: '?',  color: '#e58f2a' },
  blunder:    { score: -3, label: 'Patzer',     sym: '??', color: '#ca3431' },
};

export const PIECE_VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

function lowestAttacker(chess, square, color) {
  let low = Infinity;
  for (const sq of chess.attackers(square, color)) {
    const p = chess.get(sq);
    if (p) low = Math.min(low, p.type === 'k' ? 100 : PIECE_VALUE[p.type]);
  }
  return low;
}

// Material the opponent can win on `square` right away (rough static exchange: one capture and one recapture).
export function hangingValue(chess, square, owner) {
  const piece = chess.get(square);
  if (!piece || piece.type === 'k') return 0;
  const opp = owner === 'w' ? 'b' : 'w';
  const v = PIECE_VALUE[piece.type];
  const att = lowestAttacker(chess, square, opp);
  if (att === Infinity) return 0;
  const defended = chess.attackers(square, owner).length > 0;
  if (!defended) return v;
  return att < 100 ? Math.max(0, v - att) : 0;
}

// Is the move a real sacrifice (the moved piece can be taken and the mover loses material if it is)?
function sacrificeInfo(fenAfter, move) {
  const type = move.promotion || move.piece;
  if (type === 'p' || type === 'k') return null;
  const chess = new Chess(fenAfter);
  const lost = hangingValue(chess, move.to, move.color);
  const gained = move.captured ? PIECE_VALUE[move.captured] : 0;
  if (lost - gained >= 2) return { square: move.to, piece: type, value: lost - gained };
  return null;
}

// Everything the board, the move list and the coach need for every half-move.
export function deriveMoves(game) {
  const a = game.analysis;
  const pos = a ? a.positions : [];
  const chess = new Chess(game.initialFen);
  const out = [];
  const lastClock = { w: game.tc && !game.tc.daily ? game.tc.base : null, b: game.tc && !game.tc.daily ? game.tc.base : null };
  for (let i = 0; i < game.moves.length; i++) {
    const m = game.moves[i];
    const fenBefore = chess.fen();
    const legalCount = chess.moves().length;
    const inCheckBefore = chess.inCheck();
    const mv = chess.move(m.san);
    const color = mv.color;
    const before = pos[i], after = pos[i + 1];
    const d = {
      ply: i + 1, color, mine: game.myColor ? color === game.myColor : color === 'w',
      san: mv.san, uci: m.uci, move: mv, fenBefore, fenAfter: m.fen,
      clock: m.clock, spent: null, before, after, cls: null, score: null,
      inCheck: chess.inCheck(), isMate: chess.isCheckmate(), inCheckBefore,
    };
    if (m.clock != null && lastClock[color] != null) {
      d.spent = Math.max(0, lastClock[color] - m.clock + (game.tc ? game.tc.inc : 0));
      if (i < 2) d.spent = null;  // the first move of each side does not run the clock on Chess.com
    }
    if (m.clock != null) lastClock[color] = m.clock;
    if (before && after) {
      const bestWp = wpFor(before, color);
      const afterWp = wpFor(after, color);
      const isBest = before.best === m.uci;
      const loss = isBest ? 0 : Math.max(0, bestWp - afterWp);
      d.wpBefore = bestWp;
      d.wpAfter = afterWp;
      d.loss = loss;
      d.isBest = isBest;
      d.accuracy = Math.max(0, Math.min(100, 103.1668 * Math.exp(-0.04354 * loss) - 3.1669));
      const s2 = secondWp(before, color);
      d.onlyMoveGap = s2 == null ? 0 : bestWp - s2;
      d.sacrifice = loss <= 2 ? sacrificeInfo(m.fen, mv) : null;
      const prev = out[i - 1];
      const recapture = prev && prev.move.captured && prev.move.to === mv.to;
      let cls;
      if (legalCount === 1) cls = 'forced';
      else if (d.sacrifice && loss <= 2 && afterWp >= 45 && bestWp < 97) cls = 'brilliant';
      else if (loss <= 2 && d.onlyMoveGap >= 18 && !recapture && bestWp >= 25 && bestWp <= 97 && !isTrivialMate(before)) cls = 'great';
      else if (isBest || loss <= 1) cls = 'best';
      else if (loss <= 3) cls = 'excellent';
      else if (loss <= 7) cls = 'good';
      else if (loss <= 14) cls = 'inaccuracy';
      else if (loss <= 25) cls = 'mistake';
      else cls = 'blunder';
      // A move that still wins with a big margin is never a "blunder", only a missed shortcut.
      if ((cls === 'mistake' || cls === 'blunder') && afterWp >= 90) cls = 'inaccuracy';
      // Walking into a forced mate is always worth a warning, even in a position that was lost anyway.
      const opp = color === 'w' ? 'b' : 'w';
      const mateAgainst = m => m && m.mate != null && (opp === 'w' ? m.mate > 0 : m.mate < 0) ? Math.abs(m.mate) : 0;
      if (cls !== 'forced' && !isBest && mateAgainst(after) && !mateAgainst(before) && CLASSES[cls].score > -2) {
        cls = mateAgainst(after) <= 2 ? 'blunder' : 'mistake';
      }
      d.cls = cls;
      d.score = CLASSES[cls].score;
    }
    out.push(d);
  }
  return out;
}

function isTrivialMate(ev) { return ev && ev.mate != null && Math.abs(ev.mate) <= 1; }

export function summarize(game, derived) {
  const res = { w: null, b: null, counts: { w: {}, b: {} }, analysed: false };
  const acc = { w: [], b: [] };
  for (const d of derived) {
    if (!d.cls) continue;
    res.analysed = true;
    acc[d.color].push(d.accuracy);
    res.counts[d.color][d.cls] = (res.counts[d.color][d.cls] || 0) + 1;
  }
  for (const c of ['w', 'b']) {
    const xs = acc[c];
    if (!xs.length) continue;
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    const harm = xs.length / xs.reduce((s, x) => s + 1 / Math.max(x, 5), 0);
    res[c] = Math.round((0.75 * mean + 0.25 * harm) * 10) / 10;
  }
  return res;
}

// White-POV number for the eval bar / text: "+1.3", "M3", "-M2".
export function evalText(ev) {
  if (!ev) return '';
  if (ev.mated) return ev.mated === 'w' ? '0-1' : '1-0';
  if (ev.draw) return '½';
  if (ev.mate != null) return (ev.mate > 0 ? 'M' : '-M') + Math.abs(ev.mate);
  const v = (ev.cp || 0) / 100;
  return (v > 0 ? '+' : '') + v.toFixed(1);
}
