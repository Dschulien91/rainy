// Turns Chess.com API games or pasted PGN into the record the app stores, and fetches games from the public
// Chess.com API (no login needed: https://api.chess.com/pub/player/<name>/games/archives).
import { Chess } from '../vendor/chess.js';
import { parsePgn, parseTimeControl } from './pgn.js';

const API = 'https://api.chess.com/pub/player/';

// In the Android app the request goes through the app itself (/proxy), so the browser's CORS rules do not apply.
function apiUrl(url) {
  return window.AndroidBridge ? '/proxy?u=' + encodeURIComponent(url) : url;
}

async function getJson(url) {
  const r = await fetch(apiUrl(url), { cache: 'no-store' });
  if (r.status === 404) { const e = new Error('not found'); e.status = 404; throw e; }
  if (!r.ok) { const e = new Error('HTTP ' + r.status); e.status = r.status; throw e; }
  return r.json();
}

export async function fetchArchiveList(user) {
  const j = await getJson(API + encodeURIComponent(user.toLowerCase()) + '/games/archives');
  return j.archives || [];
}

export async function fetchArchive(url) {
  const j = await getJson(url);
  return j.games || [];
}

const RESULT_TEXT = {
  checkmated: 'Schachmatt', resigned: 'Aufgabe', timeout: 'Zeit abgelaufen', abandoned: 'Partie verlassen',
  agreed: 'Remis vereinbart', repetition: 'Remis durch Zugwiederholung', stalemate: 'Patt',
  insufficient: 'Remis – zu wenig Material', '50move': 'Remis – 50-Züge-Regel',
  timevsinsufficient: 'Remis – Zeit abgelaufen, aber zu wenig Material zum Mattsetzen',
};

function terminationFromResults(w, b) {
  const loser = w === 'win' ? b : b === 'win' ? w : null;
  if (loser) return RESULT_TEXT[loser] || loser;
  return RESULT_TEXT[w] || RESULT_TEXT[b] || 'Remis';
}

function terminationFromHeader(t) {
  if (!t) return '';
  const s = t.toLowerCase();
  if (s.includes('checkmate')) return 'Schachmatt';
  if (s.includes('resignation')) return 'Aufgabe';
  if (s.includes('abandon')) return 'Partie verlassen';
  if (s.includes('timeout vs insufficient')) return RESULT_TEXT.timevsinsufficient;
  if (s.includes('on time')) return 'Zeit abgelaufen';
  if (s.includes('repetition')) return RESULT_TEXT.repetition;
  if (s.includes('agreement')) return RESULT_TEXT.agreed;
  if (s.includes('stalemate')) return 'Patt';
  if (s.includes('insufficient')) return RESULT_TEXT.insufficient;
  if (s.includes('50')) return RESULT_TEXT['50move'];
  return t;
}

export function openingFromEcoUrl(url) {
  if (!url) return '';
  const last = decodeURIComponent(url.split('/').pop() || '');
  const parts = last.split('-');
  const out = [];
  for (const p of parts) { if (/^\d+\./.test(p)) break; out.push(p); }
  return out.join(' ').replace(/\s+/g, ' ').trim();
}

function utcMs(date, time) {
  if (!date || !time || /\?/.test(date)) return null;
  const d = date.split('.').map(Number), t = time.split(':').map(Number);
  if (d.length < 3 || t.length < 2) return null;
  return Date.UTC(d[0], d[1] - 1, d[2], t[0], t[1], t[2] || 0);
}

function timeClassOf(tc) {
  if (!tc) return '';
  if (tc.daily) return 'daily';
  const est = tc.base + 40 * tc.inc;
  if (est < 180) return 'bullet';
  if (est < 600) return 'blitz';
  return 'rapid';
}

// Replays the SAN moves once so every move has its UCI form and the position after it.
function buildMoves(parsed, startFen) {
  const chess = new Chess(startFen || undefined);
  const initialFen = chess.fen();
  const moves = [];
  for (const m of parsed.moves) {
    let mv;
    try { mv = chess.move(m.san); } catch { mv = null; }
    if (!mv) break;
    moves.push({ san: mv.san, uci: mv.from + mv.to + (mv.promotion || ''), fen: chess.fen(), clock: m.clock });
  }
  return { initialFen, moves };
}

function outcomeFor(color, result) {
  if (!color) return null;
  if (result === '1/2-1/2') return 'draw';
  if (result === '1-0') return color === 'w' ? 'win' : 'loss';
  if (result === '0-1') return color === 'b' ? 'win' : 'loss';
  return null;
}

export function gameFromApi(g, user) {
  const parsed = parsePgn(g.pgn || '');
  const h = parsed.headers;
  const startFen = g.initial_setup && !g.initial_setup.startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq')
    ? g.initial_setup : (h.SetUp === '1' ? h.FEN : null);
  const { initialFen, moves } = buildMoves(parsed, startFen);
  const tc = parseTimeControl(g.time_control || h.TimeControl);
  const u = (user || '').toLowerCase();
  const myColor = g.white.username.toLowerCase() === u ? 'w' : g.black.username.toLowerCase() === u ? 'b' : null;
  const result = h.Result || (g.white.result === 'win' ? '1-0' : g.black.result === 'win' ? '0-1' : '1/2-1/2');
  const id = 'cc-' + (((g.url || '').match(/\/(\d+)$/) || [])[1] || g.uuid);
  return {
    id, source: 'chess.com', url: g.url || h.Link || '',
    white: { name: g.white.username, rating: g.white.rating || null },
    black: { name: g.black.username, rating: g.black.rating || null },
    myColor, result, outcome: outcomeFor(myColor, result),
    termination: terminationFromResults(g.white.result, g.black.result),
    timeControl: g.time_control || h.TimeControl || '', tc, timeClass: g.time_class || timeClassOf(tc),
    rated: !!g.rated,
    startTime: (g.start_time ? g.start_time * 1000 : null) || utcMs(h.UTCDate, h.UTCTime) || null,
    endTime: g.end_time ? g.end_time * 1000 : utcMs(h.EndDate, h.EndTime) || Date.now(),
    opening: openingFromEcoUrl(g.eco || h.ECOUrl), eco: h.ECO || '',
    initialFen, moves, pgn: g.pgn || '',
    chesscomAccuracy: g.accuracies || null,
    analysis: null, summary: null, importedAt: Date.now(),
  };
}

export function gameFromPgn(text, user) {
  const parsed = parsePgn(text);
  const h = parsed.headers;
  const { initialFen, moves } = buildMoves(parsed, h.SetUp === '1' ? h.FEN : null);
  if (!moves.length) throw new Error('Keine gültigen Züge in der PGN gefunden.');
  const tc = parseTimeControl(h.TimeControl);
  const u = (user || '').toLowerCase();
  const white = h.White || 'Weiß', black = h.Black || 'Schwarz';
  const myColor = u && white.toLowerCase() === u ? 'w' : u && black.toLowerCase() === u ? 'b' : null;
  const result = parsed.result !== '*' ? parsed.result : (h.Result || '*');
  const link = h.Link || (h.Site && /^https?:/.test(h.Site) ? h.Site : '');
  const linkId = (link.match(/\/game\/(?:live|daily)\/(\d+)/) || [])[1];
  let hash = 0;
  for (const m of moves) for (let k = 0; k < m.san.length; k++) hash = (hash * 31 + m.san.charCodeAt(k)) | 0;
  return {
    id: linkId ? 'cc-' + linkId : 'pgn-' + (h.Date || '') + '-' + white + '-' + black + '-' + (hash >>> 0).toString(36),
    source: 'pgn', url: link,
    white: { name: white, rating: +h.WhiteElo || null },
    black: { name: black, rating: +h.BlackElo || null },
    myColor, result, outcome: outcomeFor(myColor, result),
    termination: terminationFromHeader(h.Termination) || (result === '1/2-1/2' ? 'Remis' : ''),
    timeControl: h.TimeControl || '', tc, timeClass: timeClassOf(tc), rated: false,
    startTime: utcMs(h.UTCDate || h.Date, h.UTCTime || h.StartTime),
    endTime: utcMs(h.EndDate || h.UTCDate || h.Date, h.EndTime) || Date.now(),
    opening: openingFromEcoUrl(h.ECOUrl) || h.Opening || '', eco: h.ECO || '',
    initialFen, moves, pgn: text,
    chesscomAccuracy: null, analysis: null, summary: null, importedAt: Date.now(),
  };
}

export function timeClassLabel(tcClass) {
  return { bullet: 'Bullet', blitz: 'Blitz', rapid: 'Schnellschach', daily: 'Täglich', classical: 'Klassisch' }[tcClass] || '';
}

export function timeControlLabel(game) {
  const tc = game.tc;
  if (!tc) return '';
  if (tc.daily) return Math.round(tc.base / 86400) + ' Tag' + (tc.base >= 172800 ? 'e' : '') + ' / Zug';
  const min = tc.base >= 60 ? (tc.base / 60) + ' min' : tc.base + ' s';
  return tc.inc ? `${min} + ${tc.inc} s` : min;
}
