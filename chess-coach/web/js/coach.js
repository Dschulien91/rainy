// The coach's words: German sentences built from the engine evaluations, the move itself and the clock.
import { Chess } from '../vendor/chess.js';
import { CLASSES, wpFor, hangingValue, PIECE_VALUE } from './analysis.js';

let notation = 'de';
export function setNotation(n) { notation = n; }

const DE = { N: 'S', B: 'L', R: 'T', Q: 'D', K: 'K' };
export function fmtSan(san) {
  if (notation !== 'de') return san;
  return san.replace(/[NBRQK]/g, c => DE[c]);
}

const NAME = { p: 'Bauer', n: 'Springer', b: 'Läufer', r: 'Turm', q: 'Dame', k: 'König' };
const ACC_THE = { p: 'den Bauern', n: 'den Springer', b: 'den Läufer', r: 'den Turm', q: 'die Dame', k: 'den König' };
const ACC_YOUR = { p: 'deinen Bauern', n: 'deinen Springer', b: 'deinen Läufer', r: 'deinen Turm', q: 'deine Dame', k: 'deinen König' };
const NOM_YOUR = { p: 'Dein Bauer', n: 'Dein Springer', b: 'Dein Läufer', r: 'Dein Turm', q: 'Deine Dame', k: 'Dein König' };
const PRON_ACC = { p: 'ihn', n: 'ihn', b: 'ihn', r: 'ihn', q: 'sie', k: 'ihn' };

export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const b = s => `<b>${esc(s)}</b>`;

const lowerFirst = t => t[0].toLowerCase() + t.slice(1);

function pick(arr, seed) { return arr[Math.abs(seed) % arr.length]; }

function moveNo(fen) {
  const parts = fen.split(' ');
  return { n: +parts[5], turn: parts[1] };
}

export function uciToMove(fen, uci) {
  if (!uci) return null;
  try {
    const c = new Chess(fen);
    return c.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
  } catch { return null; }
}

// "12. Sxe5 dxe5 13. Dxd8+" for the first `max` half-moves of an engine line.
export function lineSan(fen, pv, max = 4) {
  if (!pv || !pv.length) return '';
  const c = new Chess(fen);
  const out = [];
  for (let i = 0; i < Math.min(max, pv.length); i++) {
    const { n, turn } = moveNo(c.fen());
    let mv;
    try { mv = c.move({ from: pv[i].slice(0, 2), to: pv[i].slice(2, 4), promotion: pv[i][4] }); } catch { break; }
    if (!mv) break;
    if (turn === 'w') out.push(n + '. ' + fmtSan(mv.san));
    else out.push((i === 0 ? n + '… ' : '') + fmtSan(mv.san));
  }
  return out.join(' ');
}

function standing(wp) {
  if (wp >= 97) return 'du stehst auf Gewinn';
  if (wp >= 80) return 'du stehst klar besser';
  if (wp >= 62) return 'du stehst etwas besser';
  if (wp > 38) return 'die Stellung ist ausgeglichen';
  if (wp > 20) return 'du stehst etwas schlechter';
  if (wp > 3) return 'du stehst klar schlechter';
  return 'die Stellung ist verloren';
}

function fmtClock(s) {
  if (s == null) return '';
  if (s >= 3600) { const h = Math.floor(s / 3600); return h + ':' + String(Math.floor(s % 3600 / 60)).padStart(2, '0') + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }
  if (s < 20) return s.toFixed(1).replace('.', ',') + ' s';
  return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
}
export { fmtClock };

function fmtSpent(s) {
  if (s < 10) return s.toFixed(1).replace('.', ',') + ' Sekunden';
  if (s < 90) return Math.round(s) + ' Sekunden';
  return (s / 60).toFixed(1).replace('.', ',') + ' Minuten';
}

// Pieces of `color` that the other side can win right now and that were safe before the move.
function newlyHanging(fenBefore, fenAfter, color) {
  const after = new Chess(fenAfter), before = new Chess(fenBefore);
  const out = [];
  for (const row of after.board()) for (const sq of row) {
    if (!sq || sq.color !== color || sq.type === 'k' || sq.type === 'p') continue;
    const v = hangingValue(after, sq.square, color);
    if (v >= 2) {
      const prevPiece = before.get(sq.square);
      const wasSame = prevPiece && prevPiece.color === color && prevPiece.type === sq.type;
      if (!wasSame || hangingValue(before, sq.square, color) < 2) out.push({ square: sq.square, type: sq.type, value: v });
    }
  }
  return out.sort((a, b2) => b2.value - a.value);
}

// Material the capturing side nets with `mv` (rough: the captured piece minus what it loses back on that square).
function wins(fen, mv) {
  if (!mv || !mv.captured) return 0;
  const c = new Chess(fen);
  return hangingValue(c, mv.to, mv.color === 'w' ? 'b' : 'w');
}

function mateFor(ev, color) {
  if (!ev || ev.mate == null || ev.mate === 0) return 0;
  const n = color === 'w' ? ev.mate : -ev.mate;
  return n > 0 ? n : 0;
}

export function moodFor(d) {
  if (!d || !d.cls) return 'neutral';
  if (d.mine) {
    if (d.cls === 'brilliant' || d.cls === 'great') return 'wow';
    if (d.cls === 'best' || d.cls === 'excellent') return 'happy';
    if (d.cls === 'inaccuracy') return 'hmm';
    if (d.cls === 'mistake' || d.cls === 'blunder') return 'sad';
    return 'neutral';
  }
  if (d.cls === 'mistake' || d.cls === 'blunder') return 'happy';
  if (d.cls === 'brilliant' || d.cls === 'great') return 'hmm';
  return 'neutral';
}

export function commentFor(game, derived, idx) {
  const d = derived[idx];
  const me = game.myColor || 'w';
  const oppName = (me === 'w' ? game.black.name : game.white.name) || 'Dein Gegner';
  const opp = b(oppName);
  const M = b(d.move.color === 'w' ? `${Math.ceil(d.ply / 2)}. ${fmtSan(d.san)}` : `${Math.ceil(d.ply / 2)}… ${fmtSan(d.san)}`);
  const s = [];
  const seed = d.ply * 7 + d.san.length;
  const last = idx === derived.length - 1;

  if (!d.cls) {
    s.push(d.mine ? `Du spielst ${M}.` : `${opp} spielt ${M}.`);
    if (d.isMate) s.push(d.mine ? 'Schachmatt – gewonnen! 🎉' : 'Schachmatt.');
    if (!game.analysis || !game.analysis.done) s.push('<i>Die Analyse läuft noch – gleich erkläre ich dir den Zug genauer.</i>');
    if (last && game.termination) s.push(`Partie beendet: ${esc(game.termination)}.`);
    return { html: s.join(' '), mood: 'neutral' };
  }

  const prev = derived[idx - 1];
  const bestMove = d.before && d.before.best ? uciToMove(d.fenBefore, d.before.best) : null;
  const bestLine = d.before ? lineSan(d.fenBefore, d.before.pv, 4) : '';
  const reply = d.after && d.after.best ? uciToMove(d.fenAfter, d.after.best) : null;
  const replyLine = d.after ? lineSan(d.fenAfter, d.after.pv, 4) : '';
  const wpMeBefore = wpFor(d.before, me), wpMeAfter = wpFor(d.after, me);
  const bad = d.cls === 'inaccuracy' || d.cls === 'mistake' || d.cls === 'blunder';

  if (d.mine) {
    const heads = {
      brilliant: [`Brillant! ${M} – ein echtes Opfer, und es funktioniert.`, `Wow, ${M}!! Mutig und völlig korrekt.`],
      great: wpMeAfter >= 80
        ? [`Stark gefunden! ${M} ist der Zug, der hier gewinnt.`, `Klasse, ${M}! Nur dieser Zug bringt den Vorteil nach Hause.`]
        : [`Stark gefunden! ${M} ist der einzige wirklich gute Zug.`, `Klasse, ${M}! Genau dieser Zug hält die Stellung zusammen.`],
      best: [`${M} – genau das hätte Stockfish auch gespielt.`, `Perfekt, ${M} ist der beste Zug.`, `${M} ist exakt richtig.`],
      excellent: [`${M} ist ein sehr guter Zug, fast so stark wie der beste.`, `Sehr gut, ${M}.`],
      good: [`${M} ist in Ordnung.`, `${M} – solide, aber es gab etwas Besseres.`],
      forced: [`${M} war der einzige legale Zug.`],
      inaccuracy: [`${M} ist etwas ungenau.`, `Mit ${M} verschenkst du ein bisschen.`],
      mistake: [`${M} ist ein Fehler.`, `Hm, ${M} war nicht gut.`],
      blunder: [`Autsch – ${M} ist ein grober Patzer.`, `${M} ist ein Patzer, der viel kostet.`],
    };
    s.push(d.isMate ? `${M} – Schachmatt!` : pick(heads[d.cls], seed));

    if (d.isMate) s.push('Die Partie ist gewonnen! 🎉');
    else if (!bad) {
      if (d.sacrifice) s.push(`Du bietest ${ACC_YOUR[d.sacrifice.piece]} auf ${d.sacrifice.square} an – nehmen darf ${opp} ${PRON_ACC[d.sacrifice.piece]} aber nicht ungestraft.`);
      const mateNow = mateFor(d.after, me);
      if (mateNow) s.push(`Jetzt hast du ein erzwungenes Matt in ${mateNow}: ${b(replyLine)}.`);
      if (d.cls === 'great' && d.before && d.before.second && d.before.second.move) {
        const alt = uciToMove(d.fenBefore, d.before.second.move);
        if (alt) s.push(`Schon die zweitbeste Möglichkeit ${b(fmtSan(alt.san))} wäre deutlich schwächer gewesen.`);
      }
      if (prev && (prev.cls === 'mistake' || prev.cls === 'blunder') && (d.cls === 'best' || d.cls === 'great' || d.cls === 'excellent' || d.cls === 'brilliant'))
        s.push(`Gut aufgepasst – du bestrafst den Fehler von ${opp}.`);
      else if (d.move.captured && !d.sacrifice && d.cls !== 'great') s.push(`Du schlägst ${ACC_THE[d.move.captured]} auf ${d.move.to}.`);
      if (d.move.flags.includes('k') || d.move.flags.includes('q')) s.push('Rochade: Dein König steht sicher und der Turm kommt ins Spiel.');
      else if (d.ply <= 20 && (d.move.piece === 'n' || d.move.piece === 'b') && (d.move.from[1] === '1' || d.move.from[1] === '8'))
        s.push(pick(['Gute Entwicklung – erst die Leichtfiguren raus, dann rochieren.', 'Eine Figur mehr im Spiel, so soll die Eröffnung aussehen.'], seed));
      if (d.move.promotion) s.push(`Umwandlung in ${d.move.promotion === 'q' ? 'eine Dame' : 'einen ' + NAME[d.move.promotion]}!`);
      if (d.cls === 'good' && bestMove) s.push(`Noch etwas genauer war ${b(fmtSan(bestMove.san))}.`);
      if (d.spent != null && d.spent >= 15 && (d.cls === 'great' || d.cls === 'brilliant' || (d.cls === 'best' && d.onlyMoveGap >= 10)))
        s.push(`Die ${fmtSpent(d.spent)} Bedenkzeit haben sich gelohnt.`);
    } else {
      const hadMate = mateFor(d.before, me);
      if (hadMate && !mateFor(d.after, me)) s.push(`Du hattest Matt in ${hadMate}: ${b(bestLine)}!`);
      else if (prev && (prev.cls === 'mistake' || prev.cls === 'blunder') && d.loss >= 8)
        s.push(`${opp} hatte dir gerade eine Chance gegeben – die hättest du nutzen können.`);

      const theirMate = mateFor(d.after, me === 'w' ? 'b' : 'w');
      const hanging = newlyHanging(d.fenBefore, d.fenAfter, me);
      if (theirMate) s.push(`Jetzt hat ${opp} Matt in ${theirMate}: ${b(replyLine)}.`);
      else if (reply && reply.captured) {
        const own = hanging.find(h => h.square === reply.to);
        if (own && own.square === d.move.to) s.push(`${NOM_YOUR[own.type]} auf ${own.square} steht ungedeckt – ${opp} schlägt ${PRON_ACC[own.type]} mit ${b(fmtSan(reply.san))}.`);
        else if (wins(d.fenAfter, reply) >= 1) s.push(`${opp} kann jetzt mit ${b(fmtSan(reply.san))} ${ACC_YOUR[reply.captured]} auf ${reply.to} schlagen${replyLine.split(' ').length > 2 ? ` – z. B. ${b(replyLine)}` : ''}.`);
        else s.push(`Die starke Antwort ist ${b(replyLine || fmtSan(reply.san))}.`);
      } else if (hanging.length) {
        const h = hanging[0];
        s.push(`Danach hängt ${lowerFirst(NOM_YOUR[h.type])} auf ${h.square}.`);
      } else if (reply) {
        s.push(`Die starke Antwort ist ${b(replyLine || fmtSan(reply.san))}.`);
      }

      if (bestMove && !hadMate) {
        let why = '';
        if (mateFor(d.before, me)) why = ' – das wäre sogar Matt gewesen';
        else if (bestMove.captured && wins(d.fenBefore, bestMove) >= 2) why = ` – damit gewinnst du ${ACC_THE[bestMove.captured]}`;
        const line = bestLine && bestLine.split(' ').length > 2 ? ` (z. B. ${b(bestLine)})` : '';
        s.push(`Besser war ${b(fmtSan(bestMove.san))}${why}${line}.`);
      }
      const sb = standing(wpMeBefore), sa = standing(wpMeAfter);
      if (sb !== sa) s.push(`Vorher: ${sb}. Jetzt: ${sa}.`);

      if (d.ply <= 14 && d.move.piece === 'q' && d.loss >= 4)
        s.push('Tipp: Die Dame früh zu ziehen macht sie zur Zielscheibe – erst Springer und Läufer entwickeln.');
      if (d.ply <= 30 && d.move.piece === 'k' && !d.move.flags.includes('k') && !d.move.flags.includes('q') && !d.inCheckBefore && (d.color === 'w' ? /[KQ]/ : /[kq]/).test(d.fenBefore.split(' ')[2] || ''))
        s.push('Dazu verlierst du dein Rochaderecht.');

      if (d.spent != null && game.tc && !game.tc.daily) {
        const clockLeft = d.clock;
        if (clockLeft != null && clockLeft < Math.max(10, game.tc.base * 0.08))
          s.push(`Mit nur noch ${fmtClock(clockLeft)} auf der Uhr passiert so etwas schnell – Zeitnot vermeiden lohnt sich.`);
        else if (d.spent <= 2.5 && (d.cls === 'mistake' || d.cls === 'blunder'))
          s.push(`Du hast nur ${fmtSpent(d.spent)} überlegt. Bei so einem Zug lohnt sich der Kontrollblick: Welche Schachs, Schlagzüge und Drohungen hat ${opp}?`);
      }
    }
  } else {
    const heads = {
      brilliant: [`${opp} findet mit ${M} ein brillantes Opfer.`],
      great: [`Starker Zug von ${opp}: ${M}.`],
      best: d.ply <= 1 ? [`${opp} eröffnet mit ${M}.`] : [`${opp} spielt ${M} – der beste Zug.`, `${opp} antwortet genau richtig: ${M}.`],
      excellent: [`${opp} spielt ${M}, ein sehr guter Zug.`],
      good: [`${opp} spielt ${M}, solide.`],
      forced: [`${opp} hat mit ${M} keine Wahl.`],
      inaccuracy: [`${M} von ${opp} ist ungenau.`],
      mistake: [`${M} ist ein Fehler von ${opp}!`],
      blunder: [`${opp} patzt mit ${M}!`],
    };
    s.push(d.isMate ? `${opp} setzt mit ${M} matt.` : pick(heads[d.cls], seed));
    if (d.isMate) s.push('Kopf hoch – schauen wir, wo die Partie gekippt ist. Tipp: „Nächster Fehler“.');
    else if (bad) {
      const myMate = mateFor(d.after, me);
      if (myMate) s.push(`Jetzt hast du Matt in ${myMate}: ${b(replyLine)}!`);
      else if (reply && reply.captured && wins(d.fenAfter, reply) >= 1) s.push(`Deine Chance: ${b(fmtSan(reply.san))} gewinnt ${ACC_THE[reply.captured]}${replyLine.split(' ').length > 2 ? ` (z. B. ${b(replyLine)})` : ''}.`);
      else if (reply) s.push(`Deine Chance: ${b(replyLine || fmtSan(reply.san))}.`);
      const nxt = derived[idx + 1];
      if (nxt && nxt.cls) {
        if (nxt.isBest || nxt.cls === 'great' || nxt.cls === 'brilliant') s.push('Und du hast sie genutzt. 👍');
        else if (nxt.loss >= 8) s.push('Leider hast du die Chance im nächsten Zug nicht genutzt.');
      }
    } else {
      if (d.move.captured) s.push(`Dabei fällt ${lowerFirst(NOM_YOUR[d.move.captured])} auf ${d.move.to}${d.inCheck ? ' – mit Schach' : ''}.`);
      else if (d.inCheck) s.push('Schach!');
      const threats = d.inCheck ? [] : newlyHanging(d.fenBefore, d.fenAfter, me);
      if (threats.length) s.push(`Achtung: ${NOM_YOUR[threats[0].type]} auf ${threats[0].square} ist jetzt angegriffen.`);
      if (d.sacrifice) s.push('Das Opfer ist gefährlich – schau genau hin, bevor du nimmst.');
      const theirMate = mateFor(d.after, me === 'w' ? 'b' : 'w');
      if (theirMate) s.push(`${opp} hat jetzt Matt in ${theirMate}.`);
    }
  }

  if (last && game.termination && !d.isMate) s.push(`Partie beendet: ${esc(game.termination)}.`);
  return { html: s.join(' '), mood: moodFor(d) };
}

export function introFor(game, summary, analysing) {
  const me = game.myColor;
  const opp = me === 'b' ? game.white : game.black;
  const s = [];
  if (me) {
    const res = game.outcome === 'win' ? 'gewonnen' : game.outcome === 'loss' ? 'verloren' : game.outcome === 'draw' ? 'remis gespielt' : 'gespielt';
    s.push(`Hallo! Du hast mit ${me === 'w' ? 'Weiß' : 'Schwarz'} gegen ${b(opp.name)}${opp.rating ? ' (' + opp.rating + ')' : ''} ${res}${game.termination ? ' – ' + esc(game.termination) : ''}.`);
  } else {
    s.push(`${b(game.white.name)} gegen ${b(game.black.name)}${game.termination ? ' – ' + esc(game.termination) : ''}.`);
  }
  if (game.opening) s.push(`Eröffnung: ${esc(game.opening)}.`);
  if (summary && summary.analysed && me) {
    const mine = summary[me], theirs = summary[me === 'w' ? 'b' : 'w'];
    s.push(`Deine Genauigkeit: ${b(fmtPct(mine))}, ${esc(opp.name)}: ${fmtPct(theirs)}.`);
    const c = summary.counts[me];
    const errs = (c.blunder || 0) + (c.mistake || 0);
    if (errs === 0) s.push('Keine echten Fehler – stark gespielt!');
    else s.push(`Ich habe ${errs} ${errs === 1 ? 'Fehler' : 'Fehler'} von dir gefunden. Mit „Nächster Fehler“ springst du direkt hin.`);
  } else if (analysing) {
    s.push('Ich analysiere die Partie gerade mit Stockfish – du kannst schon durch die Züge blättern.');
  }
  s.push('Wische über das Brett oder tippe auf die Pfeile, um durch die Züge zu gehen.');
  return { html: s.join(' '), mood: game.outcome === 'win' ? 'happy' : game.outcome === 'loss' ? 'hmm' : 'neutral' };
}

export function moveLabel(d) {
  return `${Math.ceil(d.ply / 2)}${d.color === 'w' ? '.' : '…'} ${fmtSan(d.san)}`;
}

export function fmtPct(x) { return x == null ? '–' : x.toFixed(1).replace('.', ',') + ' %'; }

export { CLASSES };
