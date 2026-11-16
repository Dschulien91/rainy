// Minimal PGN reader: headers, main line SAN moves and the clock comments ({[%clk 0:02:59.9]}) Chess.com writes.
// Variations are skipped, only the first game in the text is read.

export function parsePgn(text) {
  text = String(text || '').replace(/\r\n?/g, '\n').replace(/^﻿/, '');
  const headers = {};
  const headerRe = /^\s*\[(\w+)\s+"((?:[^"\\]|\\.)*)"\s*\]\s*$/;
  const lines = text.split('\n');
  let i = 0;
  while (i < lines.length && lines[i].trim() === '') i++;
  for (; i < lines.length; i++) {
    const m = lines[i].match(headerRe);
    if (m) headers[m[1]] = m[2].replace(/\\(.)/g, '$1');
    else if (lines[i].trim() !== '') break;
  }
  const body = lines.slice(i).join('\n');
  const moves = [];
  let result = headers.Result || '*';
  let p = 0, depth = 0;
  while (p < body.length) {
    const ch = body[p];
    if (ch === '{') {
      const end = body.indexOf('}', p + 1);
      const comment = body.slice(p + 1, end < 0 ? body.length : end);
      p = end < 0 ? body.length : end + 1;
      if (depth === 0 && moves.length) {
        const clk = comment.match(/\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]/);
        if (clk) moves[moves.length - 1].clock = (+clk[1]) * 3600 + (+clk[2]) * 60 + parseFloat(clk[3]);
      }
      continue;
    }
    if (ch === ';') { const end = body.indexOf('\n', p); p = end < 0 ? body.length : end + 1; continue; }
    if (ch === '(') { depth++; p++; continue; }
    if (ch === ')') { depth = Math.max(0, depth - 1); p++; continue; }
    if (/\s/.test(ch)) { p++; continue; }
    let q = p;
    while (q < body.length && !/[\s{}();]/.test(body[q])) q++;
    let tok = body.slice(p, q);
    p = q;
    if (depth > 0) continue;
    if (/^(1-0|0-1|1\/2-1\/2|\*)$/.test(tok)) { result = tok; break; }
    if (/^\$\d+$/.test(tok)) continue;
    tok = tok.replace(/^\d+\.+/, '');
    if (!tok || /^\d+$/.test(tok)) continue;
    tok = tok.replace(/[!?]+$/, '').replace(/^0-0-0/, 'O-O-O').replace(/^0-0/, 'O-O');
    if (tok) moves.push({ san: tok, clock: null });
  }
  return { headers, moves, result };
}

export function looksLikePgn(text) {
  return /\[\s*(Event|White|Black|Site)\s+"/.test(text) || /(^|\s)1\.\s*[a-hNBRQKO]/.test(text);
}

// "180+2" -> {base:180, inc:2}; "1/86400" (daily) -> {daily:true, base:86400}
export function parseTimeControl(tc) {
  if (!tc || tc === '-') return null;
  const d = tc.match(/^1\/(\d+)$/);
  if (d) return { daily: true, base: +d[1], inc: 0 };
  const m = tc.match(/^(\d+)(?:\+(\d+(?:\.\d+)?))?/);
  if (!m) return null;
  return { daily: false, base: +m[1], inc: m[2] ? +m[2] : 0 };
}
