// A plain DOM chessboard: 64 squares, piece images, last-move highlight, a badge for the move class and arrows.
import { CLASSES } from './analysis.js';

const FILES = 'abcdefgh';

function parseFen(fen) {
  const rows = fen.split(' ')[0].split('/');
  const board = {};
  rows.forEach((row, r) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += +ch; continue; }
      const color = ch === ch.toUpperCase() ? 'w' : 'b';
      board[FILES[f] + (8 - r)] = color + ch.toLowerCase();
      f++;
    }
  });
  return board;
}

function xy(square, flip) {
  const f = FILES.indexOf(square[0]), r = +square[1] - 1;
  return flip ? { x: 7 - f, y: r } : { x: f, y: 7 - r };
}

export class Board {
  constructor(el) {
    this.el = el;
    el.classList.add('board');
    el.innerHTML = '<div class="sq-layer"></div><div class="piece-layer"></div><svg class="arrow-layer" viewBox="0 0 8 8"></svg><div class="badge-layer"></div>';
    this.sqLayer = el.querySelector('.sq-layer');
    this.pieceLayer = el.querySelector('.piece-layer');
    this.arrowLayer = el.querySelector('.arrow-layer');
    this.badgeLayer = el.querySelector('.badge-layer');
    this.flip = null;
    this.pieces = new Map();   // square -> img, reused so unchanged pieces do not flicker
  }

  drawSquares(flip) {
    let html = '';
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const f = flip ? 7 - x : x, r = flip ? y : 7 - y;
      const sq = FILES[f] + (r + 1);
      const dark = (f + r) % 2 === 0;
      let coord = '';
      if (y === 7) coord += `<span class="cf">${FILES[f]}</span>`;
      if (x === 0) coord += `<span class="cr">${r + 1}</span>`;
      html += `<div class="sq ${dark ? 'd' : 'l'}" data-sq="${sq}" style="left:${x * 12.5}%;top:${y * 12.5}%">${coord}</div>`;
    }
    this.sqLayer.innerHTML = html;
    this.flip = flip;
    this.pieceLayer.innerHTML = '';
    this.pieces.clear();
  }

  render(fen, { flip = false, lastMove = null, cls = null, arrows = [], check = null } = {}) {
    if (this.flip !== flip) this.drawSquares(flip);
    for (const s of this.sqLayer.children) {
      const sq = s.dataset.sq;
      s.classList.toggle('hl', !!lastMove && (sq === lastMove.from || sq === lastMove.to));
      s.classList.toggle('chk', sq === check);
      s.style.removeProperty('--hl');
      if (lastMove && cls && (sq === lastMove.from || sq === lastMove.to)) s.style.setProperty('--hl', CLASSES[cls].color);
    }
    const board = parseFen(fen);
    const next = new Map();
    for (const [sq, pc] of Object.entries(board)) {
      let img = this.pieces.get(sq);
      if (!img || img.dataset.pc !== pc) {
        img = document.createElement('img');
        img.src = `vendor/pieces/${pc}.svg`;
        img.alt = '';
        img.dataset.pc = pc;
        img.className = 'pc';
        const { x, y } = xy(sq, flip);
        img.style.left = x * 12.5 + '%';
        img.style.top = y * 12.5 + '%';
        this.pieceLayer.appendChild(img);
      }
      next.set(sq, img);
    }
    for (const [sq, img] of this.pieces) if (next.get(sq) !== img) img.remove();
    this.pieces = next;

    this.arrowLayer.innerHTML = arrows.map(a => this.arrowSvg(a, flip)).join('');
    this.badgeLayer.innerHTML = '';
    if (lastMove && cls) {
      const c = CLASSES[cls];
      const { x, y } = xy(lastMove.to, flip);
      const b = document.createElement('div');
      b.className = 'badge';
      b.textContent = c.sym;
      b.style.background = c.color;
      b.style.left = `calc(${(x + 1) * 12.5}% - 3.2%)`;
      b.style.top = `calc(${y * 12.5}% - 3%)`;
      if (x === 7) b.style.left = `calc(${(x + 1) * 12.5}% - 5.5%)`;
      if (y === 0) b.style.top = '0.4%';
      this.badgeLayer.appendChild(b);
    }
  }

  arrowSvg({ from, to, color, dashed }, flip) {
    const a = xy(from, flip), b = xy(to, flip);
    const x1 = a.x + 0.5, y1 = a.y + 0.5, x2 = b.x + 0.5, y2 = b.y + 0.5;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
    if (!len) return '';
    const ux = dx / len, uy = dy / len;
    const head = 0.42, w = 0.16;
    const ex = x2 - ux * head, ey = y2 - uy * head;
    const sx = x1 + ux * 0.18, sy = y1 + uy * 0.18;
    const px = -uy, py = ux;
    const pts = [
      [sx + px * w / 2, sy + py * w / 2], [ex + px * w / 2, ey + py * w / 2], [ex + px * head * 0.62, ey + py * head * 0.62],
      [x2 - ux * 0.08, y2 - uy * 0.08], [ex - px * head * 0.62, ey - py * head * 0.62], [ex - px * w / 2, ey - py * w / 2], [sx - px * w / 2, sy - py * w / 2],
    ].map(p => p[0].toFixed(3) + ',' + p[1].toFixed(3)).join(' ');
    return `<polygon points="${pts}" fill="${color}" opacity="${dashed ? 0.55 : 0.8}"/>`;
  }
}
