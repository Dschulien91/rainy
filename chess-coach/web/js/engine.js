// Stockfish 19 (lite, single-threaded WASM) in a Web Worker, spoken to over UCI.
// The worker is started only when a game is analysed and stopped again after a short idle time to save battery.
const ENGINE_URL = 'vendor/stockfish/stockfish-19-lite-single.js';
const IDLE_MS = 20000;

export class Engine {
  constructor() {
    this.worker = null;
    this.ready = null;
    this.listener = null;
    this.idleTimer = null;
    this.chain = Promise.resolve();
  }

  start() {
    if (this.ready) return this.ready;
    this.worker = new Worker(ENGINE_URL);
    this.worker.onmessage = e => { if (this.listener) this.listener(String(e.data)); };
    this.ready = new Promise((resolve, reject) => {
      this.worker.onerror = err => reject(err.message || err);
      this.listener = line => {
        if (line === 'uciok') {
          this.send('setoption name Hash value 16');
          this.send('isready');
        } else if (line === 'readyok') {
          this.listener = null;
          resolve();
        }
      };
      this.send('uci');
    });
    return this.ready;
  }

  send(cmd) { this.worker.postMessage(cmd); }

  stop() {
    clearTimeout(this.idleTimer);
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.ready = null;
    this.listener = null;
  }

  // One search. Scores come back from the side to move's point of view; the caller converts them.
  analyse(fen, { depth = 14, multipv = 2, movetime = 4000 } = {}) {
    const job = this.chain.then(() => this._analyse(fen, depth, multipv, movetime));
    this.chain = job.catch(() => {});
    return job;
  }

  async _analyse(fen, depth, multipv, movetime) {
    clearTimeout(this.idleTimer);
    await this.start();
    const lines = [];
    let reached = 0;
    const result = await new Promise(resolve => {
      this.listener = line => {
        if (line.startsWith('info ') && line.includes(' pv ') && line.includes(' score ')) {
          if (/ (lowerbound|upperbound) /.test(line)) return;
          const d = +(line.match(/ depth (\d+)/) || [])[1] || 0;
          const k = +(line.match(/ multipv (\d+)/) || [0, 1])[1];
          const sc = line.match(/ score (cp|mate) (-?\d+)/);
          const pv = line.split(' pv ')[1].trim().split(/\s+/);
          if (!sc) return;
          lines[k - 1] = { depth: d, cp: sc[1] === 'cp' ? +sc[2] : null, mate: sc[1] === 'mate' ? +sc[2] : null, pv: pv.slice(0, 10) };
          reached = Math.max(reached, d);
        } else if (line.startsWith('bestmove')) {
          this.listener = null;
          resolve({ depth: reached, best: line.split(/\s+/)[1], lines: lines.filter(Boolean) });
        }
      };
      this.send('setoption name MultiPV value ' + multipv);
      this.send('position fen ' + fen);
      this.send(`go depth ${depth} movetime ${movetime}`);
    });
    this.idleTimer = setTimeout(() => this.stop(), IDLE_MS);
    return result;
  }
}

export const DEPTHS = {
  fast: { depth: 10, movetime: 1500, label: 'Schnell' },
  normal: { depth: 14, movetime: 3000, label: 'Normal' },
  deep: { depth: 18, movetime: 6000, label: 'Gründlich' },
};
