// Games live in IndexedDB on the phone (one record per game, analysis included). Settings in localStorage.
const DB_NAME = 'chess-coach';
let dbp = null;

function open() {
  if (dbp) return dbp;
  dbp = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('games')) {
        const s = db.createObjectStore('games', { keyPath: 'id' });
        s.createIndex('endTime', 'endTime');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbp;
}

function tx(mode, fn) {
  return open().then(db => new Promise((resolve, reject) => {
    const t = db.transaction('games', mode);
    const store = t.objectStore('games');
    let out;
    Promise.resolve(fn(store)).then(v => { out = v; });
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  }));
}

const wrap = req => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });

export const db = {
  all: () => tx('readonly', s => wrap(s.getAll())).then(list => list.sort((a, b) => (b.endTime || 0) - (a.endTime || 0))),
  get: id => tx('readonly', s => wrap(s.get(id))),
  put: game => tx('readwrite', s => { s.put(game); }),
  putMany: games => tx('readwrite', s => { for (const g of games) s.put(g); }),
  remove: id => tx('readwrite', s => { s.delete(id); }),
  clear: () => tx('readwrite', s => { s.clear(); }),
  keys: () => tx('readonly', s => wrap(s.getAllKeys())),
};

const DEFAULTS = {
  username: '', depth: 'normal', autoAnalyse: 'charging', notation: 'de', voice: false,
  notify: false, lastSync: 0, monthsLoaded: 0,
};

export const settings = {
  get() {
    let s = {};
    try { s = JSON.parse(localStorage.getItem('cc-settings') || '{}'); } catch { s = {}; }
    return { ...DEFAULTS, ...s };
  },
  set(patch) {
    const s = { ...settings.get(), ...patch };
    try { localStorage.setItem('cc-settings', JSON.stringify(s)); } catch { /* ignore */ }
    return s;
  },
};
