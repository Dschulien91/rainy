// "Ole", the coach: an owl in SVG whose face follows the move (moods: neutral, happy, wow, hmm, sad).
export const AVATAR_SVG = `
<svg class="owl" viewBox="0 0 100 100" aria-hidden="true">
  <g class="owl-body">
    <path class="tuft" d="M22 30 L18 8 L38 22 Z"/>
    <path class="tuft" d="M78 30 L82 8 L62 22 Z"/>
    <g class="wing wing-l"><ellipse cx="18" cy="66" rx="10" ry="20"/></g>
    <g class="wing wing-r"><ellipse cx="82" cy="66" rx="10" ry="20"/></g>
    <ellipse class="body" cx="50" cy="58" rx="34" ry="38"/>
    <ellipse class="belly" cx="50" cy="74" rx="21" ry="20"/>
    <path class="belly-mark" d="M42 70 q4 4 8 0 q4 4 8 0 M44 80 q3 3 6 0 q3 3 6 0" fill="none"/>
    <circle class="eye" cx="36" cy="44" r="12"/>
    <circle class="eye" cx="64" cy="44" r="12"/>
    <g class="pupils"><circle class="pupil" cx="36" cy="45" r="5.5"/><circle class="pupil" cx="64" cy="45" r="5.5"/>
      <circle class="shine" cx="38" cy="42.5" r="1.8"/><circle class="shine" cx="66" cy="42.5" r="1.8"/></g>
    <circle class="rim" cx="36" cy="44" r="13.5"/><circle class="rim" cx="64" cy="44" r="13.5"/>
    <path class="rim" d="M49.5 44 h1"/>
    <path class="brow brow-l" d="M25 29 Q36 24 46 29"/>
    <path class="brow brow-r" d="M54 29 Q64 24 75 29"/>
    <ellipse class="blush" cx="27" cy="60" rx="5" ry="2.6"/><ellipse class="blush" cx="73" cy="60" rx="5" ry="2.6"/>
    <g class="beak"><path class="beak-top" d="M44 56 L56 56 L50 64 Z"/><path class="beak-low" d="M46 60 L54 60 L50 66 Z"/></g>
    <path class="sweat" d="M82 30 q4 7 0 9 q-4 -2 0 -9 Z"/>
    <g class="sparkle"><path d="M10 18 l2 5 l5 2 l-5 2 l-2 5 l-2 -5 l-5 -2 l5 -2 Z"/><path d="M88 46 l1.5 3.5 l3.5 1.5 l-3.5 1.5 l-1.5 3.5 l-1.5 -3.5 l-3.5 -1.5 l3.5 -1.5 Z"/></g>
  </g>
</svg>`;

export function setMood(el, mood) {
  el.dataset.mood = mood || 'neutral';
}

let talkTimer = null;
export function talk(el, ms) {
  el.classList.add('talking');
  clearTimeout(talkTimer);
  talkTimer = setTimeout(() => el.classList.remove('talking'), Math.min(ms, 15000));
}
