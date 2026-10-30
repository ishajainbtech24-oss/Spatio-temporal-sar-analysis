// Landing page: radar / flood-map comparison and the topbar background on scroll.
(function () {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- comparison
  const hero = document.querySelector("[data-compare]");
  const range = document.querySelector("[data-compare-range]");
  if (hero && range) {
    const set = (v) => hero.style.setProperty("--pos", `${v}%`);
    range.addEventListener("input", () => {
      hero.classList.remove("is-intro");
      set(range.value);
    });
    // One opening sweep from the radar view to the resting split, so the reveal is noticed.
    if (!reduce) {
      hero.classList.add("is-intro");
      set(100);
      requestAnimationFrame(() => requestAnimationFrame(() => set(range.value)));
      setTimeout(() => hero.classList.remove("is-intro"), 1900);
    }
  }

  // ---------------------------------------------------------------- topbar
  const bar = document.querySelector("[data-topbar]");
  if (bar && document.body.classList.contains("landing")) {
    const onScroll = () => bar.classList.toggle("is-solid", window.scrollY > 40);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }
})();
