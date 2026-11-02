// Busy state for analysis forms: disable the submit button and show the status line,
// so a slow first model load does not look like a dead click.
document.querySelectorAll("form[data-busy-form]").forEach((form) => {
  form.addEventListener("submit", () => {
    const button = form.querySelector("button[type=submit]");
    const busy = form.querySelector(".busy");
    if (button) button.disabled = true;
    if (busy) busy.hidden = false;
  });
});
// Restore buttons when the page is shown again from the back/forward cache.
window.addEventListener("pageshow", () => {
  document.querySelectorAll("form[data-busy-form] button[type=submit]").forEach((b) => (b.disabled = false));
  document.querySelectorAll("form[data-busy-form] .busy").forEach((b) => (b.hidden = true));
});

// Mobile navigation toggle (all pages).
(function () {
  const toggle = document.querySelector("[data-navtoggle]");
  const nav = document.getElementById("mainnav");
  if (!toggle || !nav) return;
  const set = (open) => {
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    nav.classList.toggle("is-open", open);
    document.body.classList.toggle("nav-open", open);
  };
  toggle.addEventListener("click", () => set(toggle.getAttribute("aria-expanded") !== "true"));
  nav.addEventListener("click", (e) => { if (e.target.closest("a")) set(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") set(false); });
  window.matchMedia("(min-width: 761px)").addEventListener("change", (m) => { if (m.matches) set(false); });
})();
