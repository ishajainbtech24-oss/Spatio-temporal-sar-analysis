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
