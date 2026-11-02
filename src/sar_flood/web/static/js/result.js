// Results page behaviour. Everything here only re-arranges data the server already
// rendered: method tabs, doughnut tooltips, overlay opacity, full-size viewer, image fallbacks.
(function () {
  "use strict";

  // ---------------------------------------------------------------- method tabs
  const tabs = Array.from(document.querySelectorAll(".seg__btn"));
  function selectMethod(key, focus) {
    tabs.forEach((t) => {
      const on = t.dataset.method === key;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    document.querySelectorAll(".rx-method, .tech__method").forEach((el) => {
      el.hidden = el.dataset.method !== key;
    });
    document.dispatchEvent(new CustomEvent("method:change", { detail: key }));
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectMethod(t.dataset.method));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      selectMethod(next.dataset.method, true);
    });
  });

  // ---------------------------------------------------------------- doughnut tooltips
  document.querySelectorAll(".donut-wrap").forEach((wrap) => {
    const tip = wrap.querySelector(".donut-tip");
    wrap.querySelectorAll(".donut__seg").forEach((seg) => {
      const show = (x, y) => {
        tip.textContent = seg.dataset.tip;
        tip.hidden = false;
        const box = wrap.getBoundingClientRect();
        tip.style.left = `${x - box.left}px`;
        tip.style.top = `${y - box.top}px`;
        seg.classList.add("is-active");
      };
      const hide = () => { tip.hidden = true; seg.classList.remove("is-active"); };
      seg.addEventListener("mousemove", (e) => show(e.clientX, e.clientY));
      seg.addEventListener("mouseleave", hide);
      seg.addEventListener("focus", () => {
        const r = seg.getBoundingClientRect();
        show(r.left + r.width / 2, r.top + r.height / 2);
      });
      seg.addEventListener("blur", hide);
    });
  });

  // ---------------------------------------------------------------- overlay opacity
  document.querySelectorAll("input[data-opacity-for]").forEach((input) => {
    const section = document.getElementById(`method-${input.dataset.opacityFor}`);
    const apply = () => section.style.setProperty("--layer-opacity", input.value / 100);
    input.addEventListener("input", apply);
    apply();
  });

  // ---------------------------------------------------------------- full-size viewer
  const dialog = document.getElementById("lightbox");
  if (dialog && typeof dialog.showModal === "function") {
    const [img, layer] = dialog.querySelectorAll(".lightbox__stage img");
    const cap = dialog.querySelector(".lightbox__cap");
    document.addEventListener("click", (e) => {
      const link = e.target.closest("a[data-lightbox]");
      if (!link) return;
      e.preventDefault();
      img.src = link.href;
      img.alt = link.dataset.caption;
      if (link.dataset.layer) {
        layer.src = link.dataset.layer;
        const section = link.closest(".rx-method");
        layer.style.opacity = section ? getComputedStyle(section).getPropertyValue("--layer-opacity") || 0.6 : 0.6;
        layer.hidden = false;
      } else {
        layer.hidden = true;
        layer.removeAttribute("src");
      }
      cap.textContent = link.dataset.caption;
      dialog.showModal();
    });
    dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  } // without <dialog> support the link simply opens the image

  // ---------------------------------------------------------------- broken images
  document.querySelectorAll("img[data-fallback]").forEach((im) => {
    const fail = () => {
      const frame = im.closest(".frame__img");
      if (!frame) return;
      frame.querySelectorAll("img, .frame__expand").forEach((el) => (el.hidden = true));
      const empty = frame.querySelector(".frame__empty");
      if (empty) { empty.hidden = false; empty.querySelector("span").textContent = "Image file is missing"; }
    };
    if (im.complete && im.naturalWidth === 0 && im.getAttribute("src")) fail();
    im.addEventListener("error", fail);
  });
})();
