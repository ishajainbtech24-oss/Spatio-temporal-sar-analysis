// Geographic view: basemap + the selected method's water layer at the GeoTIFF's bounds.
(function () {
  const el = document.getElementById("map");
  if (!el) return;
  if (typeof L === "undefined") {
    el.innerHTML = '<p class="map__offline">The map library could not be loaded (offline?). The imagery panels above show the same result.</p>';
    return;
  }
  const b = [[+el.dataset.south, +el.dataset.west], [+el.dataset.north, +el.dataset.east]];
  const layers = JSON.parse(el.dataset.layers || "{}");
  const map = L.map(el, { scrollWheelZoom: false });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  L.rectangle(b, { color: "#8ec5ff", weight: 1, fill: false, dashArray: "4 3" }).addTo(map);
  let overlay = null;
  function show(key) {
    if (!layers[key]) return;
    if (overlay) map.removeLayer(overlay);
    overlay = L.imageOverlay(layers[key], b, { opacity: 0.85 }).addTo(map);
  }
  show(el.dataset.active);
  map.fitBounds(b, { padding: [24, 24] });
  document.addEventListener("method:change", (e) => show(e.detail));
})();
