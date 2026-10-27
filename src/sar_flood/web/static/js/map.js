// Places the prediction overlay at the GeoTIFF's lat/lon bounds on an OSM basemap.
(function () {
  const el = document.getElementById("map");
  if (!el || typeof L === "undefined") return;
  const b = [[+el.dataset.south, +el.dataset.west], [+el.dataset.north, +el.dataset.east]];
  const map = L.map(el, { scrollWheelZoom: false });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  L.imageOverlay(el.dataset.overlay, b, { opacity: 0.8 }).addTo(map);
  L.rectangle(b, { color: "#1f6fb2", weight: 1, fill: false }).addTo(map);
  map.fitBounds(b, { padding: [20, 20] });
})();
