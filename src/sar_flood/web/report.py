"""PDF analysis report built from the same view model as the results page."""

from __future__ import annotations

from pathlib import Path

from .results import fmt_area, fmt_int

INK = "#141C28"
MUTED = "#5A6676"
CORAL = "#E8604C"
TEAL = "#1E9C8A"

DISCLAIMER = (
    "Research prototype. Water is detected from Sentinel-1 backscatter; labels and outputs "
    "include permanent water bodies (rivers, lakes), so 'flooded' here means water-covered at "
    "acquisition time. Smooth surfaces and radar shadow can be mistaken for water, and flooded "
    "vegetation or streets can be missed. Percentages are shares of valid analysed pixels, not "
    "model confidence. Not validated for emergency response or any decision affecting people "
    "or property."
)


def _t(s) -> str:
    """Make text safe for the built-in PDF fonts (no U+2212 / U+2248 glyphs)."""
    return (str(s).replace("−", "-").replace("≈", "~").replace("→", "->")
            .replace("km²", "sq km").replace("m²", "sq m"))


def build_report(view: dict, run_dir: Path, out_path: Path) -> Path:
    from reportlab.lib import colors  # noqa: PLC0415
    from reportlab.lib.pagesizes import A4  # noqa: PLC0415
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: PLC0415
    from reportlab.lib.units import mm  # noqa: PLC0415
    from reportlab.platypus import (  # noqa: PLC0415
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    r = view["r"]
    methods = {m["key"]: m for m in view["methods"]}
    primary = methods.get(view["primary"])
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, leading=21, alignment=0,
                        textColor=colors.HexColor(INK), spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor(INK),
                        spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.8, leading=12,
                          textColor=colors.HexColor(INK))
    small = ParagraphStyle("s", parent=body, fontSize=7.8, leading=10.5, textColor=colors.HexColor(MUTED))

    def table(rows, widths, header=True):
        t = Table([[Paragraph(_t(c), body) for c in row] for row in rows], colWidths=widths)
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DBE3")),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F6"))]
        t.setStyle(TableStyle(style))
        return t

    story = [Paragraph("SAR Flood Intelligence", small),
             Paragraph("Flood Detection Analysis", h1),
             Paragraph(_t(f"Analysis {view['rid']}  |  {r.get('created', '-')}  |  {view['status']['label']}"), small),
             Spacer(1, 6)]

    story.append(table([
        ["Input", r.get("input", "-"), "Format", r.get("input_format", "-")],
        ["Size", f"{r.get('width')} x {r.get('height')} px, {r.get('bands')} band(s) "
                 f"({', '.join(r.get('band_names', []))})", "CRS", r.get("crs") or "not georeferenced"],
        ["No-data pixels", f"{100 * (r.get('nodata_fraction') or 0):.1f} %", "Calibrated dB",
         "yes" if r.get("calibrated_db") else "no"],
    ], [28 * mm, 62 * mm, 26 * mm, 58 * mm], header=False))

    if primary and primary.get("has_counts") and primary.get("water_pct") is not None:
        c = primary["counts"]
        story += [Paragraph(_t(f"Prediction summary - {primary['title']}"), h2),
                  Paragraph(_t(
                      f"<b><font color='{CORAL}'>{primary['water_pct']:.1f} %</font></b> of "
                      f"{fmt_int(c['valid'])} valid pixels classified as water "
                      f"({fmt_int(c['water'])} px); <b><font color='{TEAL}'>{primary['dry_pct']:.1f} %</font></b> "
                      f"not water ({fmt_int(c['non_water'])} px). {fmt_int(c['nodata'])} no-data pixels excluded."
                      + (f" Estimated water area {fmt_area(primary['area']['water'])} of "
                         f"{fmt_area(primary['area']['valid'])} analysed ({primary['area']['method']})."
                         if primary.get("area") else "")), body)]

        imgs, caps = [], []
        for name, cap in [(r["images"].get("vv"), "SAR input (VV)"), (primary.get("mask"), "Predicted mask"),
                          (primary.get("overlay"), "Overlay"), (r["images"].get("ground_truth"), "Ground truth")]:
            if name and (run_dir / name).exists():
                imgs.append(Image(str(run_dir / name), width=40 * mm, height=40 * mm))
                caps.append(Paragraph(cap, small))
        if imgs:
            grid = Table([imgs, caps], colWidths=[43 * mm] * len(imgs))
            grid.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story += [Spacer(1, 6), grid,
                      Paragraph("Colours: coral = water, teal = not water, slate = no data.", small)]

    if view["methods"]:
        has_gt = any(m.get("metrics") for m in view["methods"])
        head = ["Method", "Water %", "Water px", "Valid px", "Water area"] + (["IoU", "Dice", "Precision", "Recall"] if has_gt else [])
        rows = [head]
        for m in view["methods"]:
            c = m.get("counts") or {}
            row = [m["title"], f"{m['water_pct']:.2f}" if m.get("water_pct") is not None else "-",
                   fmt_int(c.get("water")), fmt_int(c.get("valid")),
                   fmt_area((m.get("area") or {}).get("water"))]
            if has_gt:
                mt = m.get("metrics") or {}
                row += [f"{mt[k]:.3f}" if mt.get(k) is not None else "-"
                        for k in ("iou", "dice_f1", "precision", "recall")]
            rows.append(row)
        widths = ([40 * mm, 14 * mm, 18 * mm, 18 * mm, 20 * mm] + [16 * mm] * 4) if has_gt else \
            [62 * mm, 24 * mm, 28 * mm, 28 * mm, 32 * mm]
        story += [Paragraph("All methods", h2), table(rows, widths)]
        if has_gt:
            story.append(Paragraph("Metrics compare each mask with the supplied ground truth over pixels "
                                   "that are valid in both.", small))

    if primary:
        model = primary.get("model") or {}
        rows = [["Method", primary["title"]], ["Output", primary.get("output_type", "-")]]
        for k, label in [("architecture", "Model"), ("checkpoint", "Checkpoint"),
                         ("encoder_init", "Encoder init"), ("created", "Trained")]:
            if model.get(k):
                rows.append([label, model[k]])
        if primary.get("trained_on"):
            rows.append(["Training data", primary["trained_on"]])
        if primary.get("threshold") is not None:
            rows.append(["Decision threshold", f"{primary['threshold']:.2f} (water probability)"])
        if primary.get("threshold_db") is not None:
            rows.append(["Threshold", f"{primary['threshold_db']} dB (VV)"])
        rows.append(["Inference time", f"{primary.get('inference_ms', '-')} ms"])
        story += [Paragraph("Model and processing", h2), table(rows, [40 * mm, 134 * mm], header=False),
                  Paragraph(_t("Preprocessing: " + "; ".join(r.get("preprocessing", []))), small)]

    if r.get("legacy_classifier"):
        lc = r["legacy_classifier"]
        story += [Paragraph("Legacy scene classifier", h2),
                  Paragraph(_t(f"{lc['label']} (probability the chip contains no water: "
                               f"{100 * lc['p_no_water']:.1f} %, threshold {lc['threshold_no_water']}). "
                               "Chip-level model from the original notebooks; no held-out evaluation."), body)]
    if r.get("warnings"):
        story += [Paragraph("Notes", h2)] + [Paragraph(_t("- " + w), body) for w in r["warnings"]]
    story += [Paragraph("Limitations", h2), Paragraph(_t(DISCLAIMER), small)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Flood Detection Analysis {view['rid']}", author="SAR Flood Intelligence")
    doc.build(story)
    return out_path
