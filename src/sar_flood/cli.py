"""Command-line interface: ``sar-flood <command> --help``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _cfg(args, **overrides):
    from .config import load_config  # noqa: PLC0415

    ov: dict = {}
    if getattr(args, "data_root", None):
        ov.setdefault("data", {})["root"] = args.data_root
    for key, value in overrides.items():
        if value is not None:
            section, name = key.split("__")
            ov.setdefault(section, {})[name] = value
    return load_config(args.config, ov)


def cmd_validate(args) -> int:
    from .data.sen1floods11 import validate_dataset  # noqa: PLC0415
    from .utils import save_json  # noqa: PLC0415

    cfg = _cfg(args)
    report = validate_dataset(cfg.data_root, cfg.data.bands)
    out = Path(args.out) if args.out else cfg.resolve(cfg.output_dir) / "data_validation.json"
    save_json(report, out)
    print(f"pairs: {report['n_pairs']}  ok: {report['n_ok']}  with problems: {report['n_with_problems']}")
    print(f"chips containing no-data pixels: {report['n_with_any_nodata']}  "
          f"chips without any water: {report['n_without_water']}")
    print(f"events: {report['events']}")
    print(f"official split files: {report['official_split_dir'] or 'not found (group split will be used)'}")
    print(f"report written to {out}")
    return 0 if report["n_ok"] > 0 else 1


def cmd_make_synthetic(args) -> int:
    from .data.synthetic import write_dataset  # noqa: PLC0415

    root = write_dataset(args.out, n_per_event=args.n_per_event, size=args.size, seed=args.seed)
    print(f"synthetic dataset written to {root} (NOT real Sentinel-1 data)")
    return 0


def cmd_train(args) -> int:
    from .data.sen1floods11 import detect_data_source  # noqa: PLC0415
    from .training.trainer import train  # noqa: PLC0415

    cfg = _cfg(args, train__epochs=args.epochs, train__ae_epochs=args.ae_epochs,
               train__max_train_samples=args.max_train_samples, train__device=args.device)
    if args.run_name:
        cfg.run_name = args.run_name
    run_dir = train(cfg, data_source=detect_data_source(cfg.data_root))
    print(f"run directory: {run_dir}")
    return 0


def cmd_evaluate(args) -> int:
    from .data.sen1floods11 import detect_data_source  # noqa: PLC0415
    from .evaluation.evaluate import evaluate  # noqa: PLC0415

    cfg = _cfg(args)
    out = evaluate(cfg, split=args.split, run_dir=args.run_dir, out_dir=args.out,
                   include_legacy=args.legacy, data_source=detect_data_source(cfg.data_root))
    print(f"results written to {out}")
    return 0


def cmd_predict(args) -> int:
    from .inference.pipeline import analyze_file  # noqa: PLC0415

    segmenter = None
    if args.run_dir:
        from .inference.segment import Segmenter  # noqa: PLC0415

        segmenter = Segmenter.from_run_dir(args.run_dir)
    res = analyze_file(args.image, args.out, label_path=args.label, segmenter=segmenter,
                       use_legacy=not args.no_legacy)
    for name, m in res["methods"].items():
        line = f"{name:6s} water fraction {m['water_fraction']:.3f}"
        if "metrics" in m:
            line += f"  IoU {m['metrics']['iou']:.3f}"
        if m.get("verdict"):
            line += f"  -> {m['verdict']['label']}"
        sc = (m.get("land_cover") or {}).get("screening")
        if sc:
            line += f"  | plantation screening: {sc['label']}"
        print(line)
    if "legacy_classifier" in res:
        print("legacy:", res["legacy_classifier"])
    for w in res["warnings"]:
        print("warning:", w)
    print(f"outputs written to {args.out}")
    return 0


def cmd_publish_model(args) -> int:
    """Copy the files needed for inference into models/unet/<run>/ so git (and deploys) carry them."""
    import shutil  # noqa: PLC0415

    from .config import PROJECT_ROOT  # noqa: PLC0415
    from .utils import load_json  # noqa: PLC0415

    run = Path(args.run_dir)
    needed = ["model.keras", "metadata.json", "config.yaml"]
    missing = [n for n in needed if not (run / n).exists()]
    if missing:
        raise FileNotFoundError(f"{run} is missing {', '.join(missing)}; train it with `sar-flood train` first")
    meta = load_json(run / "metadata.json")
    if meta.get("data_source") == "synthetic" and not args.allow_synthetic:
        raise ValueError("this model was trained on synthetic practice data; publishing it would "
                         "show meaningless results on the website (use --allow-synthetic to override)")
    dest = PROJECT_ROOT / "models" / "unet" / run.name
    dest.mkdir(parents=True, exist_ok=True)
    for n in needed:
        shutil.copy2(run / n, dest / n)
    size = sum((dest / n).stat().st_size for n in needed) / 1e6
    print(f"published {run.name} to {dest.relative_to(PROJECT_ROOT)} ({size:.1f} MB)")
    print("next: git add models/unet && git commit -m \"Publish trained model\" && git push")
    return 0


def cmd_serve(args) -> int:
    from .web.app import create_app  # noqa: PLC0415

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sar-flood", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", default="configs/default.yaml", help="YAML config file")
        sp.add_argument("--data-root", help="override data.root (or set SAR_DATA_ROOT)")

    sp = sub.add_parser("validate-data", help="check dataset files, values and georeferencing")
    common(sp)
    sp.add_argument("--out", help="where to write the JSON report")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("make-synthetic", help="write a small synthetic dataset (for smoke tests)")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--n-per-event", type=int, default=4)
    sp.add_argument("--size", type=int, default=128)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_make_synthetic)

    sp = sub.add_parser("train", help="autoencoder pre-training + U-Net training")
    common(sp)
    sp.add_argument("--run-name")
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--ae-epochs", type=int)
    sp.add_argument("--max-train-samples", type=int)
    sp.add_argument("--device", choices=["auto", "cpu", "gpu"])
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="evaluate U-Net and baselines on a split")
    common(sp)
    sp.add_argument("--run-dir", help="trained run (omit to evaluate baselines only)")
    sp.add_argument("--split", default="test", choices=["train", "val", "test"])
    sp.add_argument("--legacy", action="store_true", help="also score the legacy scene classifier")
    sp.add_argument("--out", help="output directory")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("predict", help="analyse one GeoTIFF/PNG and write masks + quick-looks")
    sp.add_argument("image")
    sp.add_argument("--label", help="optional LabelHand GeoTIFF for per-image metrics")
    sp.add_argument("--run-dir", help="trained run directory")
    sp.add_argument("--out", default="outputs/predictions")
    sp.add_argument("--no-legacy", action="store_true")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("publish-model", help="add a trained run to models/unet/ so it is committed and deployed")
    sp.add_argument("run_dir", help="e.g. outputs/runs/unet_sen1floods11")
    sp.add_argument("--allow-synthetic", action="store_true")
    sp.set_defaults(func=cmd_publish_model)

    sp = sub.add_parser("serve", help="start the dashboard")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=5000)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
