"""Deploy the dashboard to a Hugging Face Space (Docker SDK) in one command.

    pip install "huggingface_hub>=0.34"
    hf auth login                                   # paste a token with "write" access
    python scripts/deploy_hf_space.py YOUR_HF_USERNAME/sar-flood-intelligence

What it does
------------
1. Copies the files git tracks (no outputs/, data/, notebooks/ or secrets) into a staging folder.
2. Writes a Space README with the settings Hugging Face needs (sdk: docker, app_port: 7860).
3. Creates the Space if it does not exist, uploads the folder (large files are handled
   automatically, no git-lfs needed) and sets a random SECRET_KEY as a Space secret.

Options
-------
--run-dir outputs/runs/<run>   also deploy a trained U-Net that is not committed. Usually you
                               commit it instead: `sar-flood publish-model outputs/runs/<run>`
                               puts it in models/unet/, and every deploy then includes it.
--lite                         build without TensorFlow (smaller; no legacy classifier/U-Net)
--private                      create the Space as private
--dry-run                      only build the staging folder and print what would be uploaded
"""

from __future__ import annotations

import argparse
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# samples/legacy/ holds images of unknown origin (one is a figure from a published paper),
# so they are not served on a public site.
SKIP_PREFIXES = ("notebooks/", ".github/", "tests/", "data/", "outputs/", "samples/legacy/")
GITHUB = "https://github.com/ishajainbtech24-oss/Spatio-temporal-sar-analysis"

SPACE_README = """---
title: SAR Flood Intelligence
emoji: 🌊
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Flood maps from radar satellite images
---

# SAR Flood Intelligence

See which areas are under water, using radar satellite images that work through cloud,
day and night. Pick a practice image or upload a Sentinel-1 GeoTIFF.

A research project, not for emergency use. Source code and documentation:
{github}
"""


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"error: could not list git files ({exc}). Run this from a clone of the repository.")
    return [f for f in out.stdout.splitlines() if f and not f.startswith(SKIP_PREFIXES)]


def check_models(dest: Path, lite: bool) -> None:
    """Refuse to publish U-Nets trained on synthetic practice data; report what will be served."""
    import json  # noqa: PLC0415

    runs = sorted((dest / "models" / "unet").glob("*/metadata.json"))
    for meta_path in runs:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("data_source") == "synthetic":
            sys.exit(f"error: {meta_path.parent.name} was trained on synthetic practice data. "
                     "Remove it from models/unet/ before deploying.")
    if runs and lite:
        sys.exit("error: models/unet/ contains a trained model but --lite has no TensorFlow to run it")
    names = ", ".join(p.parent.name for p in runs) or "none (threshold methods + original classifier)"
    print(f"flood model to serve: {names}")


def stage(dest: Path, lite: bool, run_dir: Path | None) -> list[str]:
    files = tracked_files()
    for rel in files:
        src = ROOT / rel
        if src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
    (dest / "README.md").write_text(SPACE_README.format(github=GITHUB), encoding="utf-8")
    # the package metadata points at README.md, so the Space README also satisfies the build

    dockerfile = (dest / "Dockerfile").read_text(encoding="utf-8")
    if lite:
        dockerfile = dockerfile.replace("ARG WITH_TF=1", "ARG WITH_TF=0", 1)
    if run_dir is not None:
        needed = ["model.keras", "metadata.json", "config.yaml"]
        missing = [n for n in needed if not (run_dir / n).exists()]
        if missing:
            sys.exit(f"error: {run_dir} is missing {', '.join(missing)}")
        if lite:
            sys.exit("error: --run-dir needs TensorFlow; drop --lite")
        target = dest / "models" / "unet" / run_dir.name
        target.mkdir(parents=True)
        for n in needed:
            shutil.copy2(run_dir / n, target / n)
        dockerfile = dockerfile.replace(
            "EXPOSE 7860", f"ENV SAR_FLOOD_RUN_DIR=/app/models/unet/{run_dir.name}\nEXPOSE 7860", 1)
    (dest / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    check_models(dest, lite)
    return sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space", help="Space id, e.g. your-username/sar-flood-intelligence")
    ap.add_argument("--run-dir", type=Path, help="trained run to include (outputs/runs/<name>)")
    ap.add_argument("--lite", action="store_true", help="build without TensorFlow")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.space.count("/") != 1:
        sys.exit("error: the Space id must look like your-username/space-name")

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        files = stage(dest, args.lite, args.run_dir.resolve() if args.run_dir else None)
        size = sum((dest / f).stat().st_size for f in files) / 1e6
        print(f"staged {len(files)} files ({size:.1f} MB) for {args.space}"
              f"{' [lite]' if args.lite else ''}{' + U-Net ' + args.run_dir.name if args.run_dir else ''}")
        if args.dry_run:
            for f in files:
                print("  ", f)
            print("dry run: nothing uploaded")
            return 0

        try:
            from huggingface_hub import HfApi
        except ImportError:
            sys.exit('error: run  pip install "huggingface_hub>=0.34"  first')
        api = HfApi()
        try:
            user = api.whoami()["name"]
        except Exception:
            sys.exit("error: not logged in. Run  hf auth login  and paste a token with write access.")
        print(f"logged in as {user}")
        api.create_repo(args.space, repo_type="space", space_sdk="docker",
                        private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(dest), repo_id=args.space, repo_type="space",
            commit_message="Deploy SAR Flood Intelligence",
            # remove files that no longer exist in the project, keep HF's own .gitattributes
            delete_patterns=["src/**", "samples/**", "models/**", "configs/**", "scripts/**", "*.py",
                             "*.toml", "*.txt", "*.cff", "*.md", "Dockerfile", ".dockerignore"],
        )
        api.add_space_secret(args.space, "SECRET_KEY", secrets.token_hex(32),
                             description="Flask session key (flash messages)")
    print(f"\nuploaded. The Space is now building (first build takes about 5-10 minutes):\n"
          f"  https://huggingface.co/spaces/{args.space}\n"
          f"Your public link, once it says 'Running':\n"
          f"  https://{args.space.replace('/', '-').replace('_', '-').lower()}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
