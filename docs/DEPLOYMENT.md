# Deployment

The dashboard ships as one Docker image (`Dockerfile`) that runs on any Docker host. The
recommended free option is a **Hugging Face Space**, whose free CPU hardware has enough memory
for TensorFlow.

| Build | Memory used (measured) | Features | Good for |
|---|---|---|---|
| Full (default) | ~640 MB after the first analysis | Everything, including the original scene classifier and a trained U-Net if you add one | Hugging Face Spaces, any host with ≥ 1 GB RAM |
| Lite (`WITH_TF=0`) | ~170 MB | Flood maps from the threshold methods, flood verdict, land cover, PDF report | Render / Railway free or 512 MB plans |

Production settings baked into the image: gunicorn (1 worker, 4 threads), port `7860`
(override with `PORT`), proxy headers trusted, analyses deleted after 24 hours
(`RESULT_TTL_HOURS`), health check at `/api/health`.

## Option A — Hugging Face Space (recommended)

1. Create a free account at <https://huggingface.co/join>.
2. Create an access token: Settings → Access Tokens → **Create new token** → type **Write**.
3. On your computer, in the project folder with your virtual environment active:
   ```bash
   pip install "huggingface_hub>=0.34"
   hf auth login                 # paste the token; answer "n" to "add as git credential"
   python scripts/deploy_hf_space.py YOUR_HF_USERNAME/sar-flood-intelligence
   ```
4. Open the link the script prints (`https://huggingface.co/spaces/...`). The first build takes
   about 5–10 minutes (it installs TensorFlow). When the status turns **Running**, the app is live at
   `https://YOUR_HF_USERNAME-sar-flood-intelligence.hf.space`.

To update the live site by hand, run the same `deploy_hf_space.py` command. Better: set up
automatic deploys (next section) once.

### Automatic deploys from GitHub

`.github/workflows/deploy.yml` uploads the site after every push to `main`, **only if the
tests workflow passed for that commit**. A broken commit never reaches the website.

One-time setup, on GitHub: repository **Settings → Secrets and variables → Actions**:
1. **Secrets** tab → **New repository secret**: name `HF_TOKEN`, value = your Hugging Face
   **Write** token.
2. **Variables** tab → **New repository variable**: name `HF_SPACE`, value =
   `YOUR_HF_USERNAME/sar-flood-intelligence`.

The workflow file must be on `main` (merge your branch first). After that, every push to
`main` goes: tests (about 5–10 min) → deploy (about 1 min upload) → the Space rebuilds (a few
minutes). Follow it in the **Actions** tab. To deploy without a new commit: Actions →
**deploy** → **Run workflow**. Pushes to other branches and pull requests never deploy.

Notes:
- The script uploads only files tracked by git, minus `notebooks/`, `tests/`, `data/`,
  `outputs/` and `samples/legacy/` (images of unknown origin are not published).
- Free Spaces go to sleep after a period without visitors and wake up on the next visit
  (the first page then takes a little longer).
- Uploaded files and results live only inside the running container; they disappear on restart.

### Which models are deployed

| Model | Deployed? |
|---|---|
| `models/legacy/simple_transformer_pretrained_final.keras` (notebook 07) | Yes, shown as the scene-level check on every result |
| `models/legacy/encoder.keras`, `simple_transformer_final.keras` | Uploaded but not used by the website |
| Other Colab models (Random Forest, XGBoost, dual-pol/fold checkpoints) | No; they only exist on Google Drive |
| U-Net flood model | Once you train and publish it (below) |

### Adding the trained U-Net

After training on Sen1Floods11 (`sar-flood train ...`):
```bash
sar-flood publish-model outputs/runs/unet_sen1floods11   # copies model.keras, metadata.json, config.yaml to models/unet/
git add models/unet && git commit -m "Publish trained model" && git push
```
With automatic deploys, that push puts the model on the website. `publish-model` and the
deploy script both refuse models trained on synthetic practice data. Only one model folder
should live in `models/unet/`; the newest is used. A depth-4 U-Net file is roughly 10–25 MB,
well under GitHub's 100 MB limit.

## Option B — Render (lite build)

1. Push the repository to GitHub (already done).
2. <https://dashboard.render.com> → **New** → **Web Service** → connect the GitHub repository.
3. Language: **Docker**. Branch: the one you deploy (for example `main`).
4. **Environment variables**: `SECRET_KEY` = any long random text.
5. **Advanced → Docker build arguments**, if offered: `WITH_TF=0`. Otherwise change
   `ARG WITH_TF=1` to `ARG WITH_TF=0` in the Dockerfile on the deployed branch.
6. Health check path: `/api/health`. Create the service.

Render sets `PORT` itself; the image reads it. The free plan sleeps when idle.

## Run the image yourself

```bash
docker build -t sar-flood .                 # or: --build-arg WITH_TF=0
docker run -p 7860:7860 -e SECRET_KEY=change-me sar-flood
# open http://localhost:7860
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Space stuck on **Building** for > 20 min or shows **Build error** | Open the **Logs** tab on the Space; a failed `pip install` line names the package. Re-run the deploy command after fixing. |
| **Runtime error**, "port" in the logs | The Space README must say `app_port: 7860`; the deploy script writes this. |
| `not logged in` from the script | Run `hf auth login` again with a **Write** token. |
| Render: "out of memory" | Use the lite build (`WITH_TF=0`). |
| Analyses vanished | Expected after a restart or after 24 hours; set `RESULT_TTL_HOURS=0` to keep them while the container runs. |
