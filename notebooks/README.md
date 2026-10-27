# Notebooks (historical)

These are the original Colab notebooks, kept unchanged as a record of how the project developed.
The maintained pipeline is the `sar_flood` package (`src/sar_flood`) and the `sar-flood` CLI.
Read `docs/PROJECT_AUDIT.md` before quoting any number from here.

| Notebook | What it does | Known issues |
|---|---|---|
| 01_data_exploration | Dataset structure, counts per event, class/pixel stats | Hard-coded Colab paths; an early cell references `DATASET_PATH` before defining it. Useful statistics. |
| 02_preprocessing | Pairing, median filter, min-max, resize to 128, chip labels, random split | **Labels inverted** by `LabelEncoder` (Flood→0, NoFlood→1). One NaN pixel turns a whole chip into NaN. Only VV used. Random (not by-event) split. |
| 03_autoencoder | Conv autoencoder, latent 32×32×16, PCA plot | Uses the test set as validation data. |
| 04_training_classifier | Random Forest on flattened latents | Evaluates on training features (`X_test` = train latents); relies on undefined variables (hidden state). 0.875 = majority-class rate. |
| 05_prediction_pipeline | Single-image inference with the deployed model | Preprocessing differs from training (`cv2.resize` + `/255`). |
| 06_Model_comparision | RF vs XGBoost vs CNN | CNN trained on NaN inputs (`loss: nan`); test set used as validation; accuracies equal majority rate. |
| 07_Transformer | MobileNetV2 + transformer, group-aware 5-fold CV, ablation, final model | Best methodology in the project, but on inverted labels; final model trained on all data. Out-of-fold AUC 0.757 (fine-tuned) / 0.723 (frozen). |

To re-run them you need the dataset on Google Drive at the paths in each notebook.
