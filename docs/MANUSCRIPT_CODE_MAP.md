# Manuscript-to-code map

This file explains which scripts reproduce which part of the manuscript.

| Manuscript component | Relevant file(s) | Notes |
|---|---|---|
| Image preprocessing | `src/preprocessing.py` | Grayscale conversion, resize to 128x128, histogram equalisation, Gaussian filtering, stack to 3 channels. |
| CNN architecture | `src/model.py` | Conv2D(32), Conv2D(64), Conv2D(128), Dense(128), Dropout(0.25), Dense(64), Dropout(0.25), sigmoid output. |
| Model training | `src/train_model.py`, `src/run_pipeline.py` | Adam, learning rate 0.001, binary cross-entropy, batch size 28, 20 epochs by default. |
| MC Dropout inference | `src/uncertainty.py` | `model(X, training=True)` repeated T=50 times. |
| Entropy equations | `src/uncertainty.py` | AU_E = E[H(p_t)], TU_E = H(E[p_t]), EU_E = TU_E - AU_E. |
| Variance equations | `src/uncertainty.py` | EU_V = Var[p_t], TU_V = pbar(1-pbar), AU_V = TU_V - EU_V. |
| IQR metric | `src/uncertainty.py` | IQR of MC probabilities and IQR of logits. |
| Threshold / referral curves | `src/evaluation.py`, `src/figures.py` | Accuracy of retained cases and excluded-count annotations. |
| MC sample-size sensitivity | `src/sensitivity_mc_dropout.py` | Stability for T=20/50/100/200 using T=200 reference. |
| Dataset-block uncertainty figures | `src/figures.py` | Scatter plots by Drishti, ORIGA, RIM-ONE and LAG. |
| End-to-end reproduction | `src/run_pipeline.py`, `notebooks/glaucoma_uq_colab_quickstart.ipynb` | Runs the full pipeline and writes manuscript outputs. |


