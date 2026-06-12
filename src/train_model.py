"""Train the CNN model and save baseline prediction files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from model import build_glaucoma_cnn
from preprocessing import make_generators, preprocess_fundus_images
from uncertainty import mc_dropout_probabilities, compute_uncertainty_metrics, set_global_seed


def train_and_save(
    data_dir: str,
    output_dir: str,
    img_size=(128, 128),
    batch_size: int = 28,
    epochs: int = 20,
    validation_split: float = 0.30,
    mc_runs: int = 50,
    seed: int = 1337,
):
    """Train the model, run MC Dropout and save key outputs."""
    set_global_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen = make_generators(
        data_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle_train=False,
        shuffle_validation=False,
    )

    model = build_glaucoma_cnn(img_height=img_size[0], img_width=img_size[1])
    history = model.fit(
        preprocess_fundus_images(train_gen.filepaths, target_size=img_size),
        train_gen.labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(preprocess_fundus_images(val_gen.filepaths, target_size=img_size), val_gen.labels),
        verbose=1,
    )

    pd.DataFrame(history.history).to_csv(output / "history.csv", index=False)
    model.save(output / "trained_model.h5")

    X_train = preprocess_fundus_images(train_gen.filepaths, target_size=img_size)
    X_val = preprocess_fundus_images(val_gen.filepaths, target_size=img_size)

    P_train = mc_dropout_probabilities(model, X_train, T=mc_runs, batch_size=32)
    P_val = mc_dropout_probabilities(model, X_val, T=mc_runs, batch_size=32)

    m_train = compute_uncertainty_metrics(P_train)
    m_val = compute_uncertainty_metrics(P_val)

    train_df = pd.DataFrame({
        "file": train_gen.filepaths,
        "label": train_gen.labels,
        "prediction": m_train["p_mc_mean"],
        "uncertainty_variance": m_train["EU_V"],
        "iqr": m_train["IQR_prob"],
        "entropy_total_bits": m_train["TU_E_bits"],
        "entropy_aleatoric_bits": m_train["AU_E_bits"],
        "entropy_epistemic_bits": m_train["EU_E_bits"],
    })

    val_df = pd.DataFrame({
        "file": val_gen.filepaths,
        "label": val_gen.labels,
        "prediction": m_val["p_mc_mean"],
        "uncertainty_variance": m_val["EU_V"],
        "iqr": m_val["IQR_prob"],
        "entropy_total_bits": m_val["TU_E_bits"],
        "entropy_aleatoric_bits": m_val["AU_E_bits"],
        "entropy_epistemic_bits": m_val["EU_E_bits"],
    })

    train_df.to_csv(output / "train_predictions_with_uncertainty.csv", index=False)
    val_df.to_csv(output / "validation_predictions_with_uncertainty.csv", index=False)

    train_f1 = f1_score(train_gen.labels, train_df["prediction"].round())
    val_f1 = f1_score(val_gen.labels, val_df["prediction"].round())
    print(f"Training F1: {train_f1:.4f}")
    print(f"Validation F1: {val_f1:.4f}")

    return model, train_gen, val_gen, train_df, val_df
