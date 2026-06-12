"""End-to-end reproducible pipeline for the glaucoma UQ manuscript."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from evaluation import classification_metrics, quantile_thresholds, threshold_curve
from figures import plot_threshold_curve, plot_uncertainty_by_dataset
from model import build_glaucoma_cnn
from preprocessing import make_generators, preprocess_fundus_images
from sensitivity_mc_dropout import run_mc_sensitivity
from uncertainty import (
    compute_uncertainty_metrics,
    deterministic_probabilities,
    mc_dropout_probabilities,
    set_global_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Glaucoma MC-Dropout uncertainty pipeline")
    parser.add_argument("--data_dir", required=True, help="Path to Combi_datasets folder")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--model_path", default=None, help="Existing trained model; if omitted, train a new model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--mc_runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation_split", type=float, default=0.30)
    parser.add_argument("--skip_sensitivity", action="store_true", help="Skip MC sensitivity analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    img_size = (128, 128)

    train_gen, val_gen = make_generators(
        args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        shuffle_train=False,
        shuffle_validation=False,
    )

    X_train = preprocess_fundus_images(train_gen.filepaths, target_size=img_size)
    X_val = preprocess_fundus_images(val_gen.filepaths, target_size=img_size)
    y_val = val_gen.labels.astype(int)

    if args.model_path:
        model = load_model(args.model_path, compile=False)
    else:
        model = build_glaucoma_cnn(img_height=128, img_width=128)
        history = model.fit(
            X_train,
            train_gen.labels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(X_val, y_val),
            verbose=1,
        )
        pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
        model.save(output_dir / "trained_model.h5")

    # Deterministic and MC-Dropout predictions
    p_det = deterministic_probabilities(model, X_val, batch_size=32)
    P = mc_dropout_probabilities(model, X_val, T=args.mc_runs, batch_size=32)
    np.savez(
        output_dir / f"mc_probs_seed{args.seed}_T{args.mc_runs}.npz",
        P=P,
        y_true=y_val,
        paths=np.asarray(val_gen.filepaths),
        p_det_all=p_det,
    )

    metrics = compute_uncertainty_metrics(P)
    p_mc = metrics["p_mc_mean"]

    df = pd.DataFrame({
        "file": val_gen.filepaths,
        "y_true": y_val,
        "p_det_no_dropout": p_det,
        "p_mc_mean": p_mc,
        "pred_det": (p_det >= 0.5).astype(int),
        "pred_mc": (p_mc >= 0.5).astype(int),
        "AU_E_bits": metrics["AU_E_bits"],
        "TU_E_bits": metrics["TU_E_bits"],
        "EU_E_bits": metrics["EU_E_bits"],
        "AU_V": metrics["AU_V"],
        "TU_V": metrics["TU_V"],
        "EU_V": metrics["EU_V"],
        "IQR_prob": metrics["IQR_prob"],
        "IQR_logit": metrics["IQR_logit"],
        "SD_logit": metrics["SD_logit"],
        "skewness": metrics["skewness"],
    })
    df.to_csv(output_dir / "predictions_with_uncertainty.csv", index=False)

    base = classification_metrics(y_val, p_mc)
    pd.DataFrame([base]).to_csv(output_dir / "baseline_metrics.csv", index=False)

    # Threshold curves for the three main manuscript metrics
    curve_frames = []
    metric_info = {
        "entropy_EU_bits": (metrics["EU_E_bits"], "Epistemic entropy threshold (bits)", "Accuracy vs. entropy threshold"),
        "variance_EU": (metrics["EU_V"], "Epistemic variance threshold", "Accuracy vs. variance threshold"),
        "iqr_probability": (metrics["IQR_prob"], "IQR probability threshold", "Accuracy vs. IQR threshold"),
    }
    for name, (u, xlabel, title) in metric_info.items():
        thresholds = quantile_thresholds(u, n=10, q_low=0.10, q_high=1.00)
        curve = threshold_curve(y_val, p_mc, u, thresholds)
        curve.insert(0, "metric", name)
        curve_frames.append(curve)
        plot_threshold_curve(
            curve,
            xlabel=xlabel,
            title=title,
            save_path=output_dir / f"fig_accuracy_vs_{name}.png",
            baseline_accuracy=base["accuracy"],
        )

    threshold_summary = pd.concat(curve_frames, ignore_index=True)
    threshold_summary.to_csv(output_dir / "threshold_summary.csv", index=False)

    # Dataset block scatter figures
    plot_uncertainty_by_dataset(
        val_gen.filepaths,
        metrics["SD_logit"],
        ylabel="Uncertainty (standard deviation of logits)",
        title="Uncertainty of predictions across datasets",
        save_path=output_dir / "fig_dataset_uncertainty_logit_sd.png",
        approximate_counts={"Drishti": 330, "ORIGA": 220, "RIM-ONE": 130},
    )
    plot_uncertainty_by_dataset(
        val_gen.filepaths,
        metrics["EU_E_bits"] * 100.0,
        ylabel="Epistemic entropy (Eq. 4, % of binary maximum)",
        title="Epistemic uncertainty by dataset",
        save_path=output_dir / "fig_dataset_epistemic_entropy_pct.png",
        approximate_counts={"Drishti": 330, "ORIGA": 220, "RIM-ONE": 130},
    )

    if not args.skip_sensitivity:
        run_mc_sensitivity(
            model,
            X_val,
            output_dir=output_dir,
            mc_values=(20, 50, 100, 200),
            reference_T=200,
            batch_size=32,
            seed=args.seed,
        )

    print("Pipeline finished. Outputs written to:", output_dir)


if __name__ == "__main__":
    main()
