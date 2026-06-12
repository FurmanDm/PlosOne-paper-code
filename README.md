# Glaucoma uncertainty quantification code package

This repository contains the cleaned, manuscript-relevant code for the study:

**Personalised automated uncertainty quantification and thresholding for glaucoma case detection from colour fundus images**

The code was cleaned from the exploratory Colab/Python snippets used during development. Only the parts needed to reproduce the manuscript pipeline are included:

1. fundus image preprocessing,
2. CNN model construction and training,
3. Monte-Carlo Dropout inference,
4. uncertainty metrics: entropy, variance, IQR and logit dispersion,
5. uncertainty-informed referral / threshold curves,
6. sensitivity analysis for the number of MC samples,
7. manuscript figures and tables.

Exploratory and non-essential blocks were removed from the clean package, including old debugging plots, repeated imports, temporary notebook cells, Graphviz drafts, calibration experiments, and unrelated interval/ROC experiments that were not central to the manuscript.

## Repository structure

```text
glaucoma_uq_plosone_code/
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── config_example.yaml
├── docs/
│   ├── CODE_AVAILABILITY_STATEMENT.txt
│   ├── MANUSCRIPT_CODE_MAP.md
│   └── RESPONSE_TO_JOURNAL_REQUIREMENT_2.txt
├── data/
│   └── README.md
├── outputs/
│   └── README.md
├── notebooks/
│   └── glaucoma_uq_colab_quickstart.ipynb
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── model.py
    ├── train_model.py
    ├── uncertainty.py
    ├── evaluation.py
    ├── figures.py
    ├── sensitivity_mc_dropout.py
    └── run_pipeline.py
```

## Expected dataset structure

The scripts expect a binary directory structure compatible with `tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory`:

```text
Combi_datasets/
├── healthy/
│   ├── image_001.jpg
│   └── ...
└── glaucoma/
    ├── image_002.jpg
    └── ...
```

The original manuscript used public colour fundus datasets combined into this two-class structure. The datasets are not redistributed here; please obtain them from their original sources and organise them as shown above.

## Quick start in Google Colab

1. Upload or copy this repository folder to Google Drive.
2. Place the combined dataset in your Drive, for example:

```text
/content/drive/My Drive/Combi_datasets/
```

3. Open `notebooks/glaucoma_uq_colab_quickstart.ipynb`.
4. Adjust the paths in the configuration cell if needed.
5. Run the pipeline cells in order.

## Quick start from Python

Edit `config_example.yaml` or pass command-line arguments directly:

```bash
python src/run_pipeline.py \
  --data_dir "/content/drive/My Drive/Combi_datasets/" \
  --output_dir "/content/drive/My Drive/glaucoma_uq_outputs" \
  --epochs 20 \
  --mc_runs 50
```

## Main outputs

The pipeline writes the following files to the output directory:

```text
trained_model.h5
history.csv
mc_probs_seed1337_T50.npz
predictions_with_uncertainty.csv
threshold_summary.csv
mc_sensitivity_summary.csv
fig_accuracy_vs_entropy.png
fig_accuracy_vs_variance.png
fig_accuracy_vs_iqr.png
fig_dataset_uncertainty.png
```

## Notes on reproducibility

The main random seed is set to `1337`. TensorFlow operations may still show minor variation depending on GPU, TensorFlow version, CUDA/cuDNN version and deterministic operation support.

