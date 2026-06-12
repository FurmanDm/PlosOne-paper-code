# Data folder

No image data are redistributed in this code package.

Expected structure:

```text
Combi_datasets/
├── healthy/
└── glaucoma/
```

The directory is read by Keras `flow_from_directory`, with a validation split of 0.30 and `shuffle=False` for reproducible alignment during evaluation.
