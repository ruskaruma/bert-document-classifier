# Saved Model Directory

This directory should contain the trained BERT model files:

- `model.pt` - The trained PyTorch model weights
- `label_encoder.joblib` - The fitted label encoder for mapping predictions to class names

## How to Get the Model Files

### Option 1: Train the Model
Run the Jupyter notebook `train_bert_classifier.ipynb` to train the model. The training process will:
1. Load and preprocess the training data
2. Train the BERT classifier for 3 epochs
3. Save the trained model to this directory

### Option 2: Download Pre-trained Model
If model files are hosted elsewhere (e.g., Google Drive, Hugging Face), download them and place them in this directory.

## Expected Files

```
saved_model/
├── model.pt              # ~440 MB (BERT model weights)
├── label_encoder.joblib  # ~1 KB (label encoder)
└── README.md            # This file
```

## Notes

- Model files are gitignored to keep the repository size manageable
- After training or downloading, ensure both files exist before running the API
- The model classifies documents into: Primary, Secondary, or Missing reference types

