# BERT Document Classifier

A document classification system powered by BERT (Bidirectional Encoder Representations from Transformers) and served through a FastAPI backend. This tool classifies uploaded PDF documents into relevant categories using state-of-the-art NLP.

---

## Features

- BERT-based transformer for deep document understanding  
- End-to-end classification pipeline for PDF files  
- FastAPI backend for real-time inference  
- Accepts uploaded documents through a REST API  
- Modular structure for easy extension and maintenance  

---

## How It Works

1. A PDF file is uploaded via the FastAPI endpoint.
2. The system extracts text from the PDF.
3. The text is tokenized and passed through a fine-tuned BERT model.
4. The model outputs a predicted label based on learned document classes.

---

## Project Structure

```
bert-document-classifier/
├── model/
│   ├── bert_classifier.py        # BERT model definition
│   ├── inference.py              # PDF-to-prediction pipeline
├── api/
│   ├── app.py                    # FastAPI app for serving the model
├── saved_model/
│   ├── model.pt                  # Trained model weights
│   ├── label_encoder.joblib      # Fitted label encoder
├── experiment_01_document_classifier.ipynb  # Initial training and experimentation
├── README.md
└── .gitignore
```

---

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/ruskaruma/bert-document-classifier.git
   cd bert-document-classifier
   ```

2. Download the dataset from Kaggle (Make Data Count competition) and place in `data/` directory

3. Create and activate your virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. Install the dependencies:

   For running the API only:
   ```bash
   pip install -r requirements.txt
   ```

   For training the model:
   ```bash
   pip install -r requirements-train.txt
   ```

5. Verify setup:

   ```bash
   python check_setup.py
   ```

6. Train the model (if not already trained):

   ```bash
   jupyter notebook experiment_01_document_classifier.ipynb
   ```

7. Run the FastAPI server:

   ```bash
   uvicorn api.app:app --reload
   ```

8. Visit `http://127.0.0.1:8000/docs` to access the interactive Swagger UI.

---

## Notes

- The model must be trained before serving. After training, save the `model.pt` and `label_encoder.joblib` into the `saved_model/` directory.
- Ensure you have `python-multipart`, `torch`, `transformers`, `fastapi`, and `uvicorn` installed.

---

## License

This project is licensed under the MIT License.

---
## The dataset for training in the directory named data.

The dataset was picked from the Make Data Count - Finding Data References competition on Kaggle. 


