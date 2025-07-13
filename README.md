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
   git clone https://github.com/<your-username>/bert-document-classifier.git
   cd bert-document-classifier
   ```

2. Create and activate your virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:

   ```bash
   uvicorn api.app:app --reload
   ```

5. Visit `http://127.0.0.1:8000/docs` to access the interactive Swagger UI.

---

## Notes

- The model must be trained before serving. After training, save the `model.pt` and `label_encoder.joblib` into the `saved_model/` directory.
- Ensure you have `python-multipart`, `torch`, `transformers`, `fastapi`, and `uvicorn` installed.

---

## License

This project is licensed under the MIT License.


---

Let me know if you'd like to include:
- API request/response examples  
- Screenshots of Swagger UI or cURL usage  
- Deployment steps (Docker, cloud, etc.)

You’re productionizing your ML—exactly how ML engineers grow.

## The dataset for training in the directory named data.


