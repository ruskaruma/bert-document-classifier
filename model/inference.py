import torch
from transformers import BertTokenizer
import joblib
import os
from pathlib import Path
from model.bert_classifier import BertClassifier

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "saved_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

label_encoder_path = MODEL_DIR / "label_encoder.joblib"
model_path = MODEL_DIR / "model.pt"

if not label_encoder_path.exists():
    raise FileNotFoundError(
        f"Label encoder not found at {label_encoder_path}. "
        "Please train the model first using the Jupyter notebook."
    )

if not model_path.exists():
    raise FileNotFoundError(
        f"Model weights not found at {model_path}. "
        "Please train the model first using the Jupyter notebook."
    )

label_encoder = joblib.load(str(label_encoder_path))

model = BertClassifier(num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load(str(model_path), map_location=device))
model.to(device)
model.eval()


def extract_text_from_pdf(pdf_path):
    try:
        import fitz
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
            
        return text
    except ImportError:
        raise ImportError("PyMuPDF not installed. Install with: pip install PyMuPDF")
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def predict_pdf(pdf_path):
    try:
        text = extract_text_from_pdf(pdf_path)
        encoding = tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=512, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            predicted = torch.argmax(outputs, dim=1)
            label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
            return label
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")
