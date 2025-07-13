import torch
from transformers import BertTokenizer
import joblib
from model.bert_classifier import BertClassifier

#Loading the tokeniser and label encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_encoder = joblib.load("saved_model/label_encoder.joblib")

#For loading the model
model = BertClassifier(num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("saved_model/model.pt", map_location=device))
model.to(device)
model.eval()

def extract_text_from_pdf(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def predict_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    
    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        predicted = torch.argmax(outputs, dim=1)
        label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        return label
