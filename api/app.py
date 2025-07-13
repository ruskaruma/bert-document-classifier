from fastapi import FastAPI, File, UploadFile
from model.inference import predict_pdf
import shutil

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    with open("temp.pdf", "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = predict_pdf("temp.pdf")
    return {"prediction": result}