from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from model.inference import predict_pdf
import shutil
import os
import tempfile

app = FastAPI(
    title="BERT Document Classifier API",
    description="Classify scientific document references into Primary, Secondary, or Missing categories",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "message": "BERT Document Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict/": "POST - Upload PDF for classification",
            "/health": "GET - Health check",
            "/docs": "Swagger UI documentation"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "bert-base-uncased"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF files are accepted."
        )
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        result = predict_pdf(temp_path)
        
        return JSONResponse(
            content={
                "filename": file.filename,
                "prediction": result,
                "status": "success"
            },
            status_code=200
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        if temp_file and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
