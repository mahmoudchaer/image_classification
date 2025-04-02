from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io
from model import classifier

app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using a pre-trained model",
    version="0.1.0"
)

@app.get("/status")
async def status():
    """Check if the API is running."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        
        # Use the model to predict
        result = classifier.predict_image(contents)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 