from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using a pre-trained model",
    version="0.1.0"
)

@app.get("/status")
async def status():
    """Check if the API is running."""
    return {"status": "ok"}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 