from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.predict import PlantDiseasePredictor


app = FastAPI(
    title="Plant Disease Detection API",
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


with open("data/classes.txt", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]


predictor = PlantDiseasePredictor(
    model_path="model/efficientnet_finetuned_best.pth",
    classes=classes,
    device=device
)


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    temp_path = "app/temp_prediction.jpg"
    image.save(temp_path)

    # Predict ONCE
    result = predictor.predict(temp_path)

    # Delete temporary image
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "filename": file.filename,
        "predictions": result
    }