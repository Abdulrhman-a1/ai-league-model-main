from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from predictor import predict_density  # نستخدم الدالة من predictor.py

app = FastAPI(
    title="Crowd Counting API",
    description="API تستخدم نموذج CSRNet لحساب عدد الأشخاص في الصورة باستخدام الذكاء الاصطناعي.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", summary="احسب عدد الأشخاص", description="يرفع صورة ويحسب عدد الأشخاص فيها باستخدام CSRNet.")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    count, _ = predict_density(image)
    return {"count": count}

