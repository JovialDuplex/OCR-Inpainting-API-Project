from ocr_detection import OCRDetection
from lama_inpainting import LamaInpainting
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(CORSMiddleware, 
                   allow_origins = ["*"],
                   allow_credentials= True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.get("/")
def hello():
    return {
        "api_name": "Image Text Erase API",
        "version": 1.0,
        "message": "Welcome to my API, ",
        "description": "This API is able to remove all the texts in image, and give in output the image without text(Inpaint-Part), the image with a text detected (OCR-Part)"
    }

