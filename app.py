import cv2, io, zipfile, json, numpy as np
from ocr_detection import OCRDetection
from lama_inpainting import LamaInpainting
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
ocr = OCRDetection()
inpainter = LamaInpainting()

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

@app.post("/text-erase")
async def eraseText(image: UploadFile=File(...)):
    if image.content_type.startswith("image/"):
        #get and decode image 
        image_decode = cv2.imdecode(np.frombuffer(await image.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

        #get height and width of the image 
        h, w = image_decode.shape[:2]

        #copy original image 
        image_copy = image_decode.copy()

        #make ocr detection 
        ocr_result = ocr.detection(image_decode)

        #creating mask for inpainting 
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for data in ocr_result:
            x, y = data["pos_x"], data["pos_y"]
            w, h = data["width"], data["height"]

            #draw fill rect on the mask at the position where the text has been detected
            mask = cv2.rectangle(mask, (x, y),(x+w, y+h), (255, 255, 255), -1)

            #draw rect with thickness on the image copy at the position where the text has been detected
            image_copy = cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 0, 255), 3)

        

        #inpainting process 
        final_result = inpainter.inpaint(image_decode, mask)
        final_result = np.array(final_result)

        # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        # final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
        
        #encode inpaint image and ocr image 
        success1, ocr_image_encode = cv2.imencode(".png", image_copy)
        success2, final_result_encode = cv2.imencode(".png", final_result)

        if success1 and success2:
            #pack all these data in a zipfile
            #         
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf :
                zipf.writestr("data.json", json.dumps(ocr_result)) 
                zipf.writestr("ocr-result-image.png", ocr_image_encode.tobytes())
                zipf.writestr("inpaint-result-image.png", final_result_encode.tobytes())
            
            zip_buffer.seek(0)

            return StreamingResponse(zip_buffer, media_type="application/zip", headers={
                "Content-Disposition" : "attachment; filename=data.zip"
            })
        
    else : 
        return "Please Select a right file before send (.png, .jpg, .jpeg, .gif, ...)"