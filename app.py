import cv2
import onnxruntime as ort 
import numpy as np 
from PIL import Image
import easyocr

class LamaInpainting:
    def __init__(self, model_path):

        #initializing of my session 
        self.session = ort.InferenceSession(model_path)

        #get inputs of my model 
        self.first_input = self.session.get_inputs()[0].name
        self.second_input = self.session.get_inputs()[1].name
        
        print(self.first_input, self.second_input)

    def preprocess(self, image, mask):
        #resizing data for the model 
        h, w = image.shape[:2]
        new_h = 512
        new_w = 512
        
        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h))

        # Formatting data for the model 
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None, ...]
        
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None, ...]

        return image, mask, (h, w)

    def inpaint(self, image_path, mask_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        input_image, input_mask, original_size = self.preprocess(image, mask)
        
        inputs = {
            str(self.first_input): input_image,
            str(self.second_input) : input_mask
        } 
        
        #runing model with inputs 
        output = self.session.run(None, inputs)[0]

        #post processing and getting output image
        result = np.clip(output[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

        result = cv2.resize(result, (original_size[1], original_size[0]))

        return Image.fromarray(result)

class OCRDetection:
    def __init__(self):

        #initializing of my ocr detection
        self.reader = easyocr.Reader(["fr", "en"], gpu=False)

    def detection(self, image):
        output_data = []

        ocr_result = self.reader.readtext(image)
        #getting of text position 
        for box, text, conf in ocr_result:
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]

            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)         

            output_data.append({
                "pos_x": x,
                "pos_y": y,
                "width": w,
                "height": h,
                "text": text,
                "precision": conf
            })
        
        return output_data

if __name__ == "__main__":
    
    model = hf_hub_download(repo_id="mayocream/lama-manga-onnx", filename="lama-manga.onnx")
    
    inpaint = LamaInpainting(model_path=model)
    image = cv2.imread("image.png", cv2.IMREAD_COLOR_RGB)
    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

    result = inpaint.inpaint(image, mask)
    result.save("result.png")
