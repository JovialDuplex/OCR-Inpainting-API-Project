import cv2
import onnxruntime as ort 
import numpy as np 
from PIL import Image

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

if __name__ == "__main__":
    inpainter = LamaInpainting("model/lama.onnx")
    result = inpainter.inpaint("image.png", "mask.png")
    result.save("result.png")