import cv2
import onnxruntime as ort 
import numpy as np 

class LamaInpainting:
    def __init__(self, model_path):

        #initializing of my session 
        self.session = ort.InferenceSession(model_path)

        