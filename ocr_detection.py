import easyocr, cv2

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
    print("test")
    ocr = OCRDetection()
    image = cv2.imread("image1.jpg")
    result = ocr.detection(image)
    print(result)