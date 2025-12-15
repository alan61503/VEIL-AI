import easyocr
import cv2

reader = easyocr.Reader(['en'])

def read_plate(plate_img):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # OCR
    result = reader.readtext(gray)
    if result:
        return result[0][1].replace(" ", "")
    return None
