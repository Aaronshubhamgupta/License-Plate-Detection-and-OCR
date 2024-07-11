import streamlit as st
import cv2
import pytesseract
import numpy as np
import imutils

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to perform license plate detection and OCR
def detect_and_ocr_license_plate(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing techniques - adjust these parameters as needed
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
    dilate = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the processed image
    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Initialize license plate contour and bounding box
    screenCnt = None
    license_plate = None

    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
     
        # if the contour has four vertices, then we have found the license plate
        if len(approx) == 4:
            screenCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            # Validate the aspect ratio to filter out non-license plate contours
            if aspect_ratio > 2.5 and aspect_ratio < 5.5:
                license_plate = image[y:y + h, x:x + w]
                break
    
    # Perform OCR on the license plate
    if license_plate is not None:
        gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_plate, config='--psm 8')
        return text.strip()
    else:
        return None

# Main Streamlit application
def main():
    st.title('License Plate Detection and OCR')
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, -1)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform license plate detection and OCR
        text = detect_and_ocr_license_plate(image)

        # Display the extracted text
        if text:
            st.header("Extracted License Plate Number:")
            st.write(text)
        else:
            st.write("No valid license plate detected or OCR failed.")

if __name__ == '__main__':
    main()
