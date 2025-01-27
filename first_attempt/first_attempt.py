import cv2
import os

def sketch_image(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray=gray
    inverted_gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted_gray, (37, 37), sigmaX=0, sigmaY=0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    cv2.imwrite(output_path, sketch)

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"sketch_{filename}")
            sketch_image(input_path, output_path)

input_folder = 'images'
output_folder = 'results'

process_images(input_folder, output_folder)

print("Processing complete. Check the 'results' folder for output.")
