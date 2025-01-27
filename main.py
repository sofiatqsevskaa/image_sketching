import cv2
import os
import numpy as np

def add_pencil_shading(image):
    height, width = image.shape[:2]
    line_overlay = np.zeros_like(image)

    for i in range(0, width, 10):
        cv2.line(line_overlay, (i, 0), (0, i), color=(50, 50, 50), thickness=1)
        cv2.line(line_overlay, (width, i), (i, height), color=(50,50,50), thickness=1)

    shaded_image = cv2.addWeighted(image, 0.9, line_overlay, 0.1, 0)
    return shaded_image

def sketch_image(image_path, output_path_bw, output_path_color):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted_gray, (37, 37), sigmaX=0, sigmaY=0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch_bw = cv2.divide(gray, inverted_blur, scale=256.0)

    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edges = cv2.bitwise_not(edges)

    smooth = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    smooth_gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    detailed_sketch_bw = cv2.addWeighted(smooth_gray, 0.6, edges, 0.4, 0)
    final_sketch_bw = cv2.addWeighted(sketch_bw, 0.5, detailed_sketch_bw, 0.5, 0)
    final_sketch_bw_with_shading = add_pencil_shading(final_sketch_bw)

    cv2.imwrite(output_path_bw, final_sketch_bw_with_shading)

    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    detailed_sketch_color = cv2.addWeighted(img, 0.5, edges_color, 0.5, 0)
    final_sketch_color = cv2.addWeighted(img, 0.6, detailed_sketch_color, 0.4, 0)

    hsv = cv2.cvtColor(final_sketch_color, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 0.5
    final_sketch_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    color_change_lines = cv2.absdiff(img, smooth)
    color_change_lines_gray = cv2.cvtColor(color_change_lines, cv2.COLOR_BGR2GRAY)
    _, color_change_lines_threshold = cv2.threshold(color_change_lines_gray, 30, 255, cv2.THRESH_BINARY)

    color_change_lines_colored = cv2.cvtColor(color_change_lines_threshold, cv2.COLOR_GRAY2BGR)
    color_change_lines_colored = cv2.bitwise_and(color_change_lines_colored, edges_color)

    final_sketch_color_with_lines = cv2.addWeighted(final_sketch_color, 1, color_change_lines_colored, 0.4, 0)

    final_sketch_color_with_shading = add_pencil_shading(final_sketch_color_with_lines)

    cv2.imwrite(output_path_color, final_sketch_color_with_shading)

def process_images(input_folder, output_folder_bw, output_folder_color):
    if not os.path.exists(output_folder_bw):
        os.makedirs(output_folder_bw)
    if not os.path.exists(output_folder_color):
        os.makedirs(output_folder_color)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path_bw = os.path.join(output_folder_bw, f"sketch_{filename}")
            output_path_color = os.path.join(output_folder_color, f"colored_sketch_{filename}")
            sketch_image(input_path, output_path_bw, output_path_color)

input_folder = 'images'
output_folder_bw = 'results'
output_folder_color = 'colored_results'

process_images(input_folder, output_folder_bw, output_folder_color)

print("Processing complete. Check the 'results' and 'colored_results' folders for output.")
