from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import math

# Initialize the Flask app
app = Flask(__name__)

# Load the Keras segmentation model (e.g., U-Net or similar)
saved_model_path = "E:/Rem/nutritionapp/segmentationmodels/best_model.h5"  # Update with your model path
segmentation_model = load_model(saved_model_path)  # Load the model

food_classes = {46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake'}

# Load the MiDaS depth model for depth estimation
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # MiDaS depth model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model.to(device).eval()

# Load the transforms for the depth model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def estimate_depth(image):
    """ Estimate depth map for the given image. """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = midas_transforms(img_rgb).to(device)
    with torch.no_grad():
        prediction = depth_model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

def calculate_volume_and_weight(mask, depth_map, ppi, height_pixels_img, width_pixels_img, density=0.80):
    """
    Calculate area, volume, and weight based on the segmentation mask and depth information.
    :param mask: Segmentation mask (binary mask)
    :param depth_map: Depth map for the image
    :param ppi: Pixels per inch for the image
    :param density: Density of the object (default is 0.96 g/cm³ for apples)
    :return: Estimated area in inches^2, volume in inches^3, and weight
    """
    # Get the bounding box of the mask
    y_indices, x_indices = np.where(mask == 1)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, 0, 0
    
    x1, y1 = np.min(x_indices), np.min(y_indices)
    x2, y2 = np.max(x_indices), np.max(y_indices)

    # Extract the depth values within the bounding box
    depth_region = depth_map[y1:y2, x1:x2]
    avg_depth = np.mean(depth_region)

    # Calculate dimensions in inches
    width_inches = (x2 - x1) / ppi
    height_inches = (y2 - y1) / ppi
    depth_inches = avg_depth / ppi

    # Calculate area in square inches
    area_in_inches = width_inches * height_inches

    # Calculate volume in cubic inches (approximate cuboid volume)
    volume_in_inches = width_inches * height_inches * depth_inches

    # Convert volume from cubic inches to cubic cm for weight calculation
    volume_in_cm3 = volume_in_inches * 16.387
    weight_in_grams = density * volume_in_cm3

    return area_in_inches, volume_in_inches, weight_in_grams

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Read the image file
    image_file = request.files['image']
    image = Image.open(image_file)

    # Convert to OpenCV format
    np_arr = np.array(image)
    img = cv2.cvtColor(np_arr)
    
    # Retrieve EXIF data and use it to calculate PPI
    width_pixels_img, height_pixels_img = image.size

    # Define PPI calculation based on screen diagonal example values
    width_pixels = 1920
    height_pixels = 1080
    diagonal_inches = 16
    ppi = math.sqrt(width_pixels**2 + height_pixels**2) / diagonal_inches

    # Perform segmentation prediction using the Keras model
    img = img / 255.0  # Normalize the image
    img_resized = cv2.resize(img, (128, 128))  # Resize to model's input size (128 is typical)
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Perform prediction using the Keras model
    prediction = segmentation_model.predict(img_array)

    # The model will output a mask or segmentation map
    mask = prediction[0]  # Assuming the model outputs the mask for a single image
    mask = (mask > 0.5).astype(np.uint8)  # Thresholding the mask to remove noise

    # Estimate depth for volume and weight calculation
    depth_map = estimate_depth(img)

    # Calculate area, volume, and weight
    area, volume, weight = calculate_volume_and_weight(mask, depth_map, ppi, height_pixels_img, width_pixels_img)

    # Assuming a simple classifying approach based on the bounding box
    predictions = [{
        "category": "food_item",  # You can customize based on your class prediction
        "area_in_inches_squared": area,
        "volume_in_inches_cubed": volume,
        "weight_in_grams": weight
    }]

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
