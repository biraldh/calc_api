from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import math

# Initialize the Flask app
app = Flask(__name__)

# Load the YOLO model
yolo_model = YOLO('yolov8m.pt')

food_classes = {46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake'}

# Load a depth estimation model (e.g., MiDaS or DPT)
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # MiDaS depth model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model.to(device).eval()

# Load the transforms for the depth model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def get_exif_data(image):
    """
    Extract EXIF data from an image and convert it to a readable format.
    """
    exif_data = {}
    try:
        raw_exif_data = image._getexif()
        if raw_exif_data is not None:
            for tag, value in raw_exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                # If the value is bytes, convert it to a readable format
                if isinstance(value, bytes):
                    value = "-"
                exif_data[tag_name] = value
        else:
            exif_data = "No EXIF data found"
    except AttributeError:
        exif_data = "No EXIF data found"
    return exif_data

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

def calculate_volume_and_weight(bbox, depth_map, ppi, height_pixels_img, width_pixels_img, density=0.80):
    """
    Calculate area, volume, and weight based on bounding box and depth information.
    :param bbox: Bounding box coordinates (x1, y1, x2, y2)
    :param depth_map: Depth map for the image
    :param ppi: Pixels per inch for the image
    :param density: Density of the object (default is 0.96 g/cm³ for apples)
    :return: Estimated area in inches^2, volume in inches^3, and weight
    """
    x1, y1, x2, y2 = map(int, bbox)
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
    img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    
    # Retrieve EXIF data and use it to calculate PPI
    exif_data = get_exif_data(image)
    width_pixels_img, height_pixels_img = image.size

    # Define PPI calculation based on screen diagonal example values
    width_pixels = 1920
    height_pixels = 1080
    diagonal_inches = 16
    ppi = math.sqrt(width_pixels**2 + height_pixels**2) / diagonal_inches
  # Perform YOLO prediction on the image
    
    results = yolo_model.predict(source=img, conf=0.25, save=False, show=False)
    depth_map = estimate_depth(img)
    
    predictions = []
    for result in results[0].boxes.data:
        x1, y1, x2, y2 = result[:4].tolist()
        conf = result[4].item()
        category = int(result[5].item())
        
        area, volume, weight = calculate_volume_and_weight(
            [x1, y1, x2, y2], depth_map, ppi, height_pixels_img, width_pixels_img)
        
        if category in food_classes:
            predictions.append({
                "category": food_classes[category],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "area_in_inches_squared": area,
                "volume_in_inches_cubed": volume,
                "weight_in_grams": weight
            })
        else:
            return jsonify({"error": "Detected object is not food"}), 400
    
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
