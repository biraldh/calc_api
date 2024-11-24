from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np
import os
import torch
import cv2
import math


app = Flask(__name__)

# load models
class_model = tf.keras.models.load_model("E:/Rem/nutritionapp/segmentationmodels/best_model_food_class.h5")
model = load_model("E:/Rem/nutritionapp/segmentationmodels/best_model.h5")
model_type = "DPT_Hybrid"  
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Preprocess the image for the model
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image



    


def food_classify(img):
    img = img.resize((224, 224))  
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    # predict
    predictions = class_model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]

    class_names = ["apple", "bean",  "boiled egg", "chicken breast", "fried egg","rice", "salad", "spaghetti","steak"]
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

def estimate_depth(image):
    img_py= np.array(image) 
    img_rgb = cv2.cvtColor(img_py, cv2.COLOR_RGB2BGR)
    input_batch = midas_transforms(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map


def calculate_mask_area(mask, ppi):
    pixel_area = np.count_nonzero(mask)
    
    # convert pixel area to real-world area in square inches
    real_area_in_square_inches = (pixel_area / (ppi ** 2)) * (12 / 10) ** 2
    return real_area_in_square_inches


def density_get(food_type):

    densities = {
        "apple": 0.95,
        "chicken breast": 0.95,
        "steak": 0.92,
        "spaghetti": 1.3,
        "bean": 0.80,
        "boiled egg": 1.03,
        "fried egg": 1.09,
        "salad" : 0.3 ,
        "rice" : 0.75
    }

    # Return the density or error if the food type is not found
    return densities.get(food_type.lower(), None)  

def food_calories(food_type):
    
    calories = {
        "apple": {
            "cal": 52,  
            "weight": 100 
        },
        "salad": {
            "cal": 20,
            "weight": 85
        },
        "chicken breast": {
            "cal": 163,
            "weight": 100
        },
        "steak": {
            "cal": 614,
            "weight": 221
        },
        "spaghetti": {
            "cal": 210,
            "weight": 140
        },
        "bean": {
            "cal": 94,
            "weight": 100
        },
        "boiled egg": {
            "cal": 97,
            "weight": 100
        },
        "fried egg": {
            "cal": 90,
            "weight": 46
        },
        "rice": {
            "cal": 205,
            "weight": 158
        }
    }
    # Check if the food type exists in the dictionary
    if food_type.lower() in calories:
        food_data = calories[food_type.lower()]
        return food_data["cal"], food_data["weight"]
    else:
        return "Food type not found"



def calculate_volume_and_weight(area_in_inches2, depth_map, ppi, density):
    # Convert depth map to inches
    depth_in_inches = depth_map / ppi  # Vectorized operation

    # Compute the per-pixel area in inches²
    pixel_area_in_inches2 = area_in_inches2 / depth_map.size

    # Compute volume per pixel and sum it
    total_volume_in_inches3 = np.sum(depth_in_inches * pixel_area_in_inches2)

    # Convert volume from inches³ to cm³
    total_volume_in_cm3 = total_volume_in_inches3 * 16.387

    # Calculate the weight in grams
    weight_in_grams = total_volume_in_cm3 * density

    return weight_in_grams


def calculate_calories(weight_in_grams, cal, weight):
    cal_per_gram = cal/weight
    calorie = weight_in_grams * cal_per_gram
    return calorie

# end point for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    ppi = 71
    food_type = food_classify(image)

    density_food = density_get(food_type)

    food_calorie, weight = food_calories(food_type)

    if density_food is None:
        return jsonify({
            "error": f"Food type '{food_type}' is not recognized. Please try with a different image."
        }), 400
        
    # preprocess and make prediction
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    threshold = float(request.args.get("threshold", 0.012))
    
    predicted_mask = (prediction[0] > threshold).astype(np.uint8)

    area = calculate_mask_area(predicted_mask, ppi)

    depth_map = estimate_depth(image)

    mask_list = predicted_mask.tolist()
    
    weight_in_grams = calculate_volume_and_weight(area, depth_map,ppi,density_food)

    calorie = calculate_calories(weight_in_grams, food_calorie, weight)

    return jsonify({
        "calories" : calorie,
        "weight": weight_in_grams,
        "food": food_type
    })

# run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


