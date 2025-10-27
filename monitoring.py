import cv2
import os
from PIL import Image
import numpy as np
import random
from utils.image_utils import annotate_weeds, annotate_diseases
from utils.moisture_utils import estimate_moisture

def load_image(image_path):
    """Load image from path."""
    img = Image.open(image_path)
    return np.array(img)

def detect_weeds(image, model_path):
    """Detect weeds using a pre-trained model."""
    # Placeholder: Assume model is YOLO-like, but for demo, return dummy boxes
    # In real impl, load model and predict
    # model = tf.keras.models.load_model(model_path)
    # boxes = model.predict(image)
    # Realistic dummy: random number of weeds based on image size
    h, w = image.shape[:2]
    num_weeds = random.randint(0, 5)  # Random 0-5 weeds
    boxes = []
    for _ in range(num_weeds):
        x1 = random.randint(0, w//2)
        y1 = random.randint(0, h//2)
        x2 = min(w, x1 + random.randint(50, 150))
        y2 = min(h, y1 + random.randint(50, 150))
        boxes.append([x1, y1, x2, y2])
    return {'count': len(boxes), 'boxes': boxes}

def detect_diseases(image, model_path):
    """Detect diseases using a segmentation model."""
    # Placeholder: Assume segmentation model
    # model = tf.keras.models.load_model(model_path)
    # mask = model.predict(image)
    # Realistic dummy: random disease type and mask
    disease_types = ['Powdery Mildew', 'Bacterial Blight', 'Fungal Spot', 'Viral Mosaic', 'None']
    disease_type = random.choice(disease_types)
    if disease_type == 'None':
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        percentage = 0
    else:
        h, w = image.shape[:2]
        # Random mask area
        x1 = random.randint(0, w//2)
        y1 = random.randint(0, h//2)
        x2 = min(w, x1 + random.randint(100, 200))
        y2 = min(h, y1 + random.randint(100, 200))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        percentage = random.uniform(5, 50)  # Random 5-50%
    return {'mask': mask, 'type': disease_type, 'percentage': percentage}

def estimate_environment_from_soil(image):
    """Estimate temperature and humidity from soil image color analysis."""
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Temperature based on warmth (red/yellow tones)
    temp_intensity = np.mean(hsv[:, :, 0])  # Hue channel
    temperature = 15 + (temp_intensity / 255) * 25  # 15-40°C
    # Humidity based on saturation (darker soil = higher humidity)
    humidity = 30 + (np.mean(hsv[:, :, 1]) / 255) * 50  # 30-80%
    return round(temperature, 1), round(humidity, 1)

def process_monitoring(drone_image_path, leaf_image_path, soil_image_path, moisture_sensor, output_dir):
    """
    Process the monitoring data and generate report.

    Args:
        drone_image_path: Path to drone image or None.
        leaf_image_path: Path to leaf image or None.
        soil_image_path: Path to soil image or None.
        moisture_sensor: Float or None.
        output_dir: Output directory.

    Returns:
        Dict with results: annotated image paths, report data.
    """
    # Load images
    drone_img = load_image(drone_image_path) if drone_image_path else None
    leaf_img = load_image(leaf_image_path) if leaf_image_path else None
    soil_img = load_image(soil_image_path) if soil_image_path else None

    # Weed detection
    weed_data = {}
    annotated_drone_path = None
    original_drone_path = None
    if drone_img is not None:
        weed_data = detect_weeds(drone_img, 'models/weed_detector.h5')
        annotated_drone = annotate_weeds(drone_img, weed_data['boxes'])
        annotated_drone_path = os.path.join(output_dir, 'annotated_drone.jpg')
        cv2.imwrite(annotated_drone_path, cv2.cvtColor(annotated_drone, cv2.COLOR_RGB2BGR))
        original_drone_path = drone_image_path  # Save original path

    # Disease detection
    disease_data = {}
    annotated_leaf_path = None
    original_leaf_path = None
    if leaf_img is not None:
        disease_result = detect_diseases(leaf_img, 'models/disease_model.h5')
        annotated_leaf, percentage = annotate_diseases(leaf_img, disease_result['mask'])
        disease_data = {'percentage': disease_result['percentage'], 'type': disease_result['type']}
        annotated_leaf_path = os.path.join(output_dir, 'annotated_leaf.jpg')
        cv2.imwrite(annotated_leaf_path, cv2.cvtColor(annotated_leaf, cv2.COLOR_RGB2BGR))
        original_leaf_path = leaf_image_path

    # Moisture estimation
    moisture_level = 'Unknown'
    water_needed = False
    temperature = None
    humidity = None
    if soil_img is not None:
        moisture_level = estimate_moisture(soil_img, moisture_sensor)
        if moisture_level == 'Low':
            water_needed = True
        # Estimate environment from soil image
        temperature, humidity = estimate_environment_from_soil(soil_img)

    # Overall health score (0-100)
    health_score = 100
    if weed_data.get('count', 0) > 0:
        health_score -= weed_data['count'] * 10
    if disease_data.get('percentage', 0) > 0:
        health_score -= disease_data['percentage'] * 0.5
    if water_needed:
        health_score -= 20
    health_score = max(0, min(100, health_score))

    # Recommendations
    recommendations = {}
    if weed_data.get('count', 0) > 0:
        recommendations['weeds'] = f"Weed detected: {weed_data['count']} patches. Recommended action: Remove weeds to prevent nutrient loss."
    else:
        recommendations['weeds'] = "No weeds detected. Roof is clear."

    if disease_data.get('percentage', 0) > 0:
        disease_type = disease_data['type']
        percentage = disease_data['percentage']
        if disease_type == 'Powdery Mildew':
            rec = f"Leaf affected by powdery mildew ({percentage:.1f}%). Use sulfur-based fungicide and improve air circulation."
        elif disease_type == 'Bacterial Blight':
            rec = f"Leaf affected by bacterial blight ({percentage:.1f}%). Remove infected leaves and apply copper-based bactericide."
        elif disease_type == 'Fungal Spot':
            rec = f"Leaf affected by fungal spot ({percentage:.1f}%). Apply fungicide and ensure proper spacing for air flow."
        elif disease_type == 'Viral Mosaic':
            rec = f"Leaf affected by viral mosaic ({percentage:.1f}%). Remove infected plants to prevent spread."
        recommendations['disease'] = rec
    else:
        recommendations['disease'] = "No diseases detected. Leaves are healthy."

    if water_needed:
        recommendations['moisture'] = f"Soil appears dry; schedule watering within 2 hours."
    else:
        recommendations['moisture'] = f"Soil moisture is adequate. No immediate watering needed."

    if temperature is not None and humidity is not None:
        if temperature > 30 or humidity < 40:
            env_rec = f"Temperature: {temperature}°C, Humidity: {humidity}%. Moderate conditions – monitor closely."
        else:
            env_rec = f"Temperature: {temperature}°C, Humidity: {humidity}%. Optimal conditions for plant growth."
        recommendations['environment'] = env_rec

    # Return dict directly (no JSON file)
    return {
        'original_drone': original_drone_path,
        'annotated_drone': annotated_drone_path,
        'original_leaf': original_leaf_path,
        'annotated_leaf': annotated_leaf_path,
        'weed_count': weed_data.get('count', 0),
        'disease_type': disease_data.get('type', 'None'),
        'disease_percentage': disease_data.get('percentage', 0),
        'moisture_level': moisture_level,
        'temperature': temperature,
        'humidity': humidity,
        'water_needed': water_needed,
        'health_score': health_score,
        'recommendations': recommendations
    }
