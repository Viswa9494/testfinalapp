from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='trained_vgg16_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define image size
image_size = (224, 224)

def preprocess_image_from_base64(base64_str):
    try:
        # Decode base64 string into bytes
        image_bytes = base64.b64decode(base64_str)
        
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))

        # Resize the image to match the input size expected by the model
        img = img.resize(image_size)

        # Convert image to array
        img_array = image.img_to_array(img)

        # Expand dimensions to match the input shape of the model
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess input (normalize pixel values and apply required transformations)
        img_array = preprocess_input(img_array)

        return img_array
    except Exception as e:
        print("Error processing image:", e)
        return None

def predict_image_class_from_base64(base64_str):
    # Preprocess the image
    img_array = preprocess_image_from_base64(base64_str)

    if img_array is not None:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Perform inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Decode predictions
        class_names = ['Gateway of India', 'India gate pics', 'Sun Temple Konark',
                       'charminar', 'lotus_temple', 'qutub_minar', 'tajmahal']  # Replace with your actual class names
        predicted_class_index = np.argmax(output_data)
        predicted_class = class_names[predicted_class_index]
        confidence = output_data[0][predicted_class_index]

        return predicted_class, confidence
    else:
        return "Error", 0.0


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    base64_str = data.get('image')

    if base64_str:
        # Predict class
        predicted_class, confidence = predict_image_class_from_base64(base64_str)
        # Convert confidence to Python float
        confidence = float(confidence)
        # Return prediction
        return jsonify({"predicted_class": predicted_class, "confidence": confidence})
    else:
        return jsonify({"error": "Base64 image data not provided"}), 400


if __name__ == "__main__":
    app.run(debug=True)
