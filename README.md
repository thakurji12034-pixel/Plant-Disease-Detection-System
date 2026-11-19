# Plant Disease Detection System

This project is an AI-based system for detecting plant diseases using a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture. It features a modern web interface where users can upload pictures of plant leaves to instantly receive disease predictions.

## Project Structure

- `mini project.html`: The frontend user interface. It provides an overview of the project and an interactive portal to upload leaf images.
- `app.py`: The backend Python server. It is built using Flask and serves as the bridge between the frontend and the AI model. 
- `model.keras`: The pre-trained Deep Learning CNN model (saved in Keras format) used to classify images into one of 7 categories.

## How It Works (The Connection)

1. **Frontend Request:** When an image is uploaded and "Predict Disease" is clicked in `mini project.html`, the JavaScript code uses the `fetch` API to send a POST request containing the image to `http://127.0.0.1:5000/predict`.
2. **Backend Processing:** The Flask application (`app.py`) listens on port 5000. It receives the image file, resizes it appropriately (usually to 224x224 for MobileNetV2 models), and standardizes the pixel values between `-1.0` and `1.0`. 
3. **AI Prediction:** The processed image array is fed into the loaded `model.keras` file. The backend runs this inference purely through PyTorch (using Keras 3 as the high-level API) which predicts the exact disease.
4. **Response:** `app.py` identifies the highest probability class, assigns a human-readable name (like "Early Blight"), and calculates the confidence percentage. It then returns this data as a JSON response.
5. **Frontend Display:** The `mini project.html` script reads the JSON response and updates the UI to show the disease name and confidence metric in real-time.

## Setup & Installation

To run this application on your local machine, you need Python installed. We use the PyTorch backend for Keras because it works nicely on Windows natively across newer Python versions.

1. Install the required Python dependencies:
   ```bash
   pip install keras torch torchvision numpy flask flask-cors pillow
   ```

2. Make sure the model file is named `model.keras`.

## Running the Application

1. **Start the Backend Server**: Open a terminal in this directory and run:
   ```bash
   python app.py
   ```
   *This starts up the Flask server to host your Keras model.*

2. **Open the Frontend UI**: Once the backend is running, open `mini project.html` in any modern web browser (Google Chrome, Microsoft Edge, etc.).

3. Upload an affected plant leaf and click **Predict Disease** to see the magic happen!

## Target Diseases Supported

The AI supports detection for the following cases:
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Target Spot
- Tomato Mosaic Virus
- Healthy Leaves
