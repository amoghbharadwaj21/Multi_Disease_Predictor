from flask import Flask, render_template, request, flash, redirect, jsonify
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import os


app = Flask(__name__)

def predict(values, dic):
    try:
        # Convert input values to NumPy array
        values = np.asarray(values)
        print(f'Length of values: {len(values)}')
        if len(values) == 8:
            model_path = 'models/diabetes.pkl'
        elif len(values) == 26:
            model_path = 'models/cancer.pkl'
        elif len(values) == 13:
            model_path = 'models/heart.pkl'
        elif len(values) == 18:
            model_path = 'models/kidney.pkl'
        elif len(values) == 10:
            model_path = 'models/liver.pkl'
        else:
            raise ValueError("Unexpected number of features for prediction")

        # Load model
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Make prediction
        prediction = model.predict(values.reshape(1, -1))[0]
        return prediction

    except FileNotFoundError as fnf_error:
        print(f"Model file not found: {fnf_error}")
        raise
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predictPage", methods = ['POST', 'GET'])
def predictPage():
    try:
        print("Inside Try block")
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            print(f'Predict dict:{to_predict_dict}')
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print(f'Predict List: {to_predict_list}')
            pred = predict(to_predict_list, to_predict_dict)
            print(f'Prediction: {pred}')
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)

# @app.route("/malariapredict", methods=['POST', 'GET'])
# def malariapredictPage():
#     pred = None
#     if request.method == 'POST':
#         try:
#             if 'image' in request.files:
#                 # Get the uploaded image
#                 file = request.files['image']
                
#                 # Convert the image to a format that Keras expects (using BytesIO)
#                 img = Image.open(file.stream)  # Open the image directly from the file stream
                
#                 # Resize the image to match the model input size
#                 img = img.resize((100, 100))  # Resize to match input size of the model
                
#                 # Convert image to numpy array and preprocess for prediction
#                 img = img_to_array(img)  # Convert to array
#                 img = np.expand_dims(img, axis=0)  # Add batch dimension
#                 img = img.astype('float32') / 255.0  # Normalize the pixel values
                
#                 # Load the malaria detection model
#                 model = load_model("models/malaria_cnn.keras")
                
#                 # Make the prediction
#                 prediction = model.predict(img)  # Assuming binary classification
#                 print(prediction)
#                 # Interpret the result (adjust prediction threshold as needed)
#                 if prediction[0] > prediction[1]:
#                     pred = "No malaria detected"
#                 else:
#                     pred = "Malaria detected"
#             else:
#                 message = "Please upload an Image"
#                 return render_template('malaria.html', message=message)
#         except Exception as e:
#             # Handle errors
#             message = f"An error occurred: {str(e)}"
#             return render_template('malaria.html', message=message)
#     print(pred)
#     return render_template('malaria_predict.html', pred=pred)

def predict_cell_status(image_path, model):
    # Load the image with target size (100, 100)
    test_image = load_img(image_path, target_size=(100, 100))
    
    # Convert image to array
    test_image = img_to_array(test_image)
    
    # Expand dimensions to match the input shape of the model
    test_image = np.expand_dims(test_image, axis=0)

    # Predict the class label (0 for uninfected, 1 for infected)
    prediction = model.predict(test_image)

    # Return the label based on the prediction
    if prediction[0][0] == 1:
        return "Uninfected"
    else:
        return "Infected"

@app.route("/malaria_predict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # Check if the user has uploaded a file
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Save the file to a separate folder (stored_images)
        stored_dir = 'stored_images'  # Ensure this directory exists or create it
        if not os.path.exists(stored_dir):
            os.makedirs(stored_dir)

        # Generate the full file path and save the uploaded file
        stored_path = os.path.join(stored_dir, file.filename)
        file.save(stored_path)

        # Make prediction using the stored image file path
        model = load_model("models/malaria_cnn.keras")
        prediction = predict_cell_status(stored_path, model)
        
        # Return the result with the image path and prediction
        return render_template('malaria_predict.html', prediction=prediction, image_path=stored_path)

    # If it's a GET request, render the page with the image upload form
    return render_template('malaria_predict.html')



@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    pred = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                # Load and preprocess the image
                img = Image.open(request.files['image']).convert('RGB')
                img = img.resize((224, 224))  # Resize to match model input size
                img = img_to_array(img)  # Convert to numpy array
                img = img.astype(np.float32) / 255.0  # Normalize pixel values
                img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 224, 224, 3)
                
                # Load the pneumonia detection model
                model = load_model("D:/CodeGround/Multi_Disease_Predictor/models/pneumonia_cnn.keras")
                
                # Make the prediction
                prediction = model.predict(img)[0]  # Assuming binary classification
                print(prediction)
                # Interpret the result
                if prediction[1] > 0.5:  # Assuming index 1 is for pneumonia
                    pred = 1  # Pneumonia detected
                else:
                    pred = 0  # No pneumonia detected
            else:
                message = "Please upload an Image"
                return render_template('pneumonia.html', message=message)
        except Exception as e:
            # Handle errors
            message = f"An error occurred: {str(e)}"
            return render_template('pneumonia.html', message=message)
    
    return render_template('pneumonia_predict.html', pred=pred)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

