from flask import Flask, render_template, request, flash, redirect, jsonify
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import os
import google.generativeai as genai


GeminiAPIKey=os.getenv("GeminiAPIKey")
print(GeminiAPIKey)
genai.configure(api_key=GeminiAPIKey)
model = genai.GenerativeModel("gemini-1.5-flash")


app = Flask(__name__)


def predict(values, dic):
    try:
        # Convert input values to NumPy array
        values = np.asarray(values)
        print(f"Length of values: {len(values)}")
        if len(values) == 8:
            model_path = "models/diabetes.pkl"
        elif len(values) == 26:
            model_path = "models/cancer.pkl"
        elif len(values) == 13:
            model_path = "models/heart.pkl"
        elif len(values) == 18:
            model_path = "models/kidney.pkl"
        elif len(values) == 10:
            model_path = "models/liver.pkl"
        else:
            raise ValueError("Unexpected number of features for prediction")

        # Load model
        with open(model_path, "rb") as model_file:
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


def get_report(pred, dic):
    print("Inside get_report")
    print(f"Dictionary: {dic}")

    if pred == 1:
        # If the patient is suffering from the illness
        prompt = f"""
        Given the medical report data of a patient, generate a comprehensive health status report. The model should:

        1. Analyze the patient's condition based on the provided medical data, identifying any signs or risks.
        2. Provide a prognosis and any associated risks, including possible links to other diseases (e.g., the risk of kidney failure due to diabetes, heart disease related to liver dysfunction).
        3. Offer treatment recommendations, including both lifestyle changes, medications (Allopathic), and any specific tests that may be required for further investigation.
        4. Mention any other risk factors or conditions that the patient may need to monitor (e.g., heart disease, kidney disease).
        5. Indicate any required follow-up actions, tests, or consultations with specialists (e.g., endocrinologist for diabetes, cardiologist for heart disease).
        6. Avoid using bold text or any kind of special formatting in the report. Use numbers for listing points. Use nested numbers (e.g., 1.1, 1.2) for sub-points.
        7. Ensure that the report is clear, concise, and medically appropriate.
        8. Provide the heading as "Patient Health Status Report" at the beginning of the report.
        9. Maintain a professional tone suitable for healthcare documentation.
        10. Add closing note as "This report is based solely on the provided laboratory data. A comprehensive evaluation by a healthcare professional is essential for a complete assessment of the patient's health and appropriate management."

        Here is the patient's medical data:
        {dic}
        """
    else:
        # If the patient is not suffering from the illness
        prompt = f"""
        Given the medical report data of a patient, generate a positive health report. The model should:

        1. Highlight the patient’s current healthy status while being vigilant about any potential concerns in the medical data.
        2. Provide recommendations for maintaining good health, including preventive measures and medications(Allopathic) if necessary.
        3. Identify specific values that the patient should monitor (e.g., slightly elevated glucose or BMI).
        4. Suggest lifestyle habits or periodic check-ups to avoid upcoming risks.
        5. Offer advice on maintaining a balanced diet, regular exercise, and stress management for long-term health.
        6. Avoid using bold text or any kind of special formatting in the report. Use numbers for listing points. Use nested numbers (e.g., 1.1, 1.2) for sub-points.
        7. Ensure that the report is clear, concise, and medically appropriate.
        8. Provide the heading as "Patient Health Status Report" at the beginning of the report.
        9. Maintain a professional tone suitable for healthcare documentation.
        10. Add closing note as "This report is based solely on the provided laboratory data. A comprehensive evaluation by a healthcare professional is essential for a complete assessment of the patient's health and appropriate management."

        Here is the patient's medical data:
        {dic}
        """

    print(f"Prompt: {prompt}")
    response = model.generate_content(prompt)
    print(f"Response: {response}")
    cleaned_response = (
        response.text.replace("\n", "<br>").replace("*", "").replace("#", "")
    )
    print(f"Cleaned Response: {cleaned_response}")
    return cleaned_response


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/general-health", methods=["GET", "POST"])
def generalHealthPage():
    if request.method == "POST":
        print(request.form.to_dict())
        form_data = request.form.to_dict()
        prompt = f""" 
        You are a friendly and professional medical assistant AI. Your goal is to provide a positive, empathetic, and actionable diagnosis for a patient based on their symptoms.  Below are the details of the patient.
        {form_data}
        Analyze the information and provide the following:
        1. A diagnosis listing possible conditions.
        2. Potential health risks associated with the symptoms.
        3. Medications or treatments to consider, including dosage and precautions.
        4. Lifestyle and dietary recommendations.
        5. Instructions for follow-up if symptoms persist.
        6. Avoid using bold text or any kind of special formatting in the report. Use numbers for listing points. Use nested numbers (e.g., 1.1, 1.2) for sub-points.
        7. Ensure that the report is clear, concise, and medically appropriate.
        8. Provide the heading as "Patient Health Status Report" at the beginning of the report.
        9. Add closing note as "This report is based solely on the provided data. A comprehensive evaluation by a healthcare professional is essential for a complete assessment of the patient's health and appropriate management."
        10. Don't tell that you don't have enough information to make a diagnosis. Instead, provide a general diagnosis based on the symptoms.
        11. Don't tell that you can not provide a prescription. Instead, provide a general prescription for the symptoms.
        
        Your Response Should Include:

        1. Diagnosis:
        List potential conditions based on the symptoms.
        
        2. Potential Risks:
        Highlight any significant health risks associated with the symptoms.
        
        3. Recommended Medications:
        Provide names of medications (if applicable), dosage, and precautions.
        Ensure to mention when to consult a doctor for prescription-only medications.
        
        4. Lifestyle and Dietary Recommendations:
        Suggest actionable steps the patient can take at home to alleviate symptoms.
        
        5. Follow-Up Instructions:
        Specify when the patient should seek medical attention if symptoms persist or worsen.
        Respond in the form of a structured prescription for clarity and usability.
        
        """
        response = model.generate_content(prompt)
        print(f"Response: {response}")
        cleaned_response = (
            response.text.replace("\n", "<br>").replace("*", "").replace("#", "")
        )
        print(f"Cleaned Response: {cleaned_response}")
        
        return render_template("general_health_diagnosis.html",diagnosis=cleaned_response)
    else:
        return render_template("general_health.html")


@app.route("/diabetes", methods=["GET", "POST"])
def diabetesPage():
    return render_template("diabetes.html")


@app.route("/cancer", methods=["GET", "POST"])
def cancerPage():
    return render_template("breast_cancer.html")


@app.route("/heart", methods=["GET", "POST"])
def heartPage():
    return render_template("heart.html")


@app.route("/kidney", methods=["GET", "POST"])
def kidneyPage():
    return render_template("kidney.html")


@app.route("/liver", methods=["GET", "POST"])
def liverPage():
    return render_template("liver.html")


@app.route("/malaria", methods=["GET", "POST"])
def malariaPage():
    return render_template("malaria.html")


@app.route("/pneumonia", methods=["GET", "POST"])
def pneumoniaPage():
    return render_template("pneumonia.html")


@app.route("/predictPage", methods=["POST", "GET"])
def predictPage():
    try:
        print("Inside Try block")
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            print(f"Predict dict:{to_predict_dict}")
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print(f"Predict List: {to_predict_list}")
            pred = predict(to_predict_list, to_predict_dict)
            report = get_report(pred, to_predict_dict)
            print(f"request: {request.form}")
            print(f"list: {to_predict_list}")
            print(f"dict: {to_predict_dict}")
            print(f"Prediction: {pred}")
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message=message)

    return render_template("predict.html", pred=pred, report=report)


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


@app.route("/malaria_predict", methods=["POST", "GET"])
def malariapredictPage():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        # Check if the user has uploaded a file
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        # Save the file to a separate folder (stored_images)
        stored_dir = "stored_images"  # Ensure this directory exists or create it
        if not os.path.exists(stored_dir):
            os.makedirs(stored_dir)

        # Generate the full file path and save the uploaded file
        stored_path = os.path.join(stored_dir, file.filename)
        file.save(stored_path)

        # Make prediction using the stored image file path
        model = load_model("models/malaria_cnn.keras")
        prediction = predict_cell_status(stored_path, model)

        # Return the result with the image path and prediction
        return render_template(
            "malaria_predict.html", prediction=prediction, image_path=stored_path
        )

    # If it's a GET request, render the page with the image upload form
    return render_template("malaria_predict.html")


@app.route("/pneumoniapredict", methods=["POST", "GET"])
def pneumoniapredictPage():
    pred = None
    if request.method == "POST":
        try:
            if "image" in request.files:
                # Load and preprocess the image
                img = Image.open(request.files["image"]).convert("RGB")
                img = img.resize((224, 224))  # Resize to match model input size
                img = img_to_array(img)  # Convert to numpy array
                img = img.astype(np.float32) / 255.0  # Normalize pixel values
                img = np.expand_dims(
                    img, axis=0
                )  # Add batch dimension (1, 224, 224, 3)

                # Load the pneumonia detection model
                model = load_model("models/pneumonia_cnn.keras")

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
                return render_template("pneumonia.html", message=message)
        except Exception as e:
            # Handle errors
            message = f"An error occurred: {str(e)}"
            return render_template("pneumonia.html", message=message)

    return render_template("pneumonia_predict.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
