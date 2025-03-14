import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and scaler
model_path = r"model.pkl"
scaler_path = r"scale.pkl"

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')  # Route to display the home page
def home():
    return render_template('index.html')  # Rendering the home page

@app.route('/predict', methods=["POST", "GET"])  # Route to show the predictions in a web UI
def predict():
    try:
        # Reading the inputs given by the user
        input_features = [float(x) for x in request.form.values()]
        features_values = [np.array(input_features)]

        # Feature names in the same order as used for training
        feature_names = [['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']]
        data = pd.DataFrame(features_values, columns=feature_names)

        # Ensure the input data is correctly scaled
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data, columns=feature_names)

        # Predict using the loaded model
        prediction = model.predict(data)
        print(prediction)

        text = "Estimated Traffic Volume is: "
        return render_template("index.html", prediction_text=text + str(prediction[0]))

    except Exception as e:
        print(f"Error: {e}")
        return render_template("index.html", prediction_text="An error occurred. Please check your input.")

if __name__ == "__main__":
    # Running the app
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
