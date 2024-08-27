from flask import Flask, request, render_template, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model/cancer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        mean_radius = float(request.form['mean_radius'])
        mean_texture = float(request.form['mean_texture'])

        # Prepare the input data as a numpy array
        input_features = np.array([[mean_radius, mean_texture]])

        # Make the prediction
        prediction = model.predict(input_features)

        # Map the prediction to benign or malignant
        output = 'Malignant' if prediction[0] == 1 else 'Benign'
        
        # Determine the image path based on the prediction
        image_path = 'malignant.png' if output == 'Malignant' else 'benign.png'
        
        return render_template('form.html', prediction_text=f'The prediction is: {output}', image_path=image_path)
    except Exception as e:
        return str(e)  # Display any errors in the browser for debugging

if __name__ == "__main__":
    app.run(debug=True)
