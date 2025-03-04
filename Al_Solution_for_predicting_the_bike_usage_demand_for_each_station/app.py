from flask import Flask, render_template,request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('bike_demand_model_v2.pkl')

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        data = {
            'season': int(request.form['season']),
            'yr': int(request.form['yr']),
            'mnth': int(request.form['mnth']),
            'hr': int(request.form['hr']),
            'holiday': int(request.form['holiday']),
            'weekday': int(request.form['weekday']),
            'workingday': int(request.form['workingday']),
            'weathersit': int(request.form['weathersit']),
            'temp': float(request.form['temp']),
            'hum': float(request.form['hum']),
            'windspeed': float(request.form['windspeed']),
        }

        # Convert the input to a DataFrame
        input_df = pd.DataFrame([data])

        # Make the prediction
        prediction = model.predict(input_df)

        # Return the result to the template
        return render_template('index.html', prediction=int(prediction[0]))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # In production, disable debug mode and set host to 0.0.0.0
    app.run(debug=False, host='0.0.0.0', port=5000)
