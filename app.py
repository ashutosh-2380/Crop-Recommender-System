from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the Crop Recommender Model
model = joblib.load("Crop_Recommendation_Model")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        n = float(request.form.get("n"))
        p = float(request.form.get("p"))
        k = float(request.form.get("k"))
        temperature = float(request.form.get("temperature"))
        humidity = float(request.form.get("humidity"))
        ph = float(request.form.get("ph"))
        rainfall = float(request.form.get("rainfall"))

        # Make the crop prediction
        input_data = [[n, p, k, temperature, humidity, ph, rainfall]]
        predicted_crop = model.predict(input_data)[0]

        return render_template("index.html", prediction=predicted_crop)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
