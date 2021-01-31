from flask import Flask, render_template,request
import joblib
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route("/tahmin", methods = ["POST"])

def tahmin_et():
    model = open("Earthquake_model.pkl","rb")
    clf = joblib.load(model)
    if request.method == "POST":
        koordinat1 = request.form["koordinat1"]
        koordinat2 = request.form["koordinat2"]
        data = [np.array([koordinat1,koordinat2])]
        prediction = clf.predict(data)
    return render_template("result.html", data=prediction[0])

if __name__ == '__main__':
    app.run()