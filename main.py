from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pickle
import io
import base64


def create_figure(pos, neg):
    fig = Figure(facecolor=(0.2, 0.2, 0.2))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_facecolor((0.2, 0.2, 0.2))
    axis.bar(["Probability of bad quality", "Probability of good quality"],
             [pos, neg], width=0.5, color=(0.5, 0, 0))
    return fig


app = Flask(__name__)

pickle_in = open("RfClassifier.pk1", 'rb')
rf_classifier = pickle.load(pickle_in)

pickle_scaler = open("scaler.pk1", 'rb')
scaler = pickle.load(pickle_scaler)


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/analyzeWine', methods=["POST", "GET"])
def analyze_wine():
    if request.method == "POST":
        fixed_acidity = request.form["fa"]
        volatile_acidity = request.form["va"]
        citric_acid = request.form["ca"]
        residual_sugar = request.form["rs"]
        chlorides = request.form["ch"]
        free_sulfur_dioxide = request.form["fsd"]
        total_sulfur_dioxide = request.form["tsd"]
        ph = request.form["pH"]
        sulphates = request.form["su"]
        alcohol = request.form["alc"]
        vec = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, ph, sulphates,
                        alcohol]).reshape(-1, 10)
        vec = scaler.transform(vec)
        result = rf_classifier.predict_proba(vec)
        return redirect(url_for("results", prob_yes=result[0][0], prob_no=result[0][1]))
    else:
        return render_template('AnalyzeWine.html')


@app.route('/results/<float:prob_yes>/<float:prob_no>')
def results(prob_yes, prob_no):
    pos = prob_yes
    neg = prob_no
    fig = create_figure(pos, neg)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(output.getvalue()).decode('utf8')

    return render_template("results.html", image=pngImageB64String)


@app.route('/howItWorks')
def how_it_works():
    return "under construction"


if __name__ == '__main__':
    app.run()
