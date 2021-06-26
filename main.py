from flask import Flask, render_template, request, redirect,url_for
import numpy as np
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__)

pickle_in = open("RfClassifier.pk1", 'rb')
rf_classifier = pickle.load(pickle_in)


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/analyzeWine', methods =["POST","GET"])
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
        vec =  np.array([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,
                         chlorides,free_sulfur_dioxide, total_sulfur_dioxide, ph,sulphates,
                         alcohol]).reshape(-1,10)
        result = rf_classifier.predict_proba(vec)
        return redirect(url_for("results", prob_yes=result[0][0], prob_no=result[0][1]))
    else:
        return render_template('AnalyzeWine.html')
@app.route('/results/<float:prob_yes>/<float:prob_no>')
def results(prob_yes,prob_no):
    return 'workiing on it'

if __name__ == '__main__':
    app.run()
