from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
knn_final=pickle.load(open('major_project_cancer_prediction-main/pickle/knn_final.pkl','rb'))


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('cancer.html')

@app.route('/cancer',methods=['POST'])
def heart():
    if request.method=='POST':
        Age=int(request.form['Age'])
        Gender=int(request.form['Gender'])
        AirPollution=int(request.form['AirPollution'])
        Alcoholuse=int(request.form['Alcoholuse'])
        DustAllergy=int(request.form['DustAllergy'])
        OccuPationalHazards=int(request.form['OccuPationalHazards'])
        GeneticRisk=int(request.form["GeneticRisk"])
        BalancedDiet=int(request.form['BalancedDiet'])
        Smoking=int(request.form['Smoking'])
        ChestPain=int(request.form['ChestPain'])
        CoughingofBlood=int(request.form['CoughingofBlood'])
        Fatigue=int(request.form['Fatigue'])
        WeightLoss=int(request.form['WeightLoss'])
        ShortnessBreath=int(request.form['ShortnessofBreath'])
        Wheezing=int(request.form['Wheezing'])
        FrequentCold=int(request.form['FrequentCold'])
        DryCough=int(request.form['DryCough'])
        outcome=knn_final.predict([[Age,Gender,AirPollution,Alcoholuse,DustAllergy,OccuPationalHazards,GeneticRisk,BalancedDiet,Smoking,ChestPain,CoughingofBlood,Fatigue,WeightLoss,ShortnessBreath,Wheezing,FrequentCold,DryCough]])
        if outcome[0]==1:
            value='Chances Of Cancer'
            return render_template('single_prediction.html',result=value)
        else:
            value='No Chances of Cancer'
            return render_template('single_prediction.html',result=value)





if __name__=="__main__":
    app.run(host="0.0.0.0")
