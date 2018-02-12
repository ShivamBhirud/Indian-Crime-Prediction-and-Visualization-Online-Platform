from flask import Flask,render_template, request, flash, url_for,jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation,preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

app = Flask("__main__")

@app.route('/')
def Index():
	return render_template("home.html")

@app.route("/home.html")
def Home():
	return render_template("home.html")

@app.route('/pred.html')
def Services1():
	return render_template("pred.html")

@app.route('/vis.html')
def Services2():
	return render_template("vis.html")

@app.route('/stats.html')
def Services3():
	return render_template("stats.html")

@app.route('/women.html',methods = ['POST'])
def women():

	#Crime_type = request.form.get("type")
	year = request.form.get("Predict_Year")
	df = pd.read_csv("static/CAW.csv")

	X = df.iloc[:,:-1].values
	y = df.iloc[:,12].values
	X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
	regressor = LinearRegression()
	regressor.fit(X_train,y_train)
	joblib.dump(regressor, "linear_regression_model.pkl")
	#joblib.dump(X_test, "training_data.pkl")
	#joblib.dump(y_test, "training_labels.pkl")
	lr = joblib.load("./linear_regression_model.pkl")
	#training_set = joblib.load("./training_data.pkl")
	#labels = joblib.load("./training_labels.pkl")
	accuracy = lr.score(X_test, y_test)*100
	tst = [[2015,38740,61511,8700,88235,10735,132877,15,1870,11230,75,0], [2004,18233,15578,7026,34567,10001,58121,89,5748,3592,1378,0]]
	prediction = lr.predict(tst)
	
	return render_template('women.html',data = [year,accuracy,X_test,prediction])
	#return jsonify({"score": accuracy, "Predicted values are: ": prediction.tolist(), "testing set is: ":X_test.tolist(), 
	#	"coefficients": lr.coef_.tolist(), "intercepts": lr.intercept_})
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)

