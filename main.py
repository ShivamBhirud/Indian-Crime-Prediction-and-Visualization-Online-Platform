from flask import Flask,render_template, request, flash, url_for,jsonify
import pandas as pd
import numpy as np
from flask import json
from sklearn import cross_validation,preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from plotly.offline import init_notebook_mode, iplot



app = Flask(__name__)

@app.route('/')
def Index():
	return render_template("home.html")

@app.route("/home.html")
def Home():
	return render_template("home.html")

@app.route('/pred.html')
def pred():
	return render_template("pred.html")

@app.route('/vis.html')
def viz():
	return render_template("vis.html")

@app.route('/womenViz.html')
def womenViz():	
	return render_template('womenViz.html')

@app.route('/stats.html')
def Services3():
	return render_template("stats.html")

@app.route('/women.html',methods = ['POST'])
def women():

	year = request.form.get("Predict_Year")
	C_type = request.form.get("C_Type")
	state = request.form.get("state")

	df = pd.read_csv("static/StateCAWPred2001_16.csv", header=None)

	data1 = df.loc[df[0]==state].values
	for x in data1:
		if x[1] == C_type:
			test = x
			break


	l = len(df.columns)
	trendChangingYear = 2
	accuracy_max = 0.65

	xTrain = np.array([2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016])
	yTrain = test[2:18]

	X = df.iloc[0,2:l].values
	y = test[2:]
	regressor = LinearRegression()
	regressor.fit(X.reshape(-1,1),y)
	accuracy = regressor.score(X.reshape(-1,1),y)
	print accuracy
	accuracy_max = 0.65
	if(accuracy < 0.65):
		for a in range(3,l-4):

			X = df.iloc[0,a:l].values
			y = test[a:]
			regressor = LinearRegression()
			regressor.fit(X.reshape(-1,1),y)
			accuracy = regressor.score(X.reshape(-1,1),y)
			if (accuracy > accuracy_max):
				accuracy_max = accuracy
				print accuracy_max
				trendChangingYear = a
	print trendChangingYear
	print test[trendChangingYear]
	print xTrain[trendChangingYear-2]
	year = int(year)
	y = test[2:]
	b = []
	if accuracy_max < 0.65:
		for k in range(2001,2017):
			a = str(k)
			b = np.append(b,a)
		y = list(y)
		yearLable = list(b)
		year = 2016
		msg = "Data is not Suitable for prediction"
	else:

		for j in range(2017,year+1):
			prediction = regressor.predict(j)
			if(prediction < 0):
				prediction = 0
			y = np.append(y,prediction)
		y = np.append(y,0)

		for k in range(2001,year+1):
			a = str(k)
			b = np.append(b,a)
		y = list(y)
		yearLable = list(b)
		msg = ""
	

	return render_template('women.html',data = [accuracy,yTrain,xTrain,state,year,data1,X,y,test,l],msg = msg, state=state, year=year, C_type=C_type,pred_data = y,years = yearLable)

@app.route('/children.html',methods = ['POST'])
def children():

	year = request.form.get("Predict_Year")
	C_type = request.form.get("C_Type")
	state = request.form.get("state")

	df = pd.read_csv("static/Statewise Cases Reported of Crimes Committed Against Children 1994-2016.csv", header=None)

	data1 = df.loc[df[0]==state].values
	for x in data1:
		if x[1] == C_type:
			test = x
			break


	l = len(df.columns)

	trendChangingYear = 2
	accuracy_max = 0.65

	xTrain = np.array([1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016])
	yTrain = test[2:25]

	X = df.iloc[0,2:l].values
	y = test[2:]
	regressor = LinearRegression()
	regressor.fit(X.reshape(-1,1),y)
	accuracy = regressor.score(X.reshape(-1,1),y)
	print accuracy
	accuracy_max = 0.65
	if(accuracy < 0.65):
		for a in range(3,l-4):

			X = df.iloc[0,a:l].values
			y = test[a:]
			regressor = LinearRegression()
			regressor.fit(X.reshape(-1,1),y)
			accuracy = regressor.score(X.reshape(-1,1),y)
			if (accuracy > accuracy_max):
				accuracy_max = accuracy
				print accuracy_max
				trendChangingYear = a
	print trendChangingYear
	print test[trendChangingYear]
	print xTrain[trendChangingYear-2]
	yTrain = test[trendChangingYear:]
	xTrain = xTrain[trendChangingYear-2:]
	regressor.fit(xTrain.reshape(-1,1),yTrain)
	accuracy = regressor.score(xTrain.reshape(-1,1),yTrain)

	year = int(year)
	#year = np.array(year)
	y = test[2:]
	for j in range(2017,year+1):
		prediction = regressor.predict(j)
		if(prediction < 0):
			prediction = 0
		y = np.append(y,prediction)
	y = np.append(y,0)
	b = []
	for k in range(1994,year+1):
		a = str(k)
		b = np.append(b,a)
	y = list(y)
	yearLable = list(b)

	

	return render_template('children.html',data = [accuracy,yTrain,xTrain,state,year,data1,X,y,test,l],state=state, year=year, C_type=C_type,pred_data = y,years = yearLable)

@app.route('/ipc.html',methods = ['POST'])
def ipc():

	year = request.form.get("Predict_Year")
	C_type = request.form.get("C_Type")
	state = request.form.get("state")

	df = pd.read_csv("static/StateIPCPred2001_16.csv", header=None)

	data1 = df.loc[df[0]==state].values
	for x in data1:
		if x[1] == C_type:
			test = x
			break


	l = len(df.columns)

	trendChangingYear = 2
	accuracy_max = 0.65

	xTrain = np.array([2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016])
	yTrain = test[2:18]

	X = df.iloc[0,2:l].values
	y = test[2:]
	regressor = LinearRegression()
	regressor.fit(X.reshape(-1,1),y)
	accuracy = regressor.score(X.reshape(-1,1),y)
	print accuracy
	accuracy_max = 0.65
	if(accuracy < 0.65):
		for a in range(3,l-8):

			X = df.iloc[0,a:l].values
			y = test[a:]
			regressor = LinearRegression()
			regressor.fit(X.reshape(-1,1),y)
			accuracy = regressor.score(X.reshape(-1,1),y)
			if (accuracy > accuracy_max):
				accuracy_max = accuracy
				print accuracy_max
				trendChangingYear = a
	print trendChangingYear
	print test[trendChangingYear]
	print xTrain[trendChangingYear-2]
	yTrain = test[trendChangingYear:]
	xTrain = xTrain[trendChangingYear-2:]
	regressor.fit(xTrain.reshape(-1,1),yTrain)
	accuracy = regressor.score(xTrain.reshape(-1,1),yTrain)

	year = int(year)
	#year = np.array(year)
	y = test[2:]
	for j in range(2017,year+1):
		prediction = regressor.predict(j)
		if(prediction < 0):
			prediction = 0
		y = np.append(y,prediction)
	y = np.append(y,0)
	b = []
	for k in range(2001,year+1):
		a = str(k)
		b = np.append(b,a)
	y = list(y)
	yearLable = list(b)

	

	return render_template('ipc.html',data = [accuracy,yTrain,xTrain,state,year,data1,X,y,test,l],state=state, year=year, C_type=C_type,pred_data = y,years = yearLable)


@app.route('/sll.html',methods = ['POST'])
def sll():

	year = request.form.get("Predict_Year")
	C_type = request.form.get("C_Type")
	state = request.form.get("state")

	df = pd.read_csv("static/StateSLLPred2001_16.csv", header=None)

	data1 = df.loc[df[0]==state].values
	for x in data1:
		if x[1] == C_type:
			test = x
			break


	l = len(df.columns)

	trendChangingYear = 2
	accuracy_max = 0.65

	xTrain = np.array([2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016])
	yTrain = test[2:18]

	X = df.iloc[0,2:l].values
	y = test[2:]
	regressor = LinearRegression()
	regressor.fit(X.reshape(-1,1),y)
	accuracy = regressor.score(X.reshape(-1,1),y)
	print accuracy
	accuracy_max = 0.65
	if(accuracy < 0.65):
		for a in range(3,l-4):

			X = df.iloc[0,a:l].values
			y = test[a:]
			regressor = LinearRegression()
			regressor.fit(X.reshape(-1,1),y)
			accuracy = regressor.score(X.reshape(-1,1),y)
			if (accuracy > accuracy_max):
				accuracy_max = accuracy
				print accuracy_max
				trendChangingYear = a
	print trendChangingYear
	print test[trendChangingYear]
	print xTrain[trendChangingYear-2]
	yTrain = test[trendChangingYear:]
	xTrain = xTrain[trendChangingYear-2:]
	regressor.fit(xTrain.reshape(-1,1),yTrain)
	accuracy = regressor.score(xTrain.reshape(-1,1),yTrain)

	year = int(year)
	#year = np.array(year)
	y = test[2:]
	for j in range(2017,year+1):
		prediction = regressor.predict(j)
		if(prediction < 0):
			prediction = 0
		y = np.append(y,prediction)
	y = np.append(y,0)
	b = []
	for k in range(2001,year+1):
		a = str(k)
		b = np.append(b,a)
	y = list(y)
	yearLable = list(b)

	

	return render_template('sll.html',data = [accuracy,yTrain,xTrain,state,year,data1,X,y,test,l],state=state, year=year, C_type=C_type,pred_data = y,years = yearLable)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)




	#joblib.dump(regressor, "linear_regression_model.pkl")
	#joblib.dump(X_test, "training_data.pkl")
	#joblib.dump(y_test, "training_labels.pkl")
	#lr = joblib.load("./linear_regression_model.pkl")
	#training_set = joblib.load("./training_data.pkl")
	#labels = joblib.load("./training_labels.pkl")
	#prediction = lr.predict(tst)
