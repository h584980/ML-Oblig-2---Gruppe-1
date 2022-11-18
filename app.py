import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# lag app
app = Flask(__name__)

# last inn modell
model = pickle.load(open("model.pkl", "rb"))

#
@app.route("/")
def Home():
	return render_template("index.html")


@app.route("/predict", methods = ["POST"])
def predict():
	# float_features = [float(x) for x in request.form.values()]

	# features = [np.array(float_features)]

	passengerId = int(request.form.get("PassengerId"))
	
	pclass = int(request.form.get("Pclass"))

	sex = int(request.form.get("Sex"))
	
	age = float(request.form.get("Age"))
	
	sibSp = int(request.form.get("SibSp"))
	
	parch = int(request.form.get("Parch"))
	
	fare = float(request.form.get("Fare"))
	
	embarked = int(request.form.get("Embarked"))
	
	title = int(request.form.get("Title"))

	lst = [[passengerId, pclass, sex, age, sibSp, parch, fare, embarked, title]]

	features = pd.DataFrame(lst, columns=('PassangerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title'))

	prediction = model.predict(features)

	if int(prediction) == 1:
		result = 'Hadde overlevd forliset!'
	else:	
		result = 'Hadde druknet under forliset!'          
	
	return render_template("index.html", prediction = result)

if __name__ == "__main__":
	app.run(debug=True)

