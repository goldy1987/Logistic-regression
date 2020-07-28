# Imports
from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__)
cross_origin()

@app.route('/', methods = ['GET'])
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        try:
            #reading user input
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            #loading model
            filename = 'modelForPrediction.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            scaler = pickle.load(open('standardScalar.sav','rb'))

            #Predicting result
            prediction =loaded_model.predict(scaler.transform([[Pregnancies,Glucose,BloodPressure,
                                                                SkinThickness,
                                                                Insulin,BMI,DiabetesPedigreeFunction,
                                                                Age ]]))
            print('Prediction is',prediction)

            if prediction == 0:
                return render_template('0.html')
            else:
                return render_template('1.html')
        except Exception as e:
            print('Exception message is: ', e)
            return 'Something is Wrong'
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)