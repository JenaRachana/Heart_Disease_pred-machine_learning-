from flask import Flask,request,render_template
import numpy as np
import pickle

app=Flask(__name__)

# Load the artifacts

model=pickle.load(open('knn_model.pkl','rb'))
scale_model=pickle.load(open('scaler_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # collect the input
    data=np.array([
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]
    )

    # Scale and predict
    # data_scaled = scaler.transform([data])
    prediction = model.predict([data])

    result = 'heart disease is detected' if prediction[0]==1 else 'no heart disease'
    return render_template('result.html',prediction=result)

if __name__ == "__main__":
    app.run(debug=True)








