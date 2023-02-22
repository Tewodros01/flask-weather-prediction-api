from flask import Flask,request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('weather_model.pkl','rb'))


@app.route('/predict',methods=['POST','GET'])
def predict():
    data = request.json
    precip = (data.get('precip'))
    tempMax = (data.get('tempMax'))
    tempMin = (data.get('tempMin'))
    final=[np.array([precip,tempMax,tempMin])]
    prediction=model.predict(final)
    return jsonify(prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
