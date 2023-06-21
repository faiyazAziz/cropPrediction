from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('cropModel.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "This is an API used to predict crop"

@app.route('/predict',methods=['POST'])
def predict():
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    input_data = np.array([[temperature,humidity,ph,rainfall]])
    crop = model.predict(input_data)[0]

    return jsonify({'crop':crop})


if __name__ == '__main__':
    app.run(debug=True)