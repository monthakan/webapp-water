import numpy as np
import pickle
from flask import Flask, render_template, request


app = Flask(__name__)



# Load the trained model
model = pickle.load(open('model/model.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data1 = request.form['rain']
        data2 = request.form['humid']
        data3 = request.form['storage']
        data4 = request.form['used']
        data5 = request.form['flow']
    
        arr =np.array([[data1, data2, data3, data4, data5]])
        pred = model.predict(arr)

        return render_template('home.html', data=pred)


if __name__ == '__main__':
    app.run()
    app.debug(True)