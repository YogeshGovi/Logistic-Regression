import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = int(prediction)
    
    print("Prediction from Web Page is : ", output)
    
    if output ==1:
        data="Will Graduate. Congrats .. "
    elif output == 0:
        data="Will Dropout. Take Care .."
    
        

    return render_template('index.html', prediction_text=data)
    


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)