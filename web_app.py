import numpy as np
from flask import Flask, request, render_template
import pickle




app = Flask(__name__)


f1=open('cv_itv','rb')
cv=pickle.load(f1)
f1.close()




f2=open('model_itv','rb')
model=pickle.load(f2)
f2.close()






@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''


    if request.method == 'POST':
        text = request.form['Review']
        data = [text]
        data_cv = cv.transform(data).toarray()
        prediction = model.predict(data_cv)
        prediction=prediction[0]
        print(prediction)
        if ("not" in text) or ("no" in text) or ("n't" in text):
            prediction= abs(prediction - 1)
    if prediction==1:
        return render_template('index.html', prediction_text='The review is Positive')
    else:
        return render_template('index.html', prediction_text='The review is Negative.')






if __name__ == "__main__":
    app.run(debug=True)
