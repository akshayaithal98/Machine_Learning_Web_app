from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('regression.pkl','rb'))
app=Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_charges():
    smoker=request.form.get('smoker')
    bmi=float(request.form.get('bmi'))
    age=int(request.form.get('age'))

    result=model.predict(np.array([smoker,bmi,age]).reshape(1,3))
    result=result[0]
    
    return render_template('index.html',result=result)
if __name__=='__main__':
    app.run(debug=True)

