
from flask import Flask,request,render_template,jsonify
app=Flask(__name__)
import pickle
import numpy as np
import pandas as pd
# df = pd.read_csv("laptop_clean (1).csv")
model=pickle.load(open('predict.pkl','rb'))



@app.route('/')
def index():
    return render_template("index.html")

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=np.array(list(data.values())).reshape(1,-1)
#     output=model.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    print(final_input)
    # query = final_input.reshape(1, -1)
    # prediction = str(float(np.exp(model.predict(query)[0])))
    output=model.predict(final_input)[0]
    return render_template("index.html",prediction_text="laptop price prediction is {}".format(output))




if __name__=="__main__":
    app.run(debug=True)