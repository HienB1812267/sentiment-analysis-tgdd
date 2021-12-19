from flask import Flask, render_template, request, url_for, redirect
from flask_cors import CORS, cross_origin
import load_model as MODEL
import process as PREPROCESS
import json
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# Use a service account
cred = credentials.Certificate('./sentiment-prediction-tgdd-firebase-adminsdk-olmlq-105780f3e4.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

#Init
app = Flask(__name__)

#Apply flask cors
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#load model bert

@app.route('/')
@cross_origin(origin='*')
def index_process():
    return render_template("index.html")

# @app.route('/dataset', methods=["POST", "GET"])
# @cross_origin(origin='*')
# def view_dataset():
#     with open("./data/data_TGDD.json", encoding='utf-8') as file:
#         data = json.loads(file.read())
#     return render_template("dataset.html", data=data)

@app.route('/predict', methods=["POST", "GET"])
@cross_origin(origin='*')
def predict_star():
    comment = request.form['input_comment']
    username = request.form['input_name']
    if len(username) == 0:
        username = "Anonymous"
    #preprocess data
    df = pd.DataFrame()
    df['comment'] = [comment]
    df['comment'] = df['comment'].apply(PREPROCESS.preProcessing)
    cmt = df['comment']
    #load vectorizes
    count_vector, tf_idf_vector = MODEL.load_vectorizes()
    input_cv = count_vector.transform(cmt)
    input_tf = tf_idf_vector.transform(cmt)

    output = []
    #Ta sẽ dùng mô hình logistic với tf-idf để dự đoán chính. Các mô hình còn lại sẽ được lưu vào database
    # để phục vụ cho chức năng xem chi tiết nếu người dùng muốn xem chi tiết dự đoán của các mô hình#

    #load bayes
    bayes_cv, bayes_tf = MODEL.load_bayes_model()
    #Lưu lại dự đoán
    output.append(bayes_cv.predict(input_cv)[0])
    output.append(bayes_tf.predict(input_tf)[0])

    #load svm
    svm_cv, svm_tf = MODEL.load_svm_model()
    #Lưu lại dự đoán
    output.append(svm_cv.predict(input_cv)[0])
    output.append(svm_tf.predict(input_tf)[0])

    #load svm
    logistic_cv, logistic_tf = MODEL.load_svm_model()
    #Lưu lại dự đoán
    output.append(logistic_cv.predict(input_cv)[0])
    output.append(logistic_tf.predict(input_tf)[0])
    #connect db
    doc_ref = db.collection(u'data').document()
    doc_ref.set({
        u'comment': comment,
        u'username': username,
        u'bayes_cv': int(output[0]),
        u'bayes_tf': int(output[1]),
        u'svm_cv': int(output[2]),
        u'svm_tf': int(output[3]),
        u'logis_cv': int(output[4]),
        u'logis_tf': int(output[5]),
    })
    return redirect("/")


#Start backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')
