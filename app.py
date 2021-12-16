from flask import Flask, render_template, request, url_for, redirect
from flask_cors import CORS, cross_origin
from flask_mysqldb import MySQL
import load_model as MODEL
import process as PREPROCESS
import pandas as pd

#Init
app = Flask(__name__)

#database connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask_app_tgdd'
# app.config['MYSQL_CHARSET'] = 'utf-8'

mysql = MySQL(app)

#Apply flask cors
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

g_content_list = []

#load model bert

@app.route('/')
@cross_origin(origin='*')
def index_process():
    #connect db
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM `data`")
    #get data
    data = cursor.fetchall()
    #close
    mysql.connection.commit()
    cursor.close()
    #load data
    g_content_list.clear()
    for line in data:
        g_content_list.append(line)
    return render_template("index.html", g_content_list=g_content_list)

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
    comment = df['comment']
    #load vectorizes
    count_vector, tf_idf_vector = MODEL.load_vectorizes()
    input_cv = count_vector.transform(comment)
    input_tf = tf_idf_vector.transform(comment)

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
    cursor = mysql.connection.cursor()
    sql = "INSERT INTO data(comment, username, bayes_cv, bayes_tf, svm_cv, svm_tf, logis_cv, logis_tf) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)"
    val = (request.form['input_comment'], username, output[0], output[1], output[2], output[3], output[4], output[5])
    cursor.execute(sql, val) 
    #close
    mysql.connection.commit()
    cursor.close()

    return redirect("/")


#Start backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')


