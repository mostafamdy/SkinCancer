import os
from datetime import datetime

import cv2
import numpy as np
import random as rn
import tensorflow as tf
from keras import backend as K
import pickle
from flask import Flask, request, url_for, render_template
import json
import base64
import pandas as pd
import sqlite3
import smtplib
from email.message import EmailMessage

# login_con = sqlite3.connect('userDB.db',check_same_thread=False)
# login_cur = login_con.cursor()

# forget_pass_con=sqlite3.connect('forgetPass.db',check_same_thread=False)
# forget_cur = forget_pass_con.cursor()


def init():

    models_path = ''
    global img_clf
    global meta_clf
    global final_model

    img_clf = pickle.load(open(models_path + 'img_clf.sk', 'rb'))
    meta_clf = pickle.load(open(models_path + 'meta_clf.sk', 'rb'))
    final_model = tf.keras.models.load_model(models_path+'final_model.h5')

    seed = 101
    num_cores = 1
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                                            inter_op_parallelism_threads=num_cores,
                                            allow_soft_placement=True,
                                            device_count={'CPU': 1, 'GPU': 0})
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)

    global img_feature_model

    img_feature_model = tf.keras.applications.MobileNetV2(input_shape=(380, 380, 3), include_top=False,
                                                          weights='imagenet')


def predict_meta(data):
    '''
    if smoke =="true":
        smoke=2
    else if smoke=="false" :
        smoke=1
    else:
        smoke=0
    if drink =="true":
        drink=2
    else if drink=="false" :
        drink=1
    else:
        drink=0

    if pesticide =="true":
        pesticide=2
    else if pesticide=="false" :
        pesticide=1
    else:
        pesticide=0

    if gender=='FEMALE' :
        gender=2
    else if gender=='MALE':
        gender=1
    else:
        gender=0

    if skin_cancer_history=="true":
        skin_cancer_history=2
    else if skin_cancer_history=="false":
        skin_cancer_history=1
    else:
        skin_cancer_history=0

    if cancer_history=="true" :
        cancer_history=2
    else if cancer_history=="false":
        cancer_history=1
    else:
        cancer_history=0

    if has_piped_water=="true":
        has_piped_water=2
    else if has_piped_water=="false":
        has_piped_water=1
    else:
        has_piped_water=0

    if has_sewage_system=="true" :
        has_sewage_system=2
    else if has_sewage_system=="false":
        has_sewage_system=1
    else:
        has_sewage_system=0

    if biopsed=="true" :
        biopsed=2
    else if biopsed=="false":
        biopsed=1
    else:
        biopsed=0

    if itch=="true":
        itch=2
    else if itch=="false":
        itch=1
    else:
        itch=0

    if grew=="true":
        grew=2
    else if grew=="false":
        grew=1
    else:
        grew=0


    if hurt=="true":
        hurt=2
    else if hurt=="false":
        hurt=1
    else:
        hurt=0

    if changed=="true" :
        changed=2
    else if changed=="false":
        changed=1
    else:
        changed=0

    if bleed=="true" :
        bleed=2
    else if bleed=="false":
        bleed=1
    else:
        bleed=0

    if elevation=="true" :
        elevation=2
    else if elevation=="false":
        elevation=1
    else:
        elevation=0

    # digits
    fitspatrick=int(fitspatrick)
    diameter_1=int(diameter_1)
    diameter_2=int(diameter_2)


    '''

    # smoke = str(data['smoke'])
    # drink = str(data['drink'])
    # background_father = str(data['background_father'])
    # background_mother = str(data['background_mother'])
    # age = str(data['age'])
    # pesticide = str(data['pesticide'])
    # gender = str(data['gender'])
    # skin_cancer_history = str(data['skin_cancer_history'])
    # cancer_history = str(data['cancer_history'])
    # has_piped_water = str(data['has_piped_water'])
    # has_sewage_system = str(data['has_sewage_system'])
    # fitspatrick = str(data['fitspatrick'])
    # region = str(data['region'])
    # diameter_1 = str(data['diameter_1'])
    # diameter_2 = str(data['diameter_2'])
    # grew = str(data['grew'])
    # hurt = str(data['hurt'])
    # changed = str(data['changed'])
    # bleed = str(data['bleed'])
    # elevation = str(data['elevation'])
    # biopsed = str(data['biopsed'])

    meta=pd.DataFrame(data, index=[0])
    # 15
    tr="true"
    fal="false"
    meta['smoke'] = meta['smoke'].map({tr: 2, fal: 1, "": 0})
    meta['drink'] = meta['drink'].map({tr: 2, fal: 1, "": 0})
    meta['pesticide'] = meta['pesticide'].map({tr: 2, fal: 1, "": 0})
    meta['gender'] = meta['gender'].map({'FEMALE': 2, 'MALE': 1, "": 0})
    meta['skin_cancer_history'] = meta['skin_cancer_history'].map({tr: 2, fal: 1, "": 0})
    meta['cancer_history'] = meta['cancer_history'].map({tr: 2, fal: 1, "": 0})
    meta['has_piped_water'] = meta['has_piped_water'].map({tr: 2, fal: 1, "": 0})
    meta['has_sewage_system'] = meta['has_sewage_system'].map({tr: 2, fal: 1, "": 0})
    meta['biopsed'] = meta['biopsed'].map({tr: 2, fal: 1, "": 0})
    meta['itch'] = meta['itch'].map({tr: 2, fal: 1, '': 0})
    meta['grew'] = meta['grew'].map({tr: 2, fal: 1, '': 0})
    meta['hurt'] = meta['hurt'].map({tr: 2, fal: 1, '': 0})
    meta['changed'] = meta['changed'].map({tr: 2,fal: 1, '': 0})
    meta['bleed'] = meta['bleed'].map({tr: 2, fal: 1, '': 0})
    meta['elevation'] = meta['elevation'].map({tr: 2,fal: 1, '': 0})

    # digits
    if meta['fitspatrick'][0] == "":
        meta['fitspatrick'][0] = -1

    if meta['age'][0] == "":
        meta['age'][0] = -1

    if meta['diameter_1'][0] == "":
        meta['diameter_1'][0] = -1

    if meta['diameter_2'][0] == "":
        meta['diameter_2'][0] = -1

    if meta['background_father'][0] == "":
        meta['background_father'][0] = "None"

    if meta['background_mother'][0] == "":
        meta['background_mother'][0] = "None"

    if meta['region'][0] == "":
        meta['region'][0] = "ABDOMEN"
    # 3
    meta['fitspatrick'][0] = int(meta['fitspatrick'][0])
    meta['diameter_1'][0] = int(meta['diameter_1'][0])
    meta['diameter_2'][0] = int(meta['diameter_2'][0])
    meta['age'][0] = int(meta['age'][0])
    # 3
    regions = ['ABDOMEN', 'ARM', 'BACK', 'CHEST', 'EAR', 'FACE', 'FOOT', 'FOREARM', 'HAND', 'LIP', 'NECK', 'NOSE', 'SCALP', 'THIGH']
    back_father = ['AUSTRIA', 'BRASIL', 'BRAZIL', 'CZECH', 'GERMANY', 'ISRAEL', 'ITALY', 'NETHERLANDS', 'None', 'POLAND', 'POMERANIA', 'PORTUGAL', 'SPAIN', 'UNK']
    back_mother = ['BRAZIL', 'FRANCE', 'GERMANY', 'ITALY', 'NETHERLANDS', 'NORWAY', 'None', 'POLAND', 'POMERANIA', 'PORTUGAL', 'SPAIN', 'UNK']

    meta['region'][0]=regions.index(meta['region'][0])
    meta['background_father'][0]=back_father.index(meta['background_father'][0])
    meta['background_mother'][0]=back_mother.index(meta['background_mother'][0])


    # meta['background_father'] = meta['background_father'].fillna('None')
    # meta['background_mother'] = meta['background_mother'].fillna('None')
    meta1=meta.drop(['imagebase64', 'background_father','email'], axis=1)
    predctions=meta_clf.predict_proba(meta1)
    return (predctions,meta)


def ex_feature_img(path):
  image = tf.keras.utils.load_img(path,target_size=(380,380))
  input_arr = tf.keras.preprocessing.image.img_to_array(image)
  input_arr = cv2.cvtColor(input_arr,cv2.COLOR_BGR2RGB)
  input_arr=np.array([input_arr])
  input_arr=input_arr/255
  features=img_feature_model.predict(input_arr)
  nx, ny,_ = features[0].shape
  img_features = features[0].reshape(nx*ny*_)
  return img_features


def predict_img(path):
    img_features = ex_feature_img(path)
    return img_clf.predict_proba([img_features])


def save_image(img64):

    image = base64.b64decode(img64)

    now = datetime.now()
    imageName = now.strftime("%d%m%y%H%M%S")
    path = os.getcwd() + "\\images_from_mobile\\" + imageName + ".jpg"
    file = open(path, 'wb')
    file.write(image)
    file.close()
    return path


app = Flask(__name__)

init()

diag=['ACK','BCC','MEL','NEV','SCC','SEK']

def save_result(meta,imgPath,imgPreds,metaPreds,final_result):
    meta['img']=imgPath
    meta['meta_predictions']=metaPreds
    meta['img_predictions']=imgPreds
    meta['final_result']=final_result
    meta=meta.drop(['imagebase64', 'background_father',], axis=1)
    # if os.path.exists()
    meta.to_csv("images_from_mobile/meta.csv",mode='a')


@app.route('/', methods=['POST'])
def from_mobile():
    data = request.get_json()
    meta_predictions,meta = predict_meta(data)
    print("meta preds "+str(meta_predictions))
    print("meta ", meta)
    img64 = str(data['imagebase64'])
    path = save_image(img64)
    # predict
    img_predictions = predict_img(path)
    preds = final_model.predict([meta_predictions, img_predictions])
    print(preds)
    save_result(meta,path,str(img_predictions),str(meta_predictions),str(preds))
    maxi=np.argmax(preds)

    if preds[0][maxi]<.6:
        return "-1"
    else:
        return diag[maxi]


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = str(data['email'])
    password = str(data['password'])
    with sqlite3.connect('userDB.db') as con:
        login_cur = con.cursor()
        record = login_cur.execute('SELECT * FROM users WHERE email=:em;', {"em": email}).fetchall()
        # con.close()

    # check if email exit
    if len(record) == 0:
        return "wrong email"
    else:
        # check if pass wrong
        if record[0][1] != password:
            return "wrong pass"
        else:
            return ""


@app.route('/createAccount', methods=['POST'])
def createAccount():
    data = request.get_json()
    email = str(data['email'])
    password = str(data['password'])
    name = str(data['name'])
    with sqlite3.connect('userDB.db') as con:
        login_cur = con.cursor()
        # check if email in db or not
        if len(login_cur.execute('SELECT * FROM users WHERE email=:em;', {"em": email}).fetchall()) != 0:
            # con.close()
            return "this email used before"

        else:
            # if not exit insert it
            login_cur.execute("INSERT INTO users VALUES (?, ?, ?)", (email, password, name))
            con.commit()
            # con.close()
            # insert it in database
            return "its ok"


@app.route('/forgetPass', methods=['POST'])
def forget_pass():
    data = request.get_json()
    email = str(data['email'])
    code = str(rn.randint(1000,9999))
    with sqlite3.connect('forgetPass.db') as con:
        forget_cur = con.cursor()
        forget_cur.execute("INSERT INTO forgetPass VALUES (?, ?)", (email, code))
        con.commit()
        msg = EmailMessage()
        msg['Subject'] = ''
        msg['From'] = 'FCAIH Skin cancer team'
        msg['To'] = email
        msg.set_content(code)
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login('skincancerfcaih@gmail.com','hfkkkpgkbprsmihr')
        server.send_message(msg)
        server.quit()
        # con.close()
    return "it's ok "


@app.route('/passCode', methods=['POST'])
def passCode():
    data = request.get_json()
    email = str(data['email'])
    code = str(data['code'])
    with sqlite3.connect('forgetPass.db') as con:
        forget_cur = con.cursor()
        db = forget_cur.execute('SELECT * FROM forgetPass WHERE email=:em;', {"em": email}).fetchall()

        if len(db) != 0:
            if db[-1][1] == code:
                forget_cur.execute('DELETE FROM forgetPass WHERE email=:em;', {"em": email})
                return "it's ok "
            else:
                return "wrong code"
        else:
            return "something went wrong try again"


@app.route('/updatePass', methods=['POST'])
def updatePass():
    data = request.get_json()
    email = str(data['email'])
    password = str(data['password'])
    with sqlite3.connect('userDB.db') as con:
        forget_cur = con.cursor()
        forget_cur.execute("UPDATE users SET password =:pass  WHERE email=:em;", {"em": email,"pass":password})
        con.commit()
    return "it's ok "

@app.route('/debugHeroku')
def debug():
    return " heroku host works good"
