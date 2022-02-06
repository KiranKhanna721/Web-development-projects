from flask import Flask, request, jsonify, render_template,send_from_directory,redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

import pandas as pd
import numpy as np
import pickle
import os
import PIL 
import cv2
from keras.models import load_model

app = Flask(__name__)

heart = pickle.load(open('heart.pkl', 'rb'))
bp = pickle.load(open('bp.pkl', 'rb'))
d = pickle.load(open('d.pkl', 'rb'))
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app.config['SECRET_KEY'] = 'NOBODY-CAN-GUESS-THIS'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20))
    email = db.Column(db.String(30), unique=True)
    password = db.Column(db.String(80))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=5, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message="Invalid Email"), Length(min=6, max=30)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=5, max=80)])


@app.route('/')
def index():
    return render_template('/index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            # compares the password hash in the db and the hash of the password typed in the form
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))
        return 'invalid username or password'

    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        # add the user form input which is form.'field'.data into the column which is 'field'
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return 'new user has been created bro!'

    return render_template('signup.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/heartpredict', methods=["GET", "POST"])
@login_required
def heartpredict():
    if request.method == "POST":
        l=[]
        age = int(request.form["age"])
        l.append(age)
        sex = int(request.form["sex"])
        l.append(sex)
        cp = int(request.form["cp"])
        l.append(cp)
        trestbps = int(request.form["trestbps"])
        l.append(trestbps)
        chol = int(request.form["chol"])
        l.append(chol)
        fbs = int(request.form["fbs"])
        l.append(fbs)
        restecg = int(request.form["restecg"])
        l.append(restecg)
        thalach = int(request.form["thalach"])
        l.append(thalach)
        exang = int(request.form["exang"])
        l.append(exang)
        oldpeak = int(request.form["oldpeak"])
        l.append(oldpeak)
        slope = int(request.form["slope"])
        l.append(slope)
        ca = int(request.form["ca"])
        l.append(ca)
        thal = int(request.form["thal"])
        l.append(thal)
        arr = np.array([l])
        prediction = heart.predict(arr)
        out=""
        if prediction == 0:
            out = "No"
        elif prediction ==1:
            out = "Yes "
        return render_template('/heart.html',predict='{} Heart disease'.format(out))
    return render_template('/heart.html',predict=None)

@app.route('/bppredict', methods=["GET", "POST"])
@login_required
def bppredict():
    if request.method == "POST":
        l=[]
        Level_of_Hemoglobin = int(request.form["Level_of_Hemoglobin"])
        l.append(Level_of_Hemoglobin)
        Genetic_Pedigree_Coefficient = int(request.form["Genetic_Pedigree_Coefficient"])
        l.append(Genetic_Pedigree_Coefficient)
        Age = int(request.form["Age"])
        l.append(Age)
        BMI = int(request.form["BMI"])
        l.append(BMI)
        Sex = int(request.form["Sex"])
        l.append(Sex)
        Smoking = int(request.form["Smoking"])
        l.append(Smoking)
        Physical_activity = int(request.form["Physical_activity"])
        l.append(Physical_activity)
        salt_content_in_the_diet = int(request.form["salt_content_in_the_diet"])
        l.append(salt_content_in_the_diet)
        alcohol_consumption_per_day = int(request.form["alcohol_consumption_per_day"])
        l.append(alcohol_consumption_per_day)
        Level_of_Stress = int(request.form["Level_of_Stress"])
        l.append(Level_of_Stress)
        Chronic_kidney_disease = int(request.form["Chronic_kidney_disease"])
        l.append(Chronic_kidney_disease)
        Adrenal_and_thyroid_disorders = int(request.form["Adrenal_and_thyroid_disorders"])
        l.append(Adrenal_and_thyroid_disorders)
        arr = np.array([l])
        
        prediction = bp.predict(arr)
        out=""
        if prediction == 0:
            out = "No"
        elif prediction ==1:
            out = "Yes"
        return render_template('/bp.html',predict='{} you have Bp'.format(out))
    return render_template('/bp.html',predict=None)



@app.route('/dpredict', methods=["GET", "POST"])
@login_required
def dpredict():
    if request.method == "POST":
        l=[]
        no_times_pregnant = int(request.form["no_times_pregnant"])
        l.append(no_times_pregnant)
        glucose_concentration = int(request.form["glucose_concentration"])
        l.append(glucose_concentration)
        blood_pressure = int(request.form["blood_pressure"])
        l.append(blood_pressure)
        skin_fold_thickness = int(request.form["skin_fold_thickness"])
        l.append(skin_fold_thickness)
        serum_insulin = int(request.form["serum_insulin"])
        l.append(serum_insulin)
        bmi = int(request.form["bmi"])
        l.append(bmi)
        diabetes_pedigree = int(request.form["diabetes pedigree"])
        l.append(diabetes_pedigree)
        age = int(request.form["age"])
        l.append(age)
        arr = np.array([l])   
        prediction = d.predict(arr)
        out=""
        if prediction == 0:
            out = "No"
        elif prediction ==1:
            out = "Yes"
        return render_template('/d.html',predict='{} you have Diabetes'.format(out))
    return render_template('/d.html',predict=None)

if __name__ == '__main__':
    app.run(debug=True)