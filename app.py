from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Muat model dan scaler
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('attrition_scaler.pkl')

# Fitur yang digunakan dalam model
features = ['Age', 'Gender', 'Time_of_service', 'Education_Level', 'Travel_Rate', 
            'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7']

# Buat DataFrame dengan kolom yang sesuai
def create_input_df(data):
    # Konversi Gender menjadi numerik
    data['Gender'] = 1 if data['Gender'] == 'M' else 0
    # Ubah nilai lainnya menjadi float
    for key in data:
        if key != 'Gender':
            data[key] = float(data[key])
    # Buat DataFrame dari data input
    input_df = pd.DataFrame(data, index=[0])
    # Tambahkan kolom yang tidak digunakan dalam prediksi dengan nilai default (misal 0)
    for col in scaler.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    # Urutkan kolom sesuai dengan scaler
    input_df = input_df[scaler.feature_names_in_]
    return input_df

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil nilai-nilai dari formulir yang di-submit
    age = request.form['Age']
    gender = request.form['Gender']
    time_of_service = request.form['Time_of_service']
    education_level = request.form['Education_Level']
    travel_rate = request.form['Travel_Rate']
    work_life_balance = request.form['Work_Life_balance']
    var1 = request.form['VAR1']
    var2 = request.form['VAR2']
    var3 = request.form['VAR3']
    var4 = request.form['VAR4']
    var5 = request.form['VAR5']
    var6 = request.form['VAR6']
    var7 = request.form['VAR7']

    # Ambil data dari form
    data = request.form.to_dict()
    input_df = create_input_df(data)
    
    # Skalakan fitur
    features_scaled = scaler.transform(input_df)
    
    # Prediksi tingkat attrisi
    prediction = model.predict(features_scaled)[0] * 100
    
    return render_template('index.html', age=age, gender=gender, time_of_service=time_of_service, 
                           education_level=education_level, travel_rate=travel_rate, 
                           work_life_balance=work_life_balance, var1=var1, var2=var2, 
                           var3=var3, var4=var4, var5=var5, var6=var6, var7=var7, prediction_text=f'Tingkat Attrisi: {prediction:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)
