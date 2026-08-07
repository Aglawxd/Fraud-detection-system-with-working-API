from flask import Flask, request, jsonify #tworzymy serwer na flask

import joblib
import pandas as pd

app=Flask(__name__) #twoerzenie obiektu apliakci

model = joblib.load("fraud_model.pkl") #wczytywanie danych
slabe_cechy = joblib.load("slabe_cechy.pkl")

print("Model wczytany i gotowy do pracy")

@app.route("/", methods=['GET']) #wlaczanie funkcji ponizej przy wchodzeniu w przegladarka
def home(): #funkcja sprawdzajaca czy serwer dziala
    return jsonify({'Status': 'Working'})

@app.route("/predict", methods=['POST'])
def predict():
    dane = request.get_json()

    df_wejscie = pd.DataFrame([dane])
    df_wejscie = df_wejscie.drop(columns=slabe_cechy)

    predykcja = model.predict(df_wejscie)[0]
    prawdopodobienstwo = model.predict_proba(df_wejscie)[0][1]

    wynik = {"Fraud": bool(predykcja),
             "prawdopodobienstwo fraudu": float(prawdopodobienstwo)}
    return jsonify(wynik)

if __name__ == "__main__": #wlacza serwer gdy odpalamy plik pod konkretnym adresem
    app.run(debug=True, port=5000)



