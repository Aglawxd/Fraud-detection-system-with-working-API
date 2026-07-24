import requests
import pandas as pd

DataFrame = pd.read_csv("creditcard.csv")
przykladowy_wiersz = DataFrame.iloc[0]

dane_do_wyslania = przykladowy_wiersz.drop("Class").to_dict() #usuwamy kolumne class bo to ma model nam powiedziec. zamiana wierszu na slownik

print("Wysylam dane: ", dane_do_wyslania)

answer = requests.post("http://localhost:5000/predict", json=dane_do_wyslania) #wylsanie zapytania na serwer

print("     Odpowiedz API: ")
print(answer.json())

fraud_wiersz = DataFrame[DataFrame["Class"] == 1].iloc[0]
dane_fraud = fraud_wiersz.drop('Class').to_dict()

answer = requests.post("http://localhost:5000/predict", json=dane_fraud)
print("     Test na prawdziwym fraudzie: ")
print(answer.json())