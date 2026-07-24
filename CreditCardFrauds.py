#Wykrywanie oszustw kartą kredytową
#Bardzo niezbalansowane dane, z którymi musisz sobie poradzić stosując różne techniki.
# Wykryj fraudy w danych, czyli wytrenuj model klasyfikacji na danych. Zanalizuj alarmy, fałszywe alarmy w danych z transakcji w bankach.
#Zobacz, które z cech potrzebujesz do predykcji, a których nie (nie zmieniają wyników modelu, są mniej #istotne). Możesz sprawdzić różne modeli i porównać wyniki.
#Link do zbioru danych (możesz wybrać inne): Credit Card Fraud Detection

#CO POWINNO ZNAJDOWAĆ SIĘ W PROJEKCIE:
#Czyszczenie i przetwarzanie danych, feature processing
#Analiza danych i tematu
#Podział danych
#Trening modelu
#Ocena modelu i wnioski
#Dodatkowo jeżeli chciałbyś stwórz prosty web service z REST API pokazujący Twój projekt

import pandas as pd #import bibliotek
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DataFrame = pd.read_csv('creditcard.csv')

print(DataFrame.shape) #sprawdzamy czy dane sa git
print(DataFrame.head()) #tylko pierwsze wiersze wyswietlamy bo calosc by rozsadzila kompaxd
print(DataFrame['Class'].value_counts()) #szybki check wymiarow
print("Braki danych: ", DataFrame.isnull().sum().sum()) #sprawdzamy ile jest jakichkolwiek brakow danych w kazdej encji
print("Dupliakty:  ", DataFrame.duplicated().sum()) #liczym duplikaty

before = DataFrame.shape[0] #liczymy ile mamy wierszy przed usuwaniem zeby ladnie porownac
DataFrame = DataFrame.drop_duplicates() #operacja troche jak sql wywalamy wszytskie duplikaty i dostajemy nowa tabele
after = DataFrame.shape[0] #nadpis tabeli
print(f"Usunieto {before - after} duplikatow ") #fancy statystki
print(f"Nowy rozmiar tabeli: {DataFrame.shape}")

liczba_normalnych = DataFrame['Class'].value_counts()[0] #wyznaczamy ilosc git transakcji fraudow i stosunek procentowy
liczba_fraudow = DataFrame['Class'].value_counts()[1]
procent_fraudow = liczba_fraudow / len(DataFrame) * 100

print(f"Transakcje normalne: {liczba_normalnych}")
print(f"Transakcje fraudowe: {liczba_fraudow}")
print(f"Procent fraudow: {procent_fraudow: .4f}%")

DataFrame["Class"].value_counts().plot(kind='bar') #tworzenie wykresu
plt.title("Liczba transakcji wg klasy")
plt.xlabel("Class (0 = normalna, 1 = fraud)")
plt.ylabel("Liczba transakcji ")
plt.yscale('log') #skala logarytczmiczna zeby widziec lepiej fraud
plt.show()

print(DataFrame.groupby("Class")["Amount"].describe()) #sprawdzamy parametry statystyczne transakcji nornmalnych i fraudow

sns.boxplot(x='Class', y = 'Amount', data=DataFrame, showfliers=False) #wykres do zoobrazowania zakresow kwot i parametrow statystycznych na transakcjach i fraudach
plt.title("Kwota transakcji wg klasy (bez wartosci odstajacych na wykresie)")
plt.xlabel("Class (0 = normalna, 1 = fraud)")
plt.ylabel("Amount")
plt.show()

from sklearn.preprocessing import StandardScaler #dane czasu i wartosci sa syntetycznie manipulowane aby dalszyt model uczenia nie faworyzowal wiekszych wartosci ktore w tym przypadku nie maja odniesienia do prowadzonej analizy

scaler = StandardScaler()
DataFrame[['Amount', 'Time']] = scaler.fit_transform(DataFrame[['Amount', 'Time']])

print(DataFrame[['Amount', 'Time']].describe())

X = DataFrame.drop(columns= ["Class"]) #parametry decyzji modelu
y = DataFrame["Class"] #wynik modelu

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split #dzielimy dane na 2 srodowiska testowe

(X_train, X_test, y_train,
 y_test) = train_test_split(X, y,
test_size = 0.2, stratify=y, random_state = 42)

print("Zbior treningowy: ", X_train.shape[0], "wierszy, fraudy: ", y_train.sum())
print("Zbior testowy: ", X_test.shape[0], "wierszy, fraudy: ", y_test.sum())

from sklearn.linear_model import LogisticRegression #tworzenie modelu

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42) #jako ze dane sa niezbalansowane to uzywamy metody balanced zeby byl bardziej warazliwy na bledne wyniki modelu. regresja jest iteracyjna, do 1000, aby model dostrajal swoje wagi
model.fit(X_train, y_train)

print("Model wytrenowany")

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test) #model dostaje dane testowe i analizuje czy fraud czy normalna

print(confusion_matrix(y_test, y_pred)) #tworzenie macierzy z wynikami i skutecznosci modelu
print(classification_report(y_test, y_pred, target_names=["Normalna", "Fraud"])) #podsumowanie w formie tabeli

from imblearn.over_sampling import SMOTE #sprawdzimy inny model i porownamy wyniki
from sklearn.ensemble import RandomForestClassifier

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Przed SMOTE: ", y_train.value_counts().to_dict())
print("Po SMOTE: ", y_train_sm.value_counts().to_dict())

model_rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model_rf.fit(X_train_sm, y_train_sm)

print("Random Forest wytrenowany")

y_pred_rf = model_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred_rf)) #nizszy recall kosztem wyzszej precyzji
print(classification_report(y_test, y_pred_rf, target_names=["Normalna", "Fraud"]))

import xgboost as xgb #3 model do porownania

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("Scale pos weight: ", scale_pos_weight)

model_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
                              random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names= ["Normalna", "Fraud"]))

from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score #import bibliotek liczacych metryki modeli

wyniki = []

for nazwa, y_pred, model_obj in [("Logistic regression", y_pred, model),
                                 ("Random Forrest + SMOTE", y_pred_rf, model_rf),
                                 ("XGBoost", y_pred_xgb, model_xgb),]:
    wyniki.append({"Model": nazwa, "Precison": precision_score(y_test, y_pred), #porownanie poprawnych odp z testowymi
                   "Recall": recall_score(y_test, y_pred),
                   "F1": f1_score(y_test, y_pred),})

wyniki_df = pd.DataFrame(wyniki)
print(wyniki_df)

importances = pd.DataFrame({"Cecha": X_train.columns, #robimy ranking najwazniejszych rekordow do trenowania modelu i mozemy zdecydoweac ktore parametry mozna pominac w usprawnianiu modelu, a ktore sa najawazniejsze
                            "Waznosc": model_xgb.feature_importances_}).sort_values("Waznosc", ascending=False)
print(importances)

plt.figure(figsize=(8,10))
plt.barh(importances["Cecha"], importances["Waznosc"])
plt.gca().invert_yaxis()
plt.title("Waznosc cech wg XGBoost")
plt.xlabel("Waznosc")
plt.tight_layout()
plt.show()

slabe_cechy= importances[importances["Waznosc"] < 0.01]["Cecha"].tolist()
print("Cechy do odrzucenia (waznosc < 0,01): ",slabe_cechy)

X_train_reduced = X_train.drop(columns=slabe_cechy) #sprawdzamy jak dziala nasz model bez slabych cech
X_test_reduced = X_test.drop(columns=slabe_cechy)

model_xgb_reduced = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      scale_pos_weight=scale_pos_weight,
                                      eval_metric='aucpr',random_state=42,n_jobs=-1)
model_xgb_reduced.fit(X_train_reduced, y_train)
y_pred_reduced = model_xgb_reduced.predict(X_test_reduced)

print("    XGBoost z wszystkimi cechami     ")
print(classification_report(y_test, y_pred_xgb, target_names=["Normalna", "Fraud"]))

print("   XGBoost bez slabych cech (", X_train_reduced.shape[1], ")    ")
print(classification_report(y_test, y_pred_reduced, target_names=["Normalna", "Fraud"]))

# WNIOSKI
# ============================================================

# Niezbalansowanie danych.
# Zbior zawieral 283 253 transakcje normalne i zaledwie 473 fraudy
# (ok. 0,17% wszystkich transakcji). Gdyby model ignorowal fraudy,
# bank bylby narazony na bezposrednie straty finansowe i pozostawial
# otwarta furtke dla kolejnych oszustw dlatego mimo skrajnej
# rzadkosci tej klasy, jej poprawne wykrywanie bylo priorytetem,
# nie dodatkiem.

# Porownanie modeli.
# Pod wzgledem samej skutecznosci wykrywania fraudow (recall)
# najlepiej wypadla regresja logistyczna (87%), ale kosztem bardzo
# niskiej precyzji (6%) model generowal mnostwo falszywych alarmow
# na normalnych transakcjach. Najlepszy ogolny balans osiagnal
# XGBoost: precyzja 94% przy recall nizszym o 11 pp. wzgledem
# regresji logistycznej (77% vs 87%). To znaczaco lepszy wynik niz
# Random Forest + SMOTE, gdzie precyzja wyniosla 71% przy
# porownywalnym recall (78%, o 1 pp. wyzszym niz XGBoost).

# Czy SMOTE pomogl?
# Ciezko odpowiedziec jednoznacznie. SMOTE (Random Forest) dal
# wyraznie wyzsza precyzje niz samo wazenie klas (Logistic
# Regression): 71% vs 6%, ale kosztem nizszego recall (78% vs 87%).
# SMOTE pomogl wiec ograniczyc liczbe falszywych alarmow, ale nie byl
# jednoznacznie "lepszy" -- wybor zalezy od tego, ktora metryke
# priorytetyzujemy.

# Redukcja cech.
# Udalo sie zmniejszyc liczbe cech na podstawie ich istotnosci --
# usunieto 10 najmniej istotnych cech (z 30 do 20), co delikatnie
# poprawilo skutecznosc wykrywania fraudow (recall: 77% do 78%,
# precyzja: 94% do 96%) przy jednoczesnym skroceniu czasu treningu
# modelu.

# Rekomendacja dla banku.
# Przede wszystkim nalezaloby policzyc, co jest drozsze dla
# konkretnego banku: wyzsza skutecznosc wykrywania fraudow kosztem
# wielu blednych wykryc, czy odpuszczenie czesci fraudow przy
# znaczaco mniejszej liczbie ogolnych podejrzen. Bez tej kalkulacji
# kosztowej wybor modelu jest subiektywny - jednak intuicyjnie
# przeoczony fraud (FN) zwykle kosztuje bank wiecej niz falszywy
# alarm (FP, ktory wymaga jedynie dodatkowej weryfikacji), co
# przemawialoby za modelem z wyzszym recall, nawet kosztem czesci
# precyzji.

import joblib

joblib.dump(model_xgb_reduced, 'fraud_model.pkl')
joblib.dump(slabe_cechy, 'slabe_cechy.pkl')

print("Model zapisany do pliku")