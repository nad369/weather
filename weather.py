import streamlit as st
import requests
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

url = "https://archive-api.open-meteo.com/v1/archive?latitude=36.7525&longitude=3.04197&start_date=2022-05-01&end_date=2023-05-31&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,precipitation,weathercode,pressure_msl,surface_pressure,cloudcover,vapor_pressure_deficit,windspeed_10m,winddirection_10m,windgusts_10m,is_day&models=best_match"
response = requests.get(url)
data = response.json()

# Afficher les données
print(data)

# Convertir les données JSON en DataFrame
df = pd.DataFrame(data['hourly'])
# s'assurer encore de l'existence de valeurs manquantes
df.isnull()
df.isnull().sum()
# Trouver les cellules "NaN"
nan_cells = df.isna()

# Afficher les cellules "NaN"
print(nan_cells)

# Convertir la colonne de temps en objet datetime

df['time'] = pd.to_datetime(df['time'])

# Extraire les caractéristiques temporelles
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

# Afficher les premières lignes du DataFrame modifié
print(df.head())

df.head()

# déplacement des colonnes "year", "month", "day" et "hour"
year = df.iloc[:, df.columns.get_loc('year')]
month = df.iloc[:, df.columns.get_loc('month')]
day = df.iloc[:, df.columns.get_loc('day')]
hour = df.iloc[:, df.columns.get_loc('hour')]

df = df.drop(['year', 'month', 'day', 'hour'], axis=1)

df.insert(1, 'year', year)
df.insert(2, 'month', month)
df.insert(3, 'day', day)
df.insert(4, 'hour', hour)

df.head()

# la variable "is day" est aussi catégorielle, donc il y a lieu de l'encoder
df_encoded = pd.get_dummies(df, columns=['is_day'])

X = df.drop(['time', 'temperature_2m'], axis=1)
y = df['temperature_2m']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Créer un objet DMatrix à partir des ensembles d'entraînement et de test
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# Définir les hyperparamètres pour le modèle XGBoost
params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror'
}

# validation croisée 5-fold avec différentes valeurs de num_boost_round
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    nfold=5,
    metrics='rmse',
    early_stopping_rounds=10
)

print(cv_results)

# recherche sur grille avec un modèle XGBoost :
from sklearn.model_selection import GridSearchCV

# Définir l'espace de recherche les hyperparamètres
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Créer un objet XGBRegressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

# Effectuer une recherche sur grille pour trouver les meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres trouvés par la recherche sur grille
print(grid_search.best_params_)

# Définir les hyperparamètres pour le modèle XGBoost
params = {
    'learning_rate': 0.1,
    'max_depth': 9,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'objective': 'reg:squarederror'
}

# Entraîner le modèle XGBoost en utilisant la méthode train() de l'objet Booster
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)

# Évaluer les performances du modèle sur l'ensemble de test
y_pred = model.predict(dtest)

from sklearn.metrics import mean_squared_error, r2_score

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Test RMSE: {rmse:.2f}')
print(f'Test R^2: {r2:.2f}')

from sklearn.model_selection import cross_val_score
import numpy as np

# Sélectionner les caractéristiques et la variable cible
X = df.drop(['time', 'temperature_2m'], axis=1)
y = df['temperature_2m']

# Créer un modèle XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror')

# Évaluer les performances du modèle à l'aide de la validation croisée
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculer la RMSE moyenne
rmse = np.sqrt(-scores.mean())

print(f'Cross-validation RMSE: {rmse:.2f}')

# Créer une instance de ChatBot
chatbot = ChatBot('Mon ChatBot')

# Entraîner le chatbot avec des données préexistantes
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.french")

# Créer un conteneur pour afficher la conversation
conversation_container = st.container()

# Créer un widget de saisie de texte pour permettre à l'utilisateur de saisir un message
user_input = st.text_input("Entrez votre message:")

# Créer un bouton pour envoyer le message au chatbot
if st.button("Envoyer"):
    # Afficher le message de l'utilisateur dans le conteneur de conversation
    conversation_container.write(f"Vous: {user_input}")

    # Obtenir une réponse du chatbot
    response = chatbot.get_response(user_input)

    # Vérifier si la réponse du chatbot contient une demande de prévisions météorologiques
    if "prévisions météorologiques" in response.text:
        # Préparer les données d'entrée pour le modèle XGBoost
        X = ...  # à remplacer par le code pour préparer les données d'entrée pour le modèle

        # Utiliser le modèle XGBoost pour générer les prévisions météorologiques
        y_pred = model.predict(X)

        # Formater les prévisions météorologiques en texte
        forecast_text = f"La température prévue est de {y_pred[0]:.1f}°C." 

        # Afficher les prévisions météorologiques dans le conteneur de conversation
        conversation_container.write(f"Chatbot: {forecast_text}")
    else:
        # Afficher la réponse du chatbot dans le conteneur de conversation
        conversation_container.write(f"Chatbot: {response}")
