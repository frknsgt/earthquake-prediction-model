import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data=pd.read_excel("Data.xlsx")
buyukluk=data.Büyüklük.values.reshape(-1,1)

koordinat=data.iloc[:,[1,2]]
regressionRandom=RandomForestRegressor(n_estimators=1000)
regressionRandom.fit(koordinat, buyukluk)

regressionRandom.predict([[38.2411, 25.7731]])
score="Doğruluk Oranı: %"+str(100*(r2_score(buyukluk, regressionRandom.predict(koordinat))))

from sklearn.externals import joblib
joblib.dump(regressionRandom, "Earthquake_model.pkl")