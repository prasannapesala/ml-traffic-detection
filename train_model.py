import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("traffic_data.csv")

X = df[["lane1","lane2","lane3","lane4"]]
y = df["green_time"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained")