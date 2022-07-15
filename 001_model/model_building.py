import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


# df = pd.read_csv("haberman.csv")
df = pd.read_csv('ecom-user-churn-data_balanced.csv')
X = df.loc[:, "ses_rec":"user_rec"]
y = df.loc[:, 'target'] # 1: died within five years, 0: survived 5+ years

model = LogisticRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
