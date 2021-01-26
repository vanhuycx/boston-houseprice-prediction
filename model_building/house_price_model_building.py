import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df_train = pd.read_csv('cleaned_train.csv')
# https://github.com/vanhuycx/houseprice-prediction/blob/master/cleaned_train.csv

# Separating X and y
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']

# Build random forest model
model = RandomForestRegressor(n_estimators=100, criterion='mae')
model.fit(X, y)

# Saving the model
pickle.dump(model, open('house_price_model.pkl', 'wb'))
