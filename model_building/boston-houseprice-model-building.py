import pandas as pd 
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# Loads the Boston House Price Dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)


# Saving the model
import pickle
pickle.dump(model, open('boston-houseprice-model.pkl', 'wb'))