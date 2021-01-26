import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

st.write("""
# House Price Prediction App

This app predicts the ** house price ** based on provided criteria!
""")
st.write('---')

# Loads the Boston House Price Dataset
df_train = pd.read_csv('cleaned_train.csv')
X = df_train.drop('SalePrice', axis=1)
Y = df_train['SalePrice']

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    OverallQual = st.sidebar.slider('Rate the overall material and finish of the house', float(X.OverallQual.min()),
                                    float(X.OverallQual.max()), float(X.OverallQual.mean()))
    GrLivArea = st.sidebar.slider('Above ground living area square feet',
                                  float(X.GrLivArea.min()), float(X.GrLivArea.max()), float(X.GrLivArea.mean()))
    GarageCars = st.sidebar.slider('Size of garage in car capacity', float(X.GarageCars.min()),
                                   float(X.GarageCars.max()), float(X.GarageCars.mean()))
    TotalBsmtSF = st.sidebar.slider('Total square feet of basement area', float(X.TotalBsmtSF.min()),
                                    float(X.TotalBsmtSF.max()), float(X.TotalBsmtSF.mean()))
    FullBath = st.sidebar.slider(
        'Full bathrooms above grade', float(X.FullBath.min()), float(X.FullBath.max()), float(X.FullBath.mean()))
    YearBuilt = st.sidebar.slider('Original construction date',
                                  float(X.YearBuilt.min()), float(X.YearBuilt.max()), float(X.YearBuilt.mean()))

    data = {'OverallQual': OverallQual,
            'GrLivArea': GrLivArea,
            'GarageCars': GarageCars,
            'TotalBsmtSF': TotalBsmtSF,
            'FullBath': FullBath,
            'YearBuilt': YearBuilt
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


load_model = pickle.load(open('house_price_model.pkl', 'rb'))

# Apply Model to Make Prediction
prediction = load_model.predict(df)

st.header('Prediction of house price (in thousand dollars)')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(load_model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X)
st.pyplot(fig, bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(fig, bbox_inches='tight')
