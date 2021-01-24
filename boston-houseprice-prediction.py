import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    CRIM = st.sidebar.slider('Crime rate by town (CRIM)', X.CRIM.min(),
                             X.CRIM.max(), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('Proportion of residential land (ZN)',
                           X.ZN.min(), X.ZN.max(), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('Proportion of non-retail business acres per town (INDUS)', X.INDUS.min(),
                              X.INDUS.max(), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('Charles River (CHAS = 1 if tract bounds river; 0 otherwise)', X.CHAS.min(),
                             X.CHAS.max(), float(X.CHAS.mean()))
    NOX = st.sidebar.slider(
        'Nitrogen oxides concentratio (NOX)', X.NOX.min(), X.NOX.max(), float(X.NOX.mean()))
    RM = st.sidebar.slider('Average number of rooms (RM)',
                           X.RM.min(), X.RM.max(), float(X.RM.mean()))
    AGE = st.sidebar.slider(
        'Proportion of owner-occupied units (AGE)', X.AGE.min(), X.AGE.max(), float(X.AGE.mean()))
    DIS = st.sidebar.slider(
        'Distances to five Boston employment centres (DIS)', X.DIS.min(), X.DIS.max(), float(X.DIS.mean()))
    RAD = st.sidebar.slider(
        'Index of accessibility to radial highways (RAD)', X.RAD.min(), X.RAD.max(), float(X.RAD.mean()))
    TAX = st.sidebar.slider(
        'Property tax rate per $10,000 (TAX)', X.TAX.min(), X.TAX.max(), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider(
        'Pupil-teacher ratio by town (PTRATIO)', X.PTRATIO.min(), X.PTRATIO.max(), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('Proportion of black people by town (B)',
                          X.B.min(), X.B.max(), float(X.B.mean()))
    LSTAT = st.sidebar.slider('Lower status of the population in percentage (LSTAT)', X.LSTAT.min(),
                              X.LSTAT.max(), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# # Build Regression Model
# model = RandomForestRegressor()
# model.fit(X, Y)
# Reads in saved classification model

load_model = pickle.load(open('boston-houseprice-model.pkl', 'rb'))

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
