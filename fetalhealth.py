# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Reading the pickle files created in pynb 
# random forest pickle
rf_pickle = open('random_forest_fetal.pickle', 'rb')
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()
# decision tree pickle
dt_pickle = open('decision_tree_fetal.pickle', 'rb')
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()
# adaboost pickle
adb_pickle = open('adb_fetal.pickle', 'rb')
adb_model = pickle.load(adb_pickle) 
adb_pickle.close()
# soft voting pickle
voting_pickle = open('voting_fetal.pickle', 'rb')
voting_model = pickle.load(voting_pickle) 
voting_pickle.close()

# Load the default dataset
fetal_df = pd.read_csv('fetal_health.csv')
fetal_df.dropna(inplace = True)
fetal_df = fetal_df.drop(columns = ['fetal_health'])

# Set up the title and description of the app
st.markdown("<h1 style='text-align: center; color: maroon;'>Fetal Health Classification:<br />A Machine Learning App</h1>", unsafe_allow_html=True)

st.image("fetal_health_image.gif", use_column_width = True)
st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")

st.sidebar.header("Fetal Features Input")
user_data = st.sidebar.file_uploader("Upload your data", help="File must be in CSV format")
st.sidebar.warning("⚠️ Ensure your data stricly follows the format below.")
st.sidebar.write(fetal_df.head())
model = st.sidebar.radio("Choose Model for Prediction", options=["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"])
st.sidebar.info(f"You selected: {model}", icon="🤖")

def make_tabs(cm_pic, report_pic, feature_pic):
    # Showing additional items in tabs
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

    # Tab 1: Confusion Matrix
    with tab1:
        st.write("### Confusion Matrix")
        st.image(cm_pic)
        st.caption("Confusion Matrix of model predictions.")

    # Tab 2: Classification Report
    with tab2:
        st.write("### Classification Report")
        report_df = pd.read_csv(report_pic, index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
    
    # Tab 3: Feature Importance Visualization
    with tab3:
        st.write("### Feature Importance")
        st.image(feature_pic)
        st.caption("Features used in this prediction are ranked by relative importance.")

def highlight_cells(val):
    if val == "Normal":
        color = 'lime'
    elif val == "Suspect":
        color = 'yellow'
    else:
        color = 'orange'
    return f'background-color: {color}'

if user_data is None:
    st.info("ℹ️ *Please upload your data to proceed*")
if user_data is not None:
    st.success("✅ *CSV file uploaded successfully*")

    if model == "Random Forest":
        user_data = pd.read_csv(user_data)      # READ the uploaded data
        user_data.dropna(inplace = True)
        user_copy = user_data.copy()

        st.header("Predicting Fetal Health Class Using Random Forest Model")
        user_data["Prediction"] = rf_model.predict(user_data)

        # Get prediction probabilities and store the highest probability for each prediction
        user_data["Prediction Probability"] = (rf_model.predict_proba(user_copy).max(axis=1)*100).round(1)
        user_data["Prediction Probability"] = user_data["Prediction Probability"].map("{:.1f}".format)

        user_data["Prediction"] = user_data['Prediction'].replace(1.000000, "Normal")
        user_data["Prediction"] = user_data['Prediction'].replace(2.000000, "Suspect")
        user_data["Prediction"] = user_data['Prediction'].replace(3.000000, "Pathological")

        df_styled = user_data.style.applymap(highlight_cells, subset=['Prediction'])
        df_styled
        make_tabs("randfor_conf_mat.svg", "randfor_class_report.csv", "randfor_feature_imp.svg")

    if model == "Decision Tree":
        user_data = pd.read_csv(user_data)      # READ the uploaded data
        user_data.dropna(inplace = True)
        user_copy = user_data.copy()

        st.header("Predicting Fetal Health Class Using Decision Tree Model")
        user_data["Prediction"] = dt_model.predict(user_data)

        # Get prediction probabilities and store the highest probability for each prediction
        user_data["Prediction Probability"] = (dt_model.predict_proba(user_copy).max(axis=1)*100).round(1)
        user_data["Prediction Probability"] = user_data["Prediction Probability"].map("{:.1f}".format)


        user_data["Prediction"] = user_data['Prediction'].replace(1.000000, "Normal")
        user_data["Prediction"] = user_data['Prediction'].replace(2.000000, "Suspect")
        user_data["Prediction"] = user_data['Prediction'].replace(3.000000, "Pathological")

        df_styled = user_data.style.applymap(highlight_cells, subset=['Prediction'])
        df_styled
        make_tabs("confusion_mat.svg", "class_report.csv", "dt_feature_imp.svg")

    if model == "AdaBoost":
        user_data = pd.read_csv(user_data)      # READ the uploaded data
        user_data.dropna(inplace = True)
        user_copy = user_data.copy()

        st.header("Predicting Fetal Health Class Using AdaBoost Model")
        user_data["Prediction"] = adb_model.predict(user_data)

        # Get prediction probabilities and store the highest probability for each prediction
        user_data["Prediction Probability"] = (adb_model.predict_proba(user_copy).max(axis=1)*100).round(1)
        user_data["Prediction Probability"] = user_data["Prediction Probability"].map("{:.1f}".format)

        user_data["Prediction"] = user_data['Prediction'].replace(1.000000, "Normal")
        user_data["Prediction"] = user_data['Prediction'].replace(2.000000, "Suspect")
        user_data["Prediction"] = user_data['Prediction'].replace(3.000000, "Pathological")

        df_styled = user_data.style.applymap(highlight_cells, subset=['Prediction'])
        df_styled
        make_tabs("adb_conf_mat.svg", "adb_class_report.csv", "adb_feature_imp.svg")

    if model == "Soft Voting":
        user_data = pd.read_csv(user_data)      # READ the uploaded data
        user_data.dropna(inplace = True)
        user_copy = user_data.copy()

        st.header("Predicting Fetal Health Class Using Soft Voting Model")
        user_data["Prediction"] = voting_model.predict(user_data)

        # Get prediction probabilities and store the highest probability for each prediction
        user_data["Prediction Probability"] = (voting_model.predict_proba(user_copy).max(axis=1)*100).round(1)
        user_data["Prediction Probability"] = user_data["Prediction Probability"].map("{:.1f}".format)

        user_data["Prediction"] = user_data['Prediction'].replace(1.000000, "Normal")
        user_data["Prediction"] = user_data['Prediction'].replace(2.000000, "Suspect")
        user_data["Prediction"] = user_data['Prediction'].replace(3.000000, "Pathological")

        df_styled = user_data.style.applymap(highlight_cells, subset=['Prediction'])
        df_styled
        make_tabs("voting_conf_mat.svg", "voting_class_report.csv", "voting_feature_imp.svg")



