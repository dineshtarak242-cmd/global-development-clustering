import streamlit as st
import pandas as pd
import pickle
with open("clustering_model.pkl","rb") as file:
    model=pickle.load(file)
with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)
with open("features.pkl","rb") as file:
    features=pickle.load(file)
st.set_page_config(page_title="Global Development Clustering",layout="wide")
st.title("🌍 Global Development Clustering App")
st.write("Enter the feature values below to predict the cluster.")
st.sidebar.header("Enter Feature Values")
input_data={}
for feature in features:
    input_data[feature]=st.sidebar.number_input(feature,value=0.0)
if st.sidebar.button("Predict Cluster"):
    input_df=pd.DataFrame([input_data])
    input_df=input_df[features]
    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)
    st.subheader("Prediction Result")
    st.success(f"The given observation belongs to Cluster {prediction[0]}")
    st.subheader("Input Data")
    st.dataframe(input_df)
