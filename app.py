import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Global Development Clustering",layout="wide")

@st.cache_resource
def load_files():
    with open("clustering_model.pkl","rb") as file:
        model=pickle.load(file)
    with open("scaler.pkl","rb") as file:
        scaler=pickle.load(file)
    with open("features.pkl","rb") as file:
        features=pickle.load(file)
    return model,scaler,features

@st.cache_data
def load_dataset():
    return pd.read_excel("World_development_mesurement.xlsx")

model,scaler,features=load_files()
df=load_dataset()

st.sidebar.title("Navigation")
page=st.sidebar.radio("Go to",["Overview","EDA","Model Building","Model Evaluation","Cluster Analysis","Prediction"])

if page=="Overview":
    st.title("Global Development Clustering")
    st.write("This project groups countries based on global economic, social and development indicators.")
    st.subheader("Business Objective")
    st.write("The objective is to create meaningful clusters of countries based on development-related measurements.")
    col1,col2,col3=st.columns(3)
    col1.metric("Total Records",df.shape[0])
    col2.metric("Original Features",df.shape[1])
    col3.metric("Final Clusters",2)
    st.subheader("Project Workflow")
    st.write("EDA → Data Preprocessing → Feature Scaling → Clustering → Model Evaluation → Cluster Profiling → Deployment")

elif page=="EDA":
    st.title("Exploratory Data Analysis")
    st.write("EDA was performed to understand the structure, distribution and relationships between the development indicators.")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    col1,col2=st.columns(2)
    col1.metric("Rows",df.shape[0])
    col2.metric("Columns",df.shape[1])
    st.subheader("Missing Values")
    missing=df.isnull().sum().sort_values(ascending=False)
    missing=missing[missing>0]
    if len(missing)>0:
        st.dataframe(missing.to_frame("Missing Values"))
    else:
        st.success("No missing values found.")
    st.subheader("Numerical Feature Distribution")
    numeric_cols=df.select_dtypes(include=np.number).columns.tolist()
    selected_feature=st.selectbox("Select a feature",numeric_cols)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.hist(df[selected_feature].dropna(),bins=30)
    ax.set_title(f"Distribution of {selected_feature}")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Correlation Heatmap")
    corr=df[numeric_cols].corr()
    fig,ax=plt.subplots(figsize=(12,8))
    im=ax.imshow(corr,cmap="coolwarm",aspect="auto")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols,rotation=90,fontsize=7)
    ax.set_yticklabels(numeric_cols,fontsize=7)
    fig.colorbar(im,ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)
    st.info("The analysis showed strong relationships among some development variables. For example, CO2 emissions and energy usage were strongly related, while female and male life expectancy were also strongly related. Highly correlated variables were reduced before clustering.")

elif page=="Model Building":
    st.title("Model Building")
    st.write("Five clustering algorithms were applied to the processed and scaled dataset.")
    model_data=pd.DataFrame({
        "Model":["K-Means","Hierarchical Clustering","Gaussian Mixture","DBSCAN","BIRCH"],
        "Description":[
            "Groups observations around cluster centroids.",
            "Gradually merges similar observations into clusters.",
            "Uses probability distributions to identify clusters.",
            "Forms clusters based on data density.",
            "Creates compact subclusters before forming final clusters."
        ]
    })
    st.dataframe(model_data,use_container_width=True)
    st.subheader("Final K-Means Configuration")
    st.write("Number of clusters selected: 2")
    st.write("Random state: 42")
    st.write("Initialization attempts: 10")
    st.info("The Elbow Method and Silhouette Score were used to study the suitable number of clusters. The final K-Means model was built using K = 2.")

elif page=="Model Evaluation":
    st.title("Model Evaluation")
    st.write("The clustering models were compared using three internal evaluation metrics.")
    results_df=pd.DataFrame({
        "Model":["K-Means","Hierarchical","Gaussian Mixture","DBSCAN","BIRCH"],
        "Silhouette Score":[0.272742,0.259199,0.244293,-0.043177,0.258550],
        "Calinski-Harabasz Score":[1208.148626,920.519300,1060.684775,66.673415,1042.265645],
        "Davies-Bouldin Score":[1.387708,1.313466,1.509181,1.235861,1.427982]
    })
    st.dataframe(results_df,use_container_width=True)
    st.subheader("Metric Interpretation")
    st.write("Silhouette Score: Higher values generally indicate better cluster separation.")
    st.write("Calinski-Harabasz Score: Higher values generally indicate more compact and separated clusters.")
    st.write("Davies-Bouldin Score: Lower values generally indicate better separation between clusters.")
    st.subheader("Model Comparison")
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(results_df["Model"],results_df["Silhouette Score"])
    ax.set_title("Silhouette Score Comparison")
    ax.set_ylabel("Silhouette Score")
    ax.tick_params(axis="x",rotation=30)
    st.pyplot(fig)
    plt.close(fig)
    st.success("Based on the comparison, K-Means was selected as the final model because it achieved the highest Silhouette Score of 0.272742.")

elif page=="Cluster Analysis":
    st.title("Cluster Analysis")
    st.write("The final K-Means model divides the observations into two clusters.")
    cluster_df=pd.read_csv("country_clusters.csv")
    col1,col2=st.columns(2)
    cluster_counts=cluster_df["Cluster"].value_counts().sort_index()
    col1.metric("Cluster 0",int(cluster_counts.get(0,0)))
    col2.metric("Cluster 1",int(cluster_counts.get(1,0)))
    st.subheader("Countries and Their Clusters")
    st.dataframe(cluster_df,use_container_width=True)
    selected_cluster=st.selectbox("Select Cluster",sorted(cluster_df["Cluster"].unique()))
    filtered=cluster_df[cluster_df["Cluster"]==selected_cluster]
    st.write(f"Countries belonging to Cluster {selected_cluster}")
    st.dataframe(filtered,use_container_width=True)
    st.info("Cluster labels are numerical identifiers. The cluster profiles can be used to understand the development characteristics of each group.")

elif page=="Prediction":
    st.title("Global Development Cluster Prediction")
    st.write("Enter the development indicator values below to identify the cluster for a new observation.")
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
