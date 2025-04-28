#!/usr/bin/env python
# coding: utf-8

# In[3]:


# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.express as px
from sklearn.metrics import confusion_matrix

# ---------------------------------------
# 🚀 Streamlit Config
# ---------------------------------------
st.set_page_config(page_title="Brain Tumor Classifier 🧠", layout="wide")
st.sidebar.title("🚀 Navigation")

# ---------------------------------------
# 🗺️ Page Selection
# ---------------------------------------
page = st.sidebar.selectbox("Select a page:", ["🏠 Home", "🔎 Image Classification", "📊 Dashboard"])

# ---------------------------------------
# 🏠 Home Page
# ---------------------------------------
if page == "🏠 Home":
    st.title("🧠 Brain Tumor Classification App")
    st.write("""
    Welcome to the Brain Tumor Detection App!  
    Upload an MRI scan and let the deep learning magic happen.  
    Stay curious, stay awesome. ✨
    """)

# ---------------------------------------
# 🔎 Image Classification Page
# ---------------------------------------
elif page == "🔎 Image Classification":
    st.title("🔎 Image Classification")

    uploaded_file = st.file_uploader("Upload an MRI Image 🖼️", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_column_width=True)

        if st.button("Predict 🚀"):
            with st.spinner('Loading model and predicting... ⏳'):
                model = load_model("brain_tumor_model_finetuned_epoch15.h5")

                # Preprocessing
                image_resized = image.resize((224, 224))
                img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

                prediction = model.predict(img_array)
                class_idx = np.argmax(prediction)

                class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
                predicted_label = class_labels[class_idx]

                st.success(f"🎯 Prediction: **{predicted_label}**")
                st.info(f"🔵 Confidence: `{np.max(prediction) * 100:.2f}%`")

                

# ---------------------------
# Dashboard Page
# ---------------------------
elif page == "📊 Dashboard":
    st.title("📊 Dashboard: Brain Tumor MRI Dataset")

    # Load Metadata CSV
    try:
        df = pd.read_csv("brain_tumor_metadata.csv")
    except FileNotFoundError:
        st.error("❌ Metadata CSV not found! Please make sure 'brain_tumor_metadata.csv' exists.")
        st.stop()

    # Sidebar filters
    st.sidebar.header("🔎 Filter the Data:")
    tumor_filter = st.sidebar.multiselect("Select Tumor Type", options=df['TumorType'].unique(), default=df['TumorType'].unique())
    gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    

    # Apply filters
    filtered_df = df[
        (df['TumorType'].isin(tumor_filter)) &
        (df['Gender'].isin(gender_filter)) 
    ]

    st.write(f"🔎 Showing `{filtered_df.shape[0]}` patients based on current filters.")

    # Layout for three graphs
    st.subheader("🔍 Visual Insights")

    # 1. Tumor Type vs Gender (Bar Chart)
    st.markdown("### 1️⃣ Tumor Type vs Gender")
    fig1 = px.histogram(filtered_df,
                        x='TumorType',
                        color='Gender',
                        barmode='group',
                        title="Tumor Type vs Gender Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    
   # 2. Tumor Type vs Age (Histogram - Frequency of Tumors in Age Groups)
    st.markdown("### 2️⃣ Tumor Type vs Age Distribution (Histogram)")

# Create a histogram showing the count of patients per age and tumor type
    fig2 = px.histogram(filtered_df,
                        x='Age',
                        color='TumorType',
                        nbins=30,  # Number of bins (you can adjust this for better granularity)
                        title="Tumor Type Distribution by Age",
                        labels={"Age": "Age", "TumorType": "Tumor Type"},
                        histfunc="count",
                        opacity=0.7)

    st.plotly_chart(fig2, use_container_width=True)

    # 3. Tumor Type across Region (Heatmap)
    st.markdown("### 3️⃣ Tumor Type Across Regions")

# 1) Aggregate & pivot so only selected regions remain
    region_tumor_count = (
        filtered_df
        .groupby(['TumorType', 'Region'])
        .size()
        .reset_index(name='Count')
    )

    pivot_df = (
        region_tumor_count
        .pivot(index='TumorType', columns='Region', values='Count')
        .fillna(0)   # treat missing combos as zero
    )

# 2) Handle case of "no regions selected"
    if pivot_df.shape[1] == 0:
        st.warning("No regions selected — nothing to display!")
    else:
    # 3) Use px.imshow so the axes come straight from pivot_df
        fig3 = px.imshow(
            pivot_df,
            labels={'x': 'Region', 'y': 'Tumor Type', 'color': 'Patient Count'},
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            title="Tumor Frequency Across Regions",
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig3, use_container_width=True)






# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


os.path.abspath('final.py')


# In[1]:


import os
import pandas as pd
import random

# Path to your dataset
train_path = r"C:\Users\DELL\Downloads\archive\Training"
test_path = r"C:\Users\DELL\Downloads\archive\Testing"

# Define tumor classes
tumor_classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# Create lists to store data
data = []

# Gender and Region options
genders = ['Male', 'Female']
regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Australia']

# Helper to generate random patient info
def generate_patient_info():
    gender = random.choice(genders)
    region = random.choice(regions)
    age = random.randint(5, 80)  # Age between 5 and 80
    return gender, region, age

# Assign unique Patient IDs
patient_counter = 1

# Scan both training and testing datasets
for base_path in [train_path, test_path]:
    for tumor_type in tumor_classes:
        folder_path = os.path.join(base_path, tumor_type)
        if not os.path.exists(folder_path):
            continue  # Skip missing folders
        
        for img_name in os.listdir(folder_path):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                gender, region, age = generate_patient_info()
                patient_id = f"P{str(patient_counter).zfill(5)}"
                data.append([patient_id, gender, region, age, tumor_type, os.path.join(folder_path, img_name)])
                patient_counter += 1

# Create DataFrame
df = pd.DataFrame(data, columns=['PatientID', 'Gender', 'Region', 'Age', 'TumorType', 'ImagePath'])

# Save to CSV
df.to_csv("brain_tumor_metadata.csv", index=False)
print("✅ Dummy metadata generated and saved to brain_tumor_metadata.csv")
print(df)


# In[ ]:




