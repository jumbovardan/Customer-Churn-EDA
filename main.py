# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("group2_dataset.csv")
    # Basic cleaning
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Handle TotalCharges blanks
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode categorical
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})

    return df

df = load_data()

# Title
st.title("📊 Customer Churn Dashboard")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churned", f"{df['Churn'].sum():,}")
col3.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")

st.markdown("---")

# Data preview
with st.expander("Preview Dataset"):
    st.dataframe(df.head(10))

# ---- Graphs ----

# 1. Churn distribution
fig1, ax1 = plt.subplots(figsize=(4, 4))
churn_val = df['Churn'].value_counts()
ax1.pie(churn_val, labels=['No', 'Yes'], autopct='%1.2f%%', colors=['green', 'red'])
ax1.set_title("Churn Distribution", fontsize=12)
st.pyplot(fig1)

# 2. Churn by demographics
demographics = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
for col in demographics:
    fig, ax = plt.subplots(figsize=(5, 3))
    churn_rate = df.groupby([col, 'Churn']).size().reset_index(name='count')
    sns.barplot(x=col, y='count', hue='Churn', data=churn_rate, palette=['green', 'red'], ax=ax)
    ax.set_title(f"Churn Distribution by {col}", fontsize=12)
    st.pyplot(fig)

# 3. Churn by Internet Service
fig, ax = plt.subplots(figsize=(5, 3))
sns.countplot(data=df, x='InternetService', hue='Churn', palette=['green', 'red'], ax=ax)
ax.set_title("Churn Distribution by Internet Service", fontsize=12)
st.pyplot(fig)

# 4. Churn rate by services
services = ['StreamingTV', 'StreamingMovies', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract']
for service in services:
    fig, ax = plt.subplots(figsize=(5, 3))
    service_churn = df.groupby(service)['Churn'].mean().reset_index()
    service_churn['Churn_rate_%'] = service_churn['Churn'] * 100
    sns.barplot(x=service, y='Churn_rate_%', data=service_churn, ax=ax, palette="Reds")
    ax.set_title(f"Churn Rate by {service}", fontsize=12)
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)

# 5. Churn by Contract
fig, ax = plt.subplots(figsize=(5, 3))
sns.countplot(x='Contract', hue='Churn', data=df, palette=['green', 'red'], ax=ax)
ax.set_title("Churn by Contract Type", fontsize=12)
st.pyplot(fig)

# 6. Churn by Payment Method
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette=['green', 'red'], ax=ax)
plt.setp(ax.get_xticklabels(), rotation=45)
ax.set_title("Churn by Payment Method", fontsize=12)
st.pyplot(fig)

# 7. KDE plots
def kdeplot(feature):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.kdeplot(df[df['Churn'] == 0][feature], color='green', label='No', fill=True, ax=ax)
    sns.kdeplot(df[df['Churn'] == 1][feature], color='red', label='Yes', fill=True, ax=ax)
    ax.set_title(f"{feature} distribution by Churn", fontsize=12)
    ax.legend()
    st.pyplot(fig)

for feat in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    kdeplot(feat)

# 8. Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df[['tenure','MonthlyCharges','TotalCharges','Churn']].corr(),
            annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap", fontsize=12)
st.pyplot(fig)
