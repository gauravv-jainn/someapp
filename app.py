"""
Streamlit deployment assignment: interactive wine-classifier demo.
Trains a RandomForest on the sklearn wine dataset once (cached), then lets the
user tweak feature sliders and see the live prediction + class probabilities.
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
st.set_page_config(page_title="Wine Classifier - Streamlit Deployment Demo", layout="wide")
 
st.title("🍷 Wine Cultivar Classifier — Streamlit Deployment Assignment")
st.caption("Self-learning assignment: deploy a trained scikit-learn model behind an interactive Streamlit UI.")
 
@st.cache_resource
def load_model():
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return model, data, test_acc
 
model, data, test_acc = load_model()
feature_names = data.feature_names
target_names = data.target_names
 
st.sidebar.header("Input wine chemistry features")
st.sidebar.write(f"Model test accuracy: **{test_acc:.2%}**")
 
default_sample = data.data[0]
inputs = {}
col_ranges = {f: (float(data.data[:, i].min()), float(data.data[:, i].max())) for i, f in enumerate(feature_names)}
 
for i, f in enumerate(feature_names):
    lo, hi = col_ranges[f]
    inputs[f] = st.sidebar.slider(f, lo, hi, float(default_sample[i]))
 
X_input = np.array([[inputs[f] for f in feature_names]])
pred_class = model.predict(X_input)[0]
pred_proba = model.predict_proba(X_input)[0]
 
col1, col2 = st.columns([1, 1.4])
 
with col1:
    st.subheader("Prediction")
    st.metric("Predicted cultivar", target_names[pred_class])
    st.write("Class probabilities:")
    proba_df = pd.DataFrame({"cultivar": target_names, "probability": pred_proba}).set_index("cultivar")
    st.bar_chart(proba_df)
 
with col2:
    st.subheader("Feature importances (trained model)")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    st.bar_chart(importances)
 
st.subheader("Current input vector")
st.dataframe(pd.DataFrame([inputs]))
 
st.divider()
st.caption("Model: RandomForestClassifier(n_estimators=200) on sklearn's wine dataset (178 samples, 13 features, 3 cultivars).")
 
