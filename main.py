import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

# ======== Functions ========

def preprocess_eeg(df):
    numeric_df = df.select_dtypes(include=['number'])
    df_corrected = numeric_df - numeric_df.mean()
    return df_corrected

def extract_features(df):
    features = []
    for col in df.columns:
        features.extend([
            df[col].mean(),
            df[col].std(),
            df[col].min(),
            df[col].max(),
        ])
    return np.array(features).reshape(1, -1)

def train_model(df):
    # Assume last column is label, rest EEG data
    X_raw = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_processed = preprocess_eeg(X_raw)

    # Extract features for all samples
    feature_list = []
    for i in range(X_processed.shape[0]):
        sample = X_processed.iloc[[i]]
        feature_vector = extract_features(sample)
        feature_list.append(feature_vector.flatten())

    X_features = np.array(feature_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(model, 'eeg_workload_model.pkl')
    return model, acc

def load_model():
    return joblib.load('eeg_workload_model.pkl')

def predict_workload(model, df):
    preprocessed = preprocess_eeg(df)
    features = extract_features(preprocessed)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return pred, proba

def label_mapping():
    return {0: "Low Workload", 1: "Medium Workload", 2: "High Workload"}

# ======== Streamlit UI ========

def run_app():
    st.title("🧠 EEG Mental Workload Classifier")

    tabs = st.tabs(["Train Model", "Predict Workload", "Model Management", "Data Explorer"])

    with tabs[0]:
        st.header("Train Model")
        st.write("Upload a labeled EEG CSV file to train the workload classifier model.")
        train_file = st.file_uploader("Upload labeled training CSV", type=["csv"], key="train")
        if train_file is not None:
            try:
                df_train = pd.read_csv(train_file)
                st.subheader("Training Data Preview")
                st.dataframe(df_train.head())

                if 'label' not in df_train.columns:
                    st.warning("No 'label' column found. Please upload labeled data with 'label' column.")
                else:
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            model, accuracy = train_model(df_train)
                        st.success(f"Model trained successfully! Accuracy on test set: {accuracy*100:.2f}%")
                        st.write("Model saved as `eeg_workload_model.pkl`")

            except Exception as e:
                st.error(f"Error reading training file: {e}")

    with tabs[1]:
        st.header("Predict Workload")
        st.write("Upload unlabeled EEG CSV file to predict mental workload.")
        pred_file = st.file_uploader("Upload EEG CSV for prediction", type=["csv"], key="predict")

        if pred_file is not None:
            try:
                df_pred = pd.read_csv(pred_file)
                st.subheader("Prediction Data Preview")
                st.dataframe(df_pred.head())

                try:
                    model = load_model()
                except Exception:
                    st.error("No trained model found. Please train a model first in the 'Train Model' tab.")
                    return

                if st.button("Predict"):
                    with st.spinner("Running prediction..."):
                        pred_label, pred_proba = predict_workload(model, df_pred)
                    labels = label_mapping()
                    st.success(f"Predicted Mental Workload: **{labels.get(pred_label, 'Unknown')}**")
                    st.write("Prediction probabilities:")
                    prob_df = pd.DataFrame({
                        "Workload Level": [labels[i] for i in range(len(pred_proba))],
                        "Probability": pred_proba
                    })
                    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

            except Exception as e:
                st.error(f"Error reading prediction file: {e}")

    with tabs[2]:
        st.header("Model Management")
        st.write("Manage your saved EEG workload classification model.")
        if st.button("Load & Show Model Details"):
            try:
                model = load_model()
                st.success("Model loaded successfully!")
                st.write(model)
            except Exception:
                st.error("No saved model found. Please train a model first.")

        if st.button("Delete Saved Model"):
            import os
            try:
                os.remove('eeg_workload_model.pkl')
                st.success("Saved model deleted.")
            except Exception as e:
                st.error(f"Error deleting model: {e}")

    with tabs[3]:
        st.header("Data Explorer")
        uploaded = st.file_uploader("Upload any EEG CSV to explore", type=["csv"], key="explorer")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.subheader("Data Preview")
                st.dataframe(df.head())

                st.subheader("Data Description (numeric columns)")
                st.write(df.describe())

                st.subheader("Correlation Heatmap")
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    run_app()
