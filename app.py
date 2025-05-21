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


def train_model(file_path):
    df = pd.read_csv(file_path)

    # Assuming last column is label, rest EEG data
    X_raw = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_processed = preprocess_eeg(X_raw)

    # Extract features for all samples (row-wise)
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
    print(f"Model trained. Accuracy: {acc * 100:.2f}%")

    # Save model
    joblib.dump(model, 'eeg_workload_model.pkl')
    return model


def load_model():
    return joblib.load('eeg_workload_model.pkl')


def predict_workload(model, df):
    preprocessed = preprocess_eeg(df)
    features = extract_features(preprocessed)
    pred = model.predict(features)[0]
    return pred


# ======== Streamlit UI ========

def run_app():
    st.title("EEG Mental Workload Classifier")

    uploaded_file = st.file_uploader("Upload training CSV with labels to train model OR upload EEG CSV for prediction",
                                     type=["csv"])
    model = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Check if last column looks like labels (integer or categorical)
        if df.shape[1] > 1 and (df.iloc[:, -1].dtype == 'int64' or df.iloc[:, -1].dtype == 'object'):
            st.write("Detected labels column, training model...")
            model = train_model(uploaded_file)
            st.success("Model trained and saved!")
        else:
            try:
                model = load_model()
            except Exception as e:
                st.error("Model not found. Please upload labeled training data first.")
                return

            st.write("Predicting mental workload...")
            pred = predict_workload(model, df)
            label_map = {0: "Low Workload", 1: "Medium Workload", 2: "High Workload"}
            st.success(f"Predicted Mental Workload: **{label_map.get(pred, 'Unknown')}**")


if __name__ == "__main__":
    run_app()
