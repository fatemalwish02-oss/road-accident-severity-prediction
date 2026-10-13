
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOGIN AUTHENTICATION ----------------
import streamlit_authenticator as stauth

# Convert Streamlit Secrets into normal Python dictionaries.
# This fixes: TypeError: Secrets does not support item assignment.
credentials = {
    "usernames": {
        username: dict(user_data)
        for username, user_data
        in st.secrets["credentials"]["usernames"].items()
    }
}

authenticator = stauth.Authenticate(
    credentials,
    st.secrets["cookie"]["name"],
    st.secrets["cookie"]["key"],
    st.secrets["cookie"]["expiry_days"],
)

authenticator.login(location="main")

if st.session_state.get("authentication_status") is not True:
    st.title("🚗 Road Accident Severity Prediction")
    st.subheader("Please log in to access the dashboard")
    st.stop()

authenticator.logout(location="sidebar")

# ---------------- DASHBOARD HEADER ----------------
st.title("🚗 Road Accident Severity Prediction")
st.subheader("Machine Learning Based Accident Severity Analysis")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    ["Home", "Dataset Analysis", "Severity Prediction", "Model Evaluation"]
)

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.header("Project Overview")

    st.write("""
    This project uses Machine Learning to classify road accident
    severity using historical accident records.

    The project includes:
    - Data cleaning and preprocessing
    - Exploratory Data Analysis
    - Machine Learning model comparison
    - Accident severity prediction
    - Model evaluation
    """)

    st.info("Selected Model: XGBoost")

    st.warning(
        "The model evaluation results are preliminary and "
        "must be validated before final reporting."
    )

# ---------------- DATASET ANALYSIS PAGE ----------------
elif page == "Dataset Analysis":
    st.header("Accident Dataset Analysis")

    uploaded_file = st.file_uploader(
        "Upload your accident dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            col1, col2 = st.columns(2)
            col1.metric("Total Records", len(data))
            col2.metric("Total Columns", len(data.columns))

            st.subheader("Dataset Preview")
            st.dataframe(data.head(20), use_container_width=True)

            st.subheader("Missing Values")
            missing = data.isnull().sum().sort_values(ascending=False)
            st.dataframe(missing.rename("Missing Values"))

            st.subheader("Numerical Summary")
            st.dataframe(data.describe(), use_container_width=True)

            missing_nonzero = missing[missing > 0]

            if not missing_nonzero.empty:
                st.subheader("Missing Values Chart")
                st.bar_chart(missing_nonzero)
            else:
                st.success("No missing values found in the dataset.")

            target = "Accident_Severity_C"

            if target in data.columns:
                st.subheader("Accident Severity Distribution")

                severity_counts = (
                    data[target].astype(str)
                    .value_counts()
                    .sort_index()
                )

                st.bar_chart(severity_counts)

                st.caption(
                    "This chart shows the number of records in each "
                    "severity class using the original dataset labels."
                )
            else:
                st.info(
                    "The column 'Accident_Severity_C' was not found. "
                    "Available columns are shown below."
                )
                st.write(list(data.columns))

            numeric_cols = data.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_cols:
                st.subheader("Numerical Feature Distribution")

                selected_col = st.selectbox(
                    "Select a numerical feature",
                    numeric_cols
                )

                fig, ax = plt.subplots(figsize=(8, 4))

                sns.histplot(
                    data[selected_col].dropna(),
                    bins=20,
                    kde=False,
                    ax=ax
                )

                ax.set_title(f"Distribution of {selected_col}")
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Number of records")

                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(f"Could not read the dataset: {e}")

# ---------------- SEVERITY PREDICTION PAGE ----------------
elif page == "Severity Prediction":
    import joblib
    from datetime import date, time

    st.header("Accident Severity Prediction")

    st.info(
        "Enter accident details using the dataset's coded values. "
        "This is a preliminary model demonstration."
    )

    @st.cache_resource
    def load_prediction_assets():
        model = joblib.load("road_accident_xgboost_v2.pkl")
        encoder = joblib.load("severity_label_encoder_v2.pkl")
        return model, encoder

    try:
        model, encoder = load_prediction_assets()

        with st.form("severity_prediction_form"):
            st.subheader("Accident Details")

            left, right = st.columns(2)

            with left:
                accident_date = st.date_input(
                    "Date",
                    value=date.today()
                )

                day = st.number_input(
                    "Day of Week code",
                    value=1,
                    step=1
                )

                accident_time = st.time_input(
                    "Time of Accident",
                    value=time(12, 0)
                )

                location = st.number_input(
                    "Accident Location A code",
                    value=1,
                    step=1
                )

                chainage = st.text_input("Location Chainage km")
                roadside = st.text_input("Roadside Chainage")
                nature1 = st.text_input("Nature B1 code")
                nature2 = st.text_input("Nature B2 code")

                nature3 = st.number_input(
                    "Nature B3 code",
                    value=1,
                    step=1
                )

                class1 = st.number_input(
                    "Classification C1 code",
                    value=1,
                    step=1
                )

                class2 = st.number_input(
                    "Classification C2 code",
                    value=1,
                    step=1
                )

                class3 = st.number_input(
                    "Classification C3 code",
                    value=1,
                    step=1
                )

            with right:
                cause1 = st.text_input("Cause D1 code")
                cause2 = st.text_input("Cause D2 code")
                cause3 = st.text_input("Cause D3 code")
                cause4 = st.text_input("Cause D4 code")

                road_feature = st.number_input(
                    "Road Feature E code",
                    value=1,
                    step=1
                )

                road_condition = st.number_input(
                    "Road Condition F code",
                    value=1,
                    step=1
                )

                intersection = st.number_input(
                    "Intersection Type G code",
                    value=1,
                    step=1
                )

                weather = st.number_input(
                    "Weather Conditions H code",
                    value=1,
                    step=1
                )

                vehicle1 = st.number_input(
                    "Vehicle Type V1 code",
                    value=1,
                    step=1
                )

                vehicle2 = st.number_input(
                    "Vehicle Type V2 code",
                    value=0,
                    step=1
                )

                vehicle3 = st.number_input(
                    "Vehicle Type V3 code",
                    value=0,
                    step=1
                )

                vehicle4 = st.number_input(
                    "Vehicle Type V4 code",
                    value=0,
                    step=1
                )

                vehicles = st.number_input(
                    "Number of Vehicles",
                    min_value=1,
                    value=1,
                    step=1
                )

            submitted = st.form_submit_button(
                "Predict Severity Class"
            )

        if submitted:
            row = {
                "Date": str(accident_date),
                "Day_of_Week": day,
                "Time_of_Accident": accident_time.strftime("%H:%M"),
                "Accident_Location_A": location,
                "Accident_Location_A_Chainage_km": chainage,
                "Accident_Location_A_Chainage_km_RoadSide": roadside,
                "Nature_of_Accident_B1": nature1,
                "Nature_of_Accident_B2": nature2,
                "Nature_of_Accident_B3": nature3,
                "Classification_of_Accident_C1": class1,
                "Classification_of_Accident_C2": class2,
                "Classification_of_Accident_C3": class3,
                "Causes_D1": cause1,
                "Causes_D2": cause2,
                "Causes_D3": cause3,
                "Causes_D4": cause4,
                "Road_Feature_E": road_feature,
                "Road_Condition_F": road_condition,
                "Intersection_Type_G": intersection,
                "Weather_Conditions_H": weather,
                "Vehicle_Type_Involved_J_V1": vehicle1,
                "Vehicle_Type_Involved_J_V2": vehicle2,
                "Vehicle_Type_Involved_J_V3": vehicle3,
                "Vehicle_Type_Involved_J_V4": vehicle4,
                "Number_of_Vehicles": vehicles
            }

            try:
                input_df = pd.DataFrame([row])

                # Use the feature columns expected by the trained model.
                input_df = input_df[model.feature_names_in_]

                predicted = model.predict(input_df)[0]

                # Convert the model prediction back to its original label.
                label = encoder.inverse_transform([predicted])[0]

                st.success(f"Predicted severity class: {label}")

                st.warning(
                    "Preliminary demonstration only. Confirm feature codes "
                    "and validate the model before interpreting this result."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    except Exception as error:
        st.error(f"Could not load model files: {error}")

# ---------------- MODEL EVALUATION PAGE ----------------
elif page == "Model Evaluation":
    import joblib

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        precision_score,
        recall_score,
        f1_score
    )

    st.header("Model Evaluation")

    st.write(
        "Evaluation of the revised XGBoost model using a held-out test set."
    )

    @st.cache_resource
    def evaluate_model():
        data = pd.read_csv("Accdataset_hk_PS_BAEL_Combined.csv")

        target = "Accident_Severity_C"

        excluded = [
            "Sl No",
            "Accident_Index",
            target,
            "Number_of_Casualties_Fatel",
            "Number_of_Casualties_GrievousInjury",
            "Number_of_Casualties_MinorInjury",
            "Number_of_Casualties_NotInjured",
            "Number_of_Casualties"
        ]

        X = data.drop(
            columns=excluded,
            errors="ignore"
        ).copy()

        y = data[target].astype(str)

        constant_cols = [
            col for col in X.columns
            if X[col].nunique(dropna=False) <= 1
        ]

        X = X.drop(columns=constant_cols)

        for col in ["Date", "Time_of_Accident"]:
            if col in X.columns:
                X[col] = X[col].astype("string")

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        model = joblib.load("road_accident_xgboost_v2.pkl")
        encoder = joblib.load("severity_label_encoder_v2.pkl")

        y_test_encoded = encoder.transform(y_test)

        predictions = model.predict(
            X_test[model.feature_names_in_]
        )

        return y_test_encoded, predictions, encoder

    try:
        y_true, y_pred, encoder = evaluate_model()

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{accuracy:.2%}")
        c2.metric("Macro Precision", f"{precision:.2%}")
        c3.metric("Macro Recall", f"{recall:.2%}")
        c4.metric("Macro F1-score", f"{f1:.2%}")

        st.subheader("Confusion Matrix")

        labels = np.arange(len(encoder.classes_))

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("XGBoost Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[str(c) for c in encoder.classes_],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Model Comparison")

        st.caption(
            "Only the revised XGBoost result is recomputed here. "
            "Other models must be evaluated on the same split "
            "before a fair comparison can be shown."
        )

        comparison = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1"
            ],
            "Score": [
                accuracy,
                precision,
                recall,
                f1
            ]
        }).set_index("Metric")

        st.bar_chart(comparison)

        st.warning(
            "These are preliminary results from a single hold-out split. "
            "Feature leakage and feature availability still need review. "
            "Do not treat the scores as validated real-world performance."
        )

    except Exception as e:
        st.error(f"Could not evaluate the model: {e}")
