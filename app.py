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

st.title("🚗 Road Accident Severity Prediction")
st.subheader("Machine Learning Based Accident Severity Analysis")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    ["Home", "Dataset Analysis", "Severity Prediction", "Model Evaluation"]
)

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

            # Missing values chart
            missing_nonzero = missing[missing > 0]
            if not missing_nonzero.empty:
                st.subheader("Missing Values Chart")
                st.bar_chart(missing_nonzero)
            else:
                st.success("No missing values found in the dataset.")

            # Accident severity distribution
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
                    "severity class. The dataset uses its original labels."
                )
            else:
                st.info(
                    "The column 'accident_severity_c' was not found. "
                    "Available columns are shown below."
                )
                st.write(list(data.columns))

            # Numerical feature distributions
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
        return (
            joblib.load("road_accident_xgboost_v2.pkl"),
            joblib.load("severity_label_encoder_v2.pkl")
        )

    try:
        model, encoder = load_prediction_assets()

        with st.form("severity_prediction_form"):
            st.subheader("Accident Details")
            left, right = st.columns(2)

            with left:
                accident_date = st.date_input("Date", value=date.today())
                day = st.number_input("Day of Week code", value=1, step=1)
                accident_time = st.time_input(
                    "Time of Accident", value=time(12, 0)
                )
                location = st.number_input(
                    "Accident Location A code", value=1, step=1
                )
                chainage = st.text_input("Location Chainage km")
                roadside = st.text_input("Roadside Chainage")
                nature1 = st.text_input("Nature B1 code")
                nature2 = st.text_input("Nature B2 code")
                nature3 = st.number_input("Nature B3 code", value=1, step=1)
                class1 = st.number_input("Classification C1 code", value=1, step=1)
                class2 = st.number_input("Classification C2 code", value=1, step=1)
                class3 = st.number_input("Classification C3 code", value=1, step=1)

            with right:
                cause1 = st.text_input("Cause D1 code")
                cause2 = st.text_input("Cause D2 code")
                cause3 = st.text_input("Cause D3 code")
                cause4 = st.text_input("Cause D4 code")
                road_feature = st.number_input("Road Feature E code", value=1, step=1)
                road_condition = st.number_input("Road Condition F code", value=1, step=1)
                intersection = st.number_input("Intersection Type G code", value=1, step=1)
                weather = st.number_input("Weather Conditions H code", value=1, step=1)
                vehicle1 = st.number_input("Vehicle Type V1 code", value=1, step=1)
                vehicle2 = st.number_input("Vehicle Type V2 code", value=0, step=1)
                vehicle3 = st.number_input("Vehicle Type V3 code", value=0, step=1)
                vehicle4 = st.number_input("Vehicle Type V4 code", value=0, step=1)
                vehicles = st.number_input(
                    "Number of Vehicles", min_value=1, value=1, step=1
                )

            submitted = st.form_submit_button("Predict Severity Class")

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
                input_df = input_df[model.feature_names_in_]
                predicted = model.predict(input_df)[0]
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



elif page == "Model Evaluation":
    st.header("Model Evaluation and Comparison")
    st.write(
        "Comparison of five machine learning algorithms across "
        "the before-accident and after-accident experiments."
    )

    try:
        results = pd.read_csv("two_experiment_model_comparison.csv")

        required = {
            "Experiment", "Model",
            "Accuracy (%)", "Macro F1-score (%)"
        }

        if not required.issubset(results.columns):
            st.error("The comparison CSV is missing required columns.")
        else:
            st.subheader("All Model Results")
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            # Accuracy comparison
            st.subheader("Accuracy Comparison")
            accuracy_chart = results.pivot(
                index="Model",
                columns="Experiment",
                values="Accuracy (%)"
            )
            st.bar_chart(accuracy_chart)

            # Macro F1 comparison
            st.subheader("Macro F1-score Comparison")
            f1_chart = results.pivot(
                index="Model",
                columns="Experiment",
                values="Macro F1-score (%)"
            )
            st.bar_chart(f1_chart)

            # Best model by Macro F1 in each experiment
            st.subheader("Best Model in Each Experiment")
            best_indices = results.groupby(
                "Experiment"
            )["Macro F1-score (%)"].idxmax()

            best_results = results.loc[best_indices].reset_index(drop=True)
            st.dataframe(
                best_results,
                use_container_width=True,
                hide_index=True
            )

            for _, row in best_results.iterrows():
                st.info(
                    f"{row['Experiment']}: {row['Model']} — "
                    f"Accuracy {row['Accuracy (%)']:.2f}%, "
                    f"Macro F1-score {row['Macro F1-score (%)']:.2f}%"
                )

            st.warning(
                "These results use a single hold-out split and are preliminary. "
                "The before-accident experiment classifies severity among "
                "recorded accidents; it is not a validated accident-risk "
                "prediction model. Feature availability and data leakage "
                "must be reviewed before drawing final conclusions."
            )

    except Exception as e:
        st.error(f"Could not load model comparison results: {e}")
