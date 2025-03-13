import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# Streamlit App
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("📈 Sales Prediction Web App")
st.markdown("## Predict Sales with Ease and Visualize Key Insights")

# Load Model and Data
dataset = pd.read_csv("./datasets/modified_sales_data.csv")  # Add your dataset file
model = joblib.load(
    "./saved_models/Sales_Prediction_Model.pkl"
)  # Update path if needed
encoder = joblib.load("./saved_models/encoders.pkl")  # Update path if needed
scaler = joblib.load("./saved_models/scalers.pkl")  # Update path if needed

with st.sidebar:
    selected_option = option_menu(
        "Sales Prediction System",
        ["Predict Sales", "Dataset Overview", "Visualization"],
        menu_icon="building-fill",
        icons=["calculator-fill", "book-fill", "bar-chart-fill"],
        default_index=0,
    )

# User input fields
user_inputs = {}

# Home Section - Dataset Overview & Sales Prediction
if selected_option == "Predict Sales":
    st.header("💡 Predict Sales")
    st.markdown("Enter the required inputs to get an accurate sales prediction.")

    for col in dataset.columns[:-1]:
        if dataset[col].dtype == "object":
            unique_values = dataset[col].unique().tolist()
            user_inputs[col] = st.selectbox(f"Select {col}", unique_values)
        else:
            user_inputs[col] = st.number_input(f"Enter {col}", value=0.0)

    # Convert user inputs to DataFrame
    input_dataset = pd.DataFrame([user_inputs])
    st.write(input_dataset)

    numerical_cols = (
        input_dataset.select_dtypes(exclude=["object"]).columns[:-1].tolist()
    )
    categorical_cols = input_dataset.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        input_dataset[col] = encoder[col].transform(input_dataset[col])

    # for col in numerical_cols:
    input_dataset = scaler.transform(input_dataset)

    # Predict Button
    if st.button("🔮 Predict Sales"):
        prediction = model.predict(
            input_dataset
        )  # Ensure input matches model training format
        st.success(f"📊 Predicted Sales: **{prediction[0]:.2f}**")


elif selected_option == "Dataset Overview":
    st.header("📊 Dataset Overview")

    # Show dataset
    st.write(dataset.head(10))

    # Dataset Information
    st.subheader("📌 Dataset Details")
    st.write(f"**Dataset Shape:** {dataset.shape}")
    st.write(f"**🔹 Number of Rows:** {dataset.shape[0]}")
    st.write(f"**🔹 Number of Columns:** {dataset.shape[1]}")

elif selected_option == "Visualization":
    st.header("📊 Data Visualizations")

    # # Correlation Heatmap
    num_cols = dataset.select_dtypes(exclude=["object"]).columns[1:]
    st.subheader("🔍 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(dataset[num_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

    # Sales Trend (Modify column names as per your dataset)
    if "Date" in dataset.columns:
        st.subheader("📈 Sales Trend Over Time")
        dataset["Date"] = pd.to_datetime(
            dataset["Date"]
        )  # Convert to datetime if not already
        dataset = dataset.sort_values("Date")  # Ensure data is sorted by date
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=dataset["Date"], y=dataset["Sales"], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Histogram for numerical columns
    st.subheader("📊 Feature Distribution (Histogram)")
    cols = dataset.columns[-1:]
    selected_feature = st.selectbox("Select a numerical feature", cols)

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.histplot(dataset[selected_feature], kde=True, bins=30, ax=ax)
    st.pyplot(fig)

    # Bar Chart for categorical data
    cat_cols = dataset.select_dtypes(include=["object"]).columns[1:]
    if len(cat_cols) > 0:
        st.subheader("📊 Bar Chart (Categorical Feature Distribution)")
        selected_cat_feature = st.selectbox("Select a categorical feature", cat_cols)
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.countplot(x=dataset[selected_cat_feature], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)


# # Sidebar for User Input
# st.sidebar.header("Input Features")
# feature1 = st.sidebar.number_input("Feature 1", min_value=0.0)
# feature2 = st.sidebar.number_input("Feature 2", min_value=0.0)

# # Prediction Button
# if st.sidebar.button("Predict Sales"):
#     prediction = model.predict([[feature1, feature2]])
#     st.sidebar.success(f"Predicted Sales: ${prediction[0]:,.2f}")

# # Data Overview
# st.subheader("Dataset Overview")
# st.dataframe(dataset.head(10))

# # # Visualization
# # st.subheader("Sales Distribution")
# # fig, ax = plt.subplots()
# # sns.histplot(dataset.iloc[:, -1], bins=20, kde=True, ax=ax)
# # st.pyplot(fig)

# # st.subheader("Feature Correlation Heatmap")
# # fig, ax = plt.subplots()
# # sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', ax=ax)
# # st.pyplot(fig)

# # Sales Trend (if time-series data is available)
# if 'Date' in dataset.columns:
#     st.subheader("Sales Trend Over Time")
#     dataset['Date'] = pd.to_datetime(dataset['Date'])
#     dataset.set_index('Date', inplace=True)
#     st.line_chart(dataset['Sales'])

# st.markdown("---")
# st.markdown("**Made with ❤️ using Streamlit**")
