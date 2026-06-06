import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🏠 House Price Prediction App")

# Load Dataset
df = pd.read_csv("Housing.csv")

# Select important columns
df = df[["area", "bedrooms", "bathrooms","price"]]
df = df.dropna()
#show data
st.subheader("Dataset Preview")
st.write(df.head())
# Features & Target
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (sqft)", min_value=500, max_value=10000, value=2000)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# Predict button
if st.button("Predict Price"):
    user_data = pd.DataFrame([[area, bedrooms, bathrooms]],
                             columns=["area", "bedrooms", "bathrooms"])

    prediction = model.predict(user_data)

    st.success(f"🏠 Predicted Price: ₹ {round(prediction[0], 2)}")
st.subheader("area vs price")
st.line_chart(df[['area','price']])