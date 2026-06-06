import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

# Basic analysis
print(df.head())
print(df.describe())

# Visualization
sns.histplot(df["price"], kde=True)
plt.show()

sns.scatterplot(x=df["area"], y=df["price"])
plt.show()

# Select features
df = df[["area", "bedrooms", "bathrooms", "price"]]
df = df.dropna()

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model R2 Score:", r2_score(y_test, y_pred))

# User Input
area = float(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))


user_data = pd.DataFrame([[area, bedrooms, bathrooms]],
                         columns=["area", "bedrooms", "bathrooms"])

prediction = model.predict(user_data)

print("🏠 Predicted Price:", round(prediction[0], 2))