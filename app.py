import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and icon
st.set_page_config(page_title="Mobile Price Prediction", page_icon="📱")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .header-text {
        font-size: 24px;
        color: #009688;
        padding: 20px;
        text-align: center;
    }
    .sidebar-text {
        font-size: 18px;
        color: #424242;
        padding-top: 20px;
    }
    .sidebar {
        background-color: #f0f0f0;
    }
    .main-content {
        padding: 20px;
    }
    .btn {
        background-color: #009688;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .btn:hover {
        background-color: #00796b;
    }
    .output {
        font-size: 50px;
        font-weight: 700
    }
    .price-range {
        font-size: 20px;
        font-weight: 700
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
@st.cache_data()
def load_data():
    return pd.read_csv("C:\\Users\\farid\\OneDrive\\Desktop\\aa\\Mobile (2).csv")  # Replace 'mobile_data.csv' with your dataset path

data = load_data()

# Sidebar
st.sidebar.title('Mobile Price Prediction')
st.sidebar.markdown("Select the features to predict the price.")

# Main content
st.title('Mobile Price Prediction')
st.markdown("This app predicts the price of mobile phones based on their features.")

# Display the dataset
if st.checkbox('Show Dataset'):
    st.dataframe(data.head())

# Data preprocessing
X = data[['battery_power', 'ram', 'int_memory']]  # Use only 3 features for prediction
y = data['price_range']  # Change to 'price'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeRegressor()  # Change to DecisionTreeRegressor
model.fit(X_train, y_train)

# Prediction
st.header('Make a Prediction')
battery_power = st.slider('Battery Power', min_value=500, max_value=7000, step=100)
ram = st.slider('RAM', min_value=0, max_value=8000, step=100)
int_memory = st.slider('Internal Memory', min_value=0, max_value=256, step=4)

# Prediction with only 3 features
prediction = model.predict([[battery_power, ram, int_memory]])[0]

# Define price ranges
price_ranges = {
    0: 'Less than ₹ 10000',
    1: '₹ 10000 - 20000',
    2: '₹ 20000 - 30000',
    3: 'More than ₹ 30000',
}

# Determine price range based on predicted price
predicted_range = ''
for price, range_str in price_ranges.items():
    if prediction <= price:
        predicted_range = range_str
        break

# Display predicted price and price range with larger font size
st.write('<span class="output" style="font-size: 20px;">Predicted Price: </span>', prediction, unsafe_allow_html=True)  # Change to 'Predicted Price'

st.write('<span class="price-range">Predicted Price Range: </span>', predicted_range, unsafe_allow_html=True)

# Plot feature importance (not necessary for regression, but can still be informative)
st.header('Feature Importance')
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=feature_importance.index, palette="viridis")
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Importance")
st.pyplot(plt)
