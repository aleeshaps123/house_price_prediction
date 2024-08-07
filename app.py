import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Custom CSS for the Streamlit app with background image
st.markdown("""
    <style>
    .stApp {
        background: url("https://www.shutterstock.com/image-photo/photo-blur-housing-estate-260nw-262512779.jpg") no-repeat center center fixed;
        background-size: cover;
        color: #333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stNumberInput input {
        width: 100%;
        padding: 8px;
        box-sizing: border-box;
        border: 2px solid #ccc;
        border-radius: 4px;
        background-color: #f8f8f8;
        resize: none;
    }
    .stSelectbox div {
        background-color: #f8f8f8;
        border: 2px solid #ccc;
        border-radius: 4px;
        padding: 8px;
    }
    .stTitle h1 {
        font-family: 'Arial', sans-serif;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Convert lot_size from acres to sqft where applicable
data.loc[data['lot_size_units'] == 'acre', 'lot_size'] *= 43560  # 1 acre = 43560 sqft
data['lot_size_units'] = 'sqft'  # Set all lot_size_units to sqft

# Fill missing lot_size values with the mean lot_size
mean_lot_size = data['lot_size'].mean()
data['lot_size'].fillna(mean_lot_size, inplace=True)

# Dropping lot_size_units column as it's now redundant
data.drop(columns=['lot_size_units'], inplace=True)

# Convert categorical features to numerical values using OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_size_units = encoder.fit_transform(data[['size_units']])

# Creating a DataFrame with the encoded size_units
encoded_size_units_df = pd.DataFrame(encoded_size_units, columns=encoder.get_feature_names_out(['size_units']))

# Concatenate the original data with the encoded columns and drop the original size_units column
data_encoded = pd.concat([data.drop(columns=['size_units']), encoded_size_units_df], axis=1)

# Feature Selection
X = data_encoded.drop(columns=['price'])
y = data_encoded['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict house price based on user input
def predict_house_price(beds, baths, size, size_units='sqft'):
    # Prepare the input data
    input_data = {
        'beds': [beds],
        'baths': [baths],
        'size': [size],
        'lot_size': [mean_lot_size],  # Using mean lot size as default
        'zip_code': [98101]  # Using a default zip code, you can modify as needed
    }
    input_df = pd.DataFrame(input_data)
    
    # Encode the size_units
    encoded_input_size_units = encoder.transform([[size_units]])
    encoded_input_size_units_df = pd.DataFrame(encoded_input_size_units, columns=encoder.get_feature_names_out(['size_units']))
    
    # Concatenate the input data with the encoded size_units
    input_df_encoded = pd.concat([input_df, encoded_input_size_units_df], axis=1)
    
    # Predict the house price
    predicted_price = model.predict(input_df_encoded)
    return predicted_price[0]

# Streamlit interface
st.title('House Price Prediction App')

# Input fields
beds = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
baths = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
size = st.number_input('Square Feet', min_value=500, max_value=10000, value=1500)
size_units = st.selectbox('Size Units', options=['sqft', 'acre'])

# Predict button
if st.button('Predict'):
    predicted_price = predict_house_price(beds, baths, size, size_units)
    st.write(f'The predicted price for a house with {beds} bedrooms, {baths} bathrooms, and {size} {size_units} is RS {predicted_price:.2f}')
