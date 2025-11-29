import streamlit as st
import pandas as pd
import joblib
import emoji 

# Load saved model, scaler, and expected columns
model = joblib.load("XGBoost_churn.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load('columns.pkl')

st.title("ECommerce Customer Churn Prediction :shopping_cart:")
st.markdown("Provide the following details to check whether the customer is churn or not: ")

sex = st.selectbox("Sex", ["Male", "Female"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
preferred_login_device = st.selectbox("Preferred Login Device ", ["Mobile Phone", "Phone", "Computer"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
warehouse_to_home = st.slider("Warehouse to Home Distance (in KM)", 1, 50, 10)
preferred_payment_mode = st.selectbox("Preferred Payment Mode", ["Credit Card", "Debit Card", "UPI", "E Wallet", "Cash on Delivery"])
hour_spend_on_app = st.slider("Hour Spend on App per Day", 0.0, 10.0, 2.0)
number_of_device_registered = st.slider("Number of Device Registered", 1, 7, 2)
preferred_order_category = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"])
satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
number_of_address = st.slider("Number of Address", 1, 24, 2)
martial_status = st.selectbox("Marital Status", ["Single", "Divorced","Married"])
complain = st.selectbox("Complain", ["Yes", "No"])
coupon_used = st.number_input("Coupon Used", 0, 50, 5)
order_amount_hike = st.number_input("Order Amount Hike from Last Year", 0.0, 40.0, 14.0)
order_count = st.number_input("Order Count", 1, 150, 30)
day_since_last_order = st.number_input("Day Since Last Order", 1, 30, 3)
cashback_amount = st.number_input("Cashback Amount", 0.0, 300.0, 150.0)

# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Sex': sex,
        'Tenure': tenure,
        'PreferredLoginDevice': preferred_login_device,
        'CityTier': city_tier,
        "WarehouseToHome": warehouse_to_home,
        'PreferredPaymentMode': preferred_payment_mode,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': number_of_device_registered,
        'PreferredOrderCategory': preferred_order_category,
        'SatisfactionScore': satisfaction_score,
        'NumberOfAddress': number_of_address,
        'MartialStatus': martial_status,
        'Complain': complain,
        'CouponUsed': coupon_used,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount
    }
    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is not likely to Churn")