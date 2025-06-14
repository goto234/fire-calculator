import streamlit as st
import pandas as pd

st.title("🔥 FIRE Calculator")

# Inputs from user
current_age = st.slider("Current Age", 20, 60, 30)
retirement_age = st.slider("Retirement Age", current_age + 1, 75, 60)
monthly_savings = st.number_input("Monthly Savings (₹)", value=20000)
expected_return = st.slider("Expected Annual Return (%)", 1.0, 15.0, 8.0)
target_amount = st.number_input("Target FIRE Corpus (₹)", value=30000000)

# Backend logic
years_to_invest = retirement_age - current_age
months_to_invest = years_to_invest * 12
monthly_return = expected_return / 12 / 100

future_value = monthly_savings * (((1 + monthly_return) ** months_to_invest - 1) / monthly_return)

# Output
st.subheader(f"📈 Projected Corpus at Age {retirement_age}: ₹{future_value:,.0f}")

if future_value >= target_amount:
    st.success("🎉 You're on track to achieve FIRE!")
else:
    st.warning("⚠️ You may need to increase savings or adjust retirement age.")
