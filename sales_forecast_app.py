import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt

st.title("Sales Forecasting App")

st.write("""
Upload your sales data CSV file. The file should have two columns:
- `ds` (date column, format YYYY-MM-DD)
- `y` (sales number)

You can also include extra columns for external factors (optional).
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
   import csv

# Try to detect delimiter automatically
sniffed_delimiter = csv.Sniffer().sniff(uploaded_file.read(1024).decode('utf-8')).delimiter
uploaded_file.seek(0)  # Reset file pointer after sniffing
data = pd.read_csv(uploaded_file, delimiter=sniffed_delimiter)
    st.write("Columns in your file:", data.columns.tolist())
    st.write("Data preview:")
    st.write(data.head())

    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("CSV must contain 'ds' and 'y' columns.")
    else:
        model = Prophet()

        extra_cols = [col for col in data.columns if col not in ['ds', 'y']]
        for col in extra_cols:
            model.add_regressor(col)

        model.fit(data)

        periods = st.number_input("Days to forecast into the future", min_value=1, max_value=365, value=30)

        future = model.make_future_dataframe(periods=periods)

        # Add external regressors to future dataframe by carrying forward last known value
        for col in extra_cols:
            last_val = data[col].iloc[-1]
            future[col] = last_val

        forecast = model.predict(future)

        st.write("Forecast:")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1)

        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
