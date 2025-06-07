import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import csv

st.title("Sales Forecasting App")

st.write("""
Upload your sales data CSV file.
The file should have two columns:
- `ds` (date column, format YYYY-MM-DD)
- `y` (sales number)
You can also include extra columns for external factors (optional).
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Detect delimiter
    sample = uploaded_file.read(1024).decode('utf-8')
    delimiter = csv.Sniffer().sniff(sample).delimiter
    uploaded_file.seek(0)  # Reset pointer
    data = pd.read_csv(uploaded_file, delimiter=delimiter)

    st.write("Columns in your file:", data.columns.tolist())
    st.write("Data preview:")
    st.write(data.head())

    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("CSV must contain 'ds' and 'y' columns.")
    else:
        model = Prophet()

        # Add any external regressors
        extra_cols = [col for col in data.columns if col not in ['ds', 'y']]
        for col in extra_cols:
            model.add_regressor(col)

        model.fit(data)

        periods = st.number_input("Days to forecast into the future", min_value=1, max_value=365, value=30)

        future = model.make_future_dataframe(periods=periods)

        for col in extra_cols:
            last_value = data[col].iloc[-1]
            future[col] = last_value

        forecast = model.predict(future)

        st.write("Forecast:")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1)

        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
