import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# ---- API Keys ----
OPENWEATHER_API_KEY = 'd846ff6ade7b9ef5a751b10359ec3944'
ALPHAVANTAGE_API_KEY = 'P8R0A9O70NECUB3C'

# ---- Functions ----

def fetch_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()
    data = response.json()
    rows = []
    for entry in data['list']:
        rows.append({
            'ds': entry['dt_txt'],
            'temperature': entry['main']['temp']
        })
    df = pd.DataFrame(rows)
    df['ds'] = pd.to_datetime(df['ds']).dt.date
    return df.groupby('ds').mean().reset_index()

def fetch_market_data():
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=ZAR&apikey={ALPHAVANTAGE_API_KEY}&outputsize=compact'
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()
    data = response.json()['Time Series FX (Daily)']
    df = pd.DataFrame([
        {'ds': pd.to_datetime(date), 'usd_zar': float(values['4. close'])}
        for date, values in data.items()
    ])
    return df.sort_values('ds')

# ---- Streamlit App ----

st.title("📊 Region-Aware Sales Forecasting App")

st.write("""
Upload your sales CSV file with:
- `ds` (date, YYYY-MM-DD)
- `y` (sales)
Optional columns:
- stock_on_hand
""")

# User Inputs
city = st.text_input("Enter your region or city for weather data (e.g. Johannesburg):", value="Johannesburg")

uploaded_file = st.file_uploader("Upload sales CSV", type="csv")

if uploaded_file:
    # Read CSV
    try:
        data = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.write("Data preview:")
        st.write(data.head())
    except Exception as e:
        st.error(f"Could not read file: {e}")
    else:
        if 'ds' not in data.columns or 'y' not in data.columns:
            st.error("CSV must contain 'ds' and 'y' columns.")
        else:
            data['ds'] = pd.to_datetime(data['ds'])

            # Fetch external data
            weather = fetch_weather_data(city)
            market = fetch_market_data()

            # Merge external factors
            df = pd.merge(data, weather, on='ds', how='left')
            df = pd.merge(df, market, on='ds', how='left')
            df.fillna(method='ffill', inplace=True)

            # Prophet model
            model = Prophet()
            for col in df.columns:
                if col not in ['ds', 'y']:
                    model.add_regressor(col)

            model.fit(df)

            # Forecast input
            periods = st.number_input("Days to forecast", min_value=1, max_value=365, value=30)

            # Create future df
            future = model.make_future_dataframe(periods=periods)
            future = pd.merge(future, weather, on='ds', how='left')
            future = pd.merge(future, market, on='ds', how='left')
            future.fillna(method='ffill', inplace=True)

            # Predict
            forecast = model.predict(future)
            st.write("Forecast:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            st.write("Forecast Components:")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
            
