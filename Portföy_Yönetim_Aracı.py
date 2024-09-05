import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# Başlık
st.title('Portföy Yönetim Aracı')

# Kullanıcıdan hisse senedi sembollerini alın
st.sidebar.header("Portföy Bilgileri")
symbols = st.sidebar.text_input("Hisse Senedi Sembolleri (Virgülle Ayırın)", "AAPL, MSFT, TSLA")
symbols = [sym.strip().upper() for sym in symbols.split(',')]

# Kullanıcıdan her hisse senedi için yatırım miktarını alın
amounts = {}
for sym in symbols:
    amount = st.sidebar.number_input(f'{sym} için yatırım miktarı girin :', min_value=0, value=1000)
    amounts[sym] = amount

def load_data(symbols):
    data = {}
    for sym in symbols:
        data[sym] = yf.download(sym, start="2022-01-01", end="2024-01-01")['Close']
    if not data:
        raise ValueError("Veri çekilmedi. Lütfen hisse senedi sembollerini kontrol edin.")
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("Oluşturulan DataFrame boş.")
    return df

# Veriyi yükleyin
df = load_data(symbols)

# Portföy performansını hesaplayın
total_investment = sum(amounts.values())
portfolio_value = df.iloc[-1] * pd.Series(amounts)
portfolio_value_total = portfolio_value.sum()

st.write(f"Toplam yatırım miktarı: ${total_investment:.2f}")
st.write(f"Her bir hissenin mevcut değeri:")
st.write(portfolio_value)
st.write(f"Portföyün toplam değeri: ${portfolio_value_total:.2f}")

# Plotly ile Etkileşimli Grafikler
st.subheader("Hisse Senedi Fiyatları")
fig = px.line(df, title = "Hisse Senedi Fiyatları")
st.plotly_chart(fig)

st.subheader("Portföy Dağılım Grafiği")
fig = px.bar(x = amounts.keys(), y = portfolio_value, labels = {"x" : 'Yatırım Araçları', 'y' : 'Portföy Değeri' })
st.plotly_chart(fig)

st.subheader("Günlük Getiriler")
daily_returns = df.pct_change().dropna()
st.dataframe(daily_returns)
st.write("Günlük getiriler, hisse senetlerinin günlük fiyat değişimlerini gösterir.")

st.subheader("Aylık Getiriler")
monthly_returns = df.resample('M').ffill().pct_change().dropna()
st.dataframe(monthly_returns)
st.write("Aylık getiriler, hisse senetlerinin her ay için fiyat değişimlerini gösterir.")

st.subheader("Yıllık Getiriler")
annual_returns = df.resample('Y').ffill().pct_change().dropna()
st.dataframe(annual_returns)
st.write("Yıllık getiriler, hisse senetlerinin her yıl için fiyat değişimlerini gösterir.")

# Risk Analizi
st.subheader("Volatilite (Yıllık)")
volatility  = daily_returns.std() * np.sqrt(252)
st.write("Volatilite, bir varlığın fiyatındaki dalgalanmaların büyüklüğünü ifade eder. Bu kod, günlük volatiliteyi yıllık volatiliteye çevirir.")
st.dataframe(volatility)

st.subheader("Beta Değeri")
beta = daily_returns.cov() / daily_returns.var()
st.write("Beta değeri, bir varlığın piyasa ile olan ilişkisini ölçer. Piyasa riskine karşı duyarlılığı anlamak için kullanılır.")
st.dataframe(beta)

# Hareketli Ortalama
st.subheader("Hareketli Ortalama")
window_size = st.sidebar.slider("Hareketli Ortalama Penceresi (Gün)", 5, 100, 20)
moving_avg = df.rolling(window=window_size).mean()
st.write(f"{window_size} Günlük Hareketli Ortalama:")
st.line_chart(moving_avg)

# Zaman Aralığı Filtreleri
st.subheader("Tarih Aralığına Göre Veri Filtreleme")
start_date = st.sidebar.date_input("Başlangıç Tarihi", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Bitiş Tarihi", pd.to_datetime("2024-01-01"))

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

if start_date < end_date:
    filtered_data = df[(df.index >= start_date) & (df.index <= end_date)]
    st.write("Seçilen tarih aralığındaki hisse senedi fiyatları:")
    st.line_chart(filtered_data)
else:
    st.error("Başlangıç tarihi bitiş tarihinden sonra olmamalıdır.")

# Monte Carlo Simülasyonu
mean_returns = daily_returns.mean()  # Ortalama günlük getiriler
simulations = 1000
simulation_df = pd.DataFrame()

for x in range(simulations):
    simulated_prices = []
    
    for symbol in symbols:
        price_series = [df[symbol].iloc[-1]]
        for _ in range(365):  # 1 yıl
            price_series.append(price_series[-1] * (1 + np.random.normal(mean_returns[symbol], volatility[symbol])))
        simulated_prices.append(price_series)

    simulation_df[x] = pd.Series([np.sum(sim) for sim in zip(*simulated_prices)])

st.write(
    """
    **Monte Carlo Simülasyonu:** Belirli bir varlığın veya portföyün gelecekteki değerlerini tahmin etmek için kullanılan bir yöntemdir.
    """
)
st.line_chart(simulation_df)
