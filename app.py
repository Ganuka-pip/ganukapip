# -- Imports (unchanged) --
import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pytz import timezone as tz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import streamlit as st
import telegram

# -- Streamlit Config (unchanged) --
st.set_page_config(page_title="Ultimate Binance AI Bot", layout="wide")
st.title("Ultimate Binance AI Real-Time Signal Bot - Ultra Pro Max")

page_bg_img = '''
<style>
.stApp {
background-image: url("https://images.pexels.com/photos/333850/pexels-photo-333850.jpeg");
background-size: cover;
background-attachment: fixed;
}
header.css-18ni7ap.e8zbici2 {
background-color: rgba(255, 0, 0, 0.8);
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# -- Auto-Scan State Setup --
if "auto_scan" not in st.session_state:
    st.session_state.auto_scan = False
if "scan_now" not in st.session_state:
    st.session_state.scan_now = False

col1, col2 = st.columns(2)
if col1.button('🔍 Scan Now'):
    st.session_state.scan_now = True
if col2.button('🔁 Toggle Auto Scan'):
    st.session_state.auto_scan = not st.session_state.auto_scan
    st.success(f"Auto Scan is now {'ON' if st.session_state.auto_scan else 'OFF'}")

# -- Constants (unchanged except smaller SYMBOLS) --
BINANCE_API = 'https://api.binance.com/api/v3/klines'

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']  # 👉 First testing small batch

INTERVAL = '1h'
LIMIT = 500
TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'
bot = telegram.Bot(token=TOKEN)
MAX_RISK_PER_TRADE = 0.01
ACCOUNT_BALANCE = 1000

feature_list = [
    'Orderblock','MSB','Liquidity_Grab','Breaker','Mitigation','FVG',
    'Equal_High_Low','Int_Ext_Sweep','Session_Smart','MTF_Confirm',
    'Choch','RSI_Div','HVN','LVN','VWAP','OB_Imbalance','Cum_Delta',
    'MP_High','MP_Low','Funding_Rate','Open_Interest','News_Sentiment',
    'Corr_Breakout','Gamma_Squeeze','Curve_Skew','Anomaly'
]

# -- 🔥 Updated fetch_binance_data() with error handling --
def fetch_binance_data(symbol, interval, limit):
    try:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        res = requests.get(BINANCE_API, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if not data:
            st.warning(f"⚠️ Empty data received for {symbol}")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','quote_asset_volume','trades',
            'taker_buy_volume','taker_buy_quote_volume','ignore'
        ])
        for col in ['open','high','low','close','volume','taker_buy_volume']:
            df[col] = df[col].astype(float)
        df['dt'] = pd.to_datetime(df['close_time'], unit='ms')
        df['session'] = df['dt'].dt.tz_localize('UTC').dt.tz_convert(tz('Asia/Colombo')).dt.hour
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch {symbol}: {str(e)}")
        return pd.DataFrame()

# -- Apply Theories (unchanged) --
def apply_theories(df):
    df['Orderblock'] = ((df['close']<df['open']) & (df['volume']>df['volume'].rolling(5).mean())).astype(int)
    df['MSB'] = (((df['close']>df['close'].shift(1)) & (df['close'].shift(1)<df['close'].shift(2)))).astype(int)
    df['Liquidity_Grab'] = (df['low']<df['low'].rolling(10).min().shift(1)).astype(int)
    df['Breaker'] = ((df['MSB']==1)&(df['close']>df['close'].shift(1))).astype(int)
    df['Mitigation'] = (((df['low']<df['low'].rolling(5).min().shift(1))&(df['close']>df['open']))).astype(int)
    df['FVG'] = (((df['low'].shift(-1)>df['high'])&(df['low'].shift(-2)>df['high']))).astype(int)
    df['Equal_High_Low'] = (((df['high']==df['high'].shift(1))|(df['low']==df['low'].shift(1)))).astype(int)
    df['Int_Ext_Sweep'] = (((df['low']<df['low'].shift(1))&(df['close']>df['low'].shift(1)))|((df['high']>df['high'].shift(1))&(df['close']<df['high'].shift(1)))).astype(int)
    df['Session_Smart'] = (((df['session']>=1)&(df['session']<9)&(df['high']>df['high'].shift(24)))|((df['session']>=9)&(df['session']<17)&(df['high']>df['high'].shift(24)))|((df['session']>=17)&(df['session']<=23)&(df['high']>df['high'].shift(24)))).astype(int)
    df['Choch'] = (((df['high'].shift(1)>df['high'])&(df['low'].shift(1)<df['low']))).astype(int)
    df['RSI'] = 100-(100/(1+df['close'].pct_change().rolling(14).mean()))
    df['RSI_Div'] = (((df['close'].diff()>0)&(df['RSI'].diff()<0))|((df['close'].diff()<0)&(df['RSI'].diff()>0))).astype(int)
    vol_mean = df['volume'].rolling(20).mean()
    df['HVN'] = (df['volume']>vol_mean).astype(int)
    df['LVN'] = (df['volume']<vol_mean).astype(int)
    df['VWAP'] = (df['volume']*df['close']).cumsum() / df['volume'].cumsum()
    df['OB_Imbalance'] = (df['taker_buy_volume'] - (df['volume']-df['taker_buy_volume'])) / df['volume']
    df['Cum_Delta'] = (df['taker_buy_volume'] - (df['volume']-df['taker_buy_volume'])).cumsum()
    df['MP_High'] = df['high'].rolling(50).max()
    df['MP_Low'] = df['low'].rolling(50).min()
    df['Funding_Rate'] = 0.0001
    df['Open_Interest'] = df['volume']*np.random.uniform(0.5,1.5,len(df))
    df['News_Sentiment'] = np.random.uniform(-1,1,len(df))
    df['Corr_Breakout'] = np.where(df['close'].pct_change().rolling(5).corr(df['close'].pct_change().shift(1))<0.5,1,0)
    df['Gamma_Squeeze'] = np.random.randint(0,2,len(df))
    df['Curve_Skew'] = np.random.uniform(-0.5,0.5,len(df))
    df['Anomaly'] = np.where(abs(df['close'].pct_change())>0.05,1,0)
    df['ATR'] = (df['high']-df['low']).rolling(14).mean()
    df['Fibo_Support'] = df['low'].rolling(20).min()*1.618
    df['Fibo_Resistance'] = df['high'].rolling(20).max()*0.618
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    df['Trend'] = np.where(df['EMA50'] > df['EMA200'], 1, 0)
    return df

# -- Train Model, Scan Signals (unchanged logic) --
# (same code you posted.)

# Model, Scaler, Trend
model, scaler, trend_1h = None, None, None

def train_model():
    global model, scaler, trend_1h

    # Fetch 1h timeframe data
    df_1h = fetch_binance_data(SYMBOLS[0], '1h', LIMIT)
    if df_1h.empty:
        st.warning("⚠️ No 1h data fetched. Skipping model training.")
        model = None
        scaler = None
        trend_1h = None
        return

    df_1h = apply_theories(df_1h)
    if df_1h.empty:
        st.warning("⚠️ No 1h data after applying theories. Skipping model training.")
        model = None
        scaler = None
        trend_1h = None
        return

    trend_1h = df_1h['Trend'].iloc[-1]

    # Fetch main interval data
    df = fetch_binance_data(SYMBOLS[0], INTERVAL, LIMIT)
    if df.empty:
        st.warning("⚠️ No main data fetched. Skipping model training.")
        model = None
        scaler = None
        return

    df = apply_theories(df)
    df = df.dropna()

    if df.empty:
        st.warning("⚠️ No data after theories. Skipping model training.")
        model = None
        scaler = None
        return

    df['TF_1h_Trend'] = trend_1h
    df['MTF_Confirm'] = (df['Trend'] == df['TF_1h_Trend']).astype(int)

    X = df[feature_list]
    y = np.where(
        (df['close'].shift(-1) < df['close']) & 
        (df['close'].shift(-2) < df['close']) & 
        (df['close'].shift(-3) < df['close']),
        0, 1
    )

    X, y = X.iloc[:-3], y[:-3]

    if len(X) < 10:
        st.warning("⚠️ Not enough data to train the model. Skipping.")
        model = None
        scaler = None
        return

    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(Xt)
    Xt_s, Xv_s = scaler.transform(Xt), scaler.transform(Xv)

    grid = {'max_depth': [3, 4], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]}
    gs = GridSearchCV(XGBClassifier(), grid, cv=3)
    gs.fit(Xt_s, yt)

    model = gs.best_estimator_
    st.success("✅ Model trained successfully!")

# Signal Scanner
def scan_signals():
    st.write("### Live Ultra Pro Max Signals for Multiple Symbols:")
    bool_feats = ['Orderblock','Breaker','Mitigation','FVG','Session_Smart','MTF_Confirm','Choch','RSI_Div','HVN']

    for sym in SYMBOLS:
        df = fetch_binance_data(sym, INTERVAL, LIMIT)
        if df.empty:
            st.warning(f"⚠️ {sym} - No data fetched. Skipping...")
            continue

        df = apply_theories(df)
        if df.empty:
            st.warning(f"⚠️ {sym} - No data after applying theories. Skipping...")
            continue

        df['TF_1h_Trend'] = trend_1h
        df['MTF_Confirm'] = (df['Trend'] == df['TF_1h_Trend']).astype(int)

        Xl = df[feature_list].tail(1).fillna(0)
        sl = df['Fibo_Support'].iat[-1]
        ep = df['close'].iat[-1]

        if np.isnan(sl) or np.isnan(ep):
            st.warning(f"⚠️ {sym} - SL or Entry price missing. Skipping...")
            continue

        risk_amt = ACCOUNT_BALANCE * MAX_RISK_PER_TRADE
        pos_size = risk_amt / abs(ep - sl)
        scaled = scaler.transform(Xl)
        pred = model.predict(scaled)[-1]
        sig = 'BUY' if pred == 1 else 'SELL'
        confirms = df[bool_feats].tail(1).values.flatten().sum()
        conf_pc = (confirms / len(bool_feats)) * 100

        if conf_pc < 10:
            continue

        risk_unit = abs(ep - sl)
        mult = 2.5 if conf_pc >= 90 else 2.0 if conf_pc >= 70 else 1.5
        tp = ep + risk_unit * mult
        out = {
            "Symbol": sym, "Signal": sig, "Confidence%": f"{conf_pc:.1f}%", 
            "Entry": ep, "SL": sl, "TP": tp, "Size": f"{pos_size:.2f} units"
        }
        st.write(pd.DataFrame([out]))


        # 📊 Chart View
        fig, ax = plt.subplots()
        ax.plot(df['dt'], df['close'], label='Close')
        ax.axhline(y=sl, color='red', linestyle='--', label='SL')
        ax.axhline(y=tp, color='green', linestyle='--', label='TP')
        ax.axhline(y=ep, color='blue', linestyle='--', label='Entry')
        ax.legend()
        ax.set_title(f"{sym} Price + Signal Levels")
        st.pyplot(fig)

# Run Scan Now or Auto
if st.session_state.get("scan_now") or st.session_state.auto_scan:
    scan_signals()
    st.session_state.scan_now = False
    if st.session_state.auto_scan:
        time.sleep(3600)
        st.experimental_rerun()
