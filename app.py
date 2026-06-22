import os
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from twilio.rest import Client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================================================================
# SYSTEM STYLING & INSTITUTIONAL THEME SETUP
# =========================================================================
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available() else 'default')
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 16,
    'grid.color': '#2A2E39', 'grid.alpha': 0.5, 'axes.facecolor': '#131722',
    'figure.facecolor': '#131722', 'text.color': '#D1D4DC', 'axes.labelcolor': '#D1D4DC',
    'xtick.color': '#D1D4DC', 'ytick.color': '#D1D4DC'
})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILE = "institutional_config.json"
TICKER_FILE = "ticker_universe.txt"

DEFAULT_CONFIG = {
    "lookback_period": 20, "rsi_period": 14, "atr_period": 14, "roc_period": 20,
    "momentum_threshold": 3.0, "atr_deviation_threshold": 1.2, "zscore_threshold": 1.5,
    "capital_base": 100000.0, "transaction_cost_pct": 0.001
}
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

def initialize_files():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f: json.dump(DEFAULT_CONFIG, f, indent=4)
    if not os.path.exists(TICKER_FILE):
        with open(TICKER_FILE, 'w') as f: f.write("\n".join(DEFAULT_TICKERS))

def load_settings():
    initialize_files()
    with open(CONFIG_FILE, 'r') as f: config = json.load(f)
    with open(TICKER_FILE, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return config, tickers

# =========================================================================
# MODULE 1: YAHOO FINANCE INGESTION ENGINE
# =========================================================================
class YahooFinanceIngestionEngine:
    @staticmethod
    def fetch_historical_data(ticker: str, periods_days: int = 365) -> pd.DataFrame:
        """Fetches adjusted daily OHLCV historical arrays seamlessly from yfinance."""
        try:
            logging.info(f"Downloading historical timeline for {ticker} from Yahoo Finance...")
            asset = yf.Ticker(ticker)
            df = asset.history(period="1y", interval="1d") # 1y covers trailing active days
            if df.empty: raise ValueError(f"No pricing table returned for ticker: {ticker}")
            
            # Lowercase columns to maintain standard compliance with backend calculation blocks
            df.columns = [col.lower() for col in df.columns]
            
            # Fetch Index benchmark for relative spread cointegration tracking
            spy = yf.Ticker("SPY").history(period="1y", interval="1d")
            df['spy_close'] = spy['Close'].reindex(df.index, method='ffill')
            return df
        except Exception as e:
            logging.error(f"Failed Ingestion on {ticker}: {str(e)}")
            return pd.DataFrame()

# =========================================================================
# MODULE 2, 4 & 5: TECHNICAL ANALYSIS & DISPATCH ALERTS MODULE
# =========================================================================
class QuantitativeSignalEngine:
    def __init__(self, config: dict): self.cfg = config
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        w_df = df.copy()
        w_df['signal_momentum_roc'] = w_df['close'].pct_change(periods=self.cfg['roc_period']) * 100
        high_low = w_df['high'] - w_df['low']
        high_cp = np.abs(w_df['high'] - w_df['close'].shift(1))
        low_cp = np.abs(w_df['low'] - w_df['close'].shift(1))
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr = tr.rolling(window=self.cfg['atr_period']).mean()
        ma_20 = w_df['close'].rolling(window=self.cfg['lookback_period']).mean()
        w_df['signal_atr_deviation'] = (w_df['close'] - ma_20) / (atr + 1e-9)
        
        spread = w_df['close'] - w_df['spy_close']
        w_df['signal_residual_zscore'] = (spread - spread.rolling(20).mean()) / (spread.rolling(20).std() + 1e-9)
        
        # Volatility Regime Filter
        ret = w_df['close'].pct_change()
        w_df['regime_volatility'] = np.where((ret.rolling(20).std() * np.sqrt(252)) > (ret.rolling(250, min_periods=1).std().median() * np.sqrt(252)) * 1.3, "HIGH_VOL", "NORMAL")
        return w_df.dropna()

class InstitutionalDecisionEngine:
    def __init__(self, config: dict): self.cfg = config
    def evaluate_row(self, row: pd.Series) -> str:
        b, s = 0, 0
        if row['signal_momentum_roc'] > self.cfg['momentum_threshold']: b += 1
        elif row['signal_momentum_roc'] < -self.cfg['momentum_threshold']: s += 1
        if row['signal_atr_deviation'] < -self.cfg['atr_deviation_threshold']: b += 1
        elif row['signal_atr_deviation'] > self.cfg['atr_deviation_threshold']: s += 1
        if row['signal_residual_zscore'] < -self.cfg['zscore_threshold']: b += 1
        elif row['signal_residual_zscore'] > self.cfg['zscore_threshold']: s += 1
        if row['regime_volatility'] == "HIGH_VOL": return "HOLD"
        return "BUY" if b >= 2 else ("SELL" if s >= 2 else "HOLD")

    def execute_and_alert(self, df: pd.DataFrame, ticker: str):
        df['execution_signal'] = df.apply(self.evaluate_row, axis=1)
        
        # Check Terminal Row for Active Triggers
        latest_signal = df['execution_signal'].iloc[-1]
        prev_signal = df['execution_signal'].iloc[-2] if len(df) > 1 else "HOLD"
        
        if latest_signal in ["BUY", "SELL"] and latest_signal != prev_signal:
            msg = f"⚠️ INSTITUTIONAL ALERT: {ticker} generated a fresh {latest_signal} signal at price ${round(df['close'].iloc[-1], 2)}."
            NotificationDispatcher.send_notifications(msg)
        return df

class NotificationDispatcher:
    @staticmethod
    def send_notifications(message_body: str):
        """Dispatches automated alerts across standard SMS and WhatsApp sandbox targets."""
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        token = os.environ.get("TWILIO_AUTH_TOKEN")
        to_phone = os.environ.get("MY_PHONE_NUMBER")
        twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER")
        
        if not all([sid, token, to_phone, twilio_phone]):
            logging.warning("Alert muted: Notification keys are missing from environmental variables.")
            return

        try:
            client = Client(sid, token)
            # 1. Standard Mobile SMS Alert Dispatch
            client.messages.create(body=message_body, from_=twilio_phone, to=to_phone)
            # 2. WhatsApp Sandbox Session Dispatch
            client.messages.create(body=message_body, from_=f"whatsapp:{twilio_phone}", to=f"whatsapp:{to_phone}")
            logging.info("Notifications successfully transmitted.")
        except Exception as e:
            logging.error(f"Notification failure: {str(e)}")

# =========================================================================
# MODULE 6 & 7: DRAWDOWN CONTROL BACKTESTING ENGINE
# =========================================================================
class PortfolioRiskBacktester:
    def __init__(self, config: dict):
        self.cap = config['capital_base']
        self.fee = config['transaction_cost_pct']
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        pos, cash, shares, curve = 0, self.cap, 0, []
        signals, close = df['execution_signal'].values, df['close'].values
        for i in range(len(df)):
            if signals[i] == "BUY" and pos == 0:
                shares = cash / close[i] * (1 - self.fee); cash = 0; pos = 1
            elif signals[i] == "SELL" and pos == 1:
                cash = shares * close[i] * (1 - self.fee); shares = 0; pos = 0
            curve.append(cash if pos == 0 else (shares * close[i]))
        df['portfolio_value'] = curve
        df['drawdown'] = (df['portfolio_value'] - df['portfolio_value'].cummax()) / df['portfolio_value'].cummax()
        return df

# =========================================================================
# INTERACTIVE CROSS-PLATFORM USER INTERFACE (STREAMLIT ENGINE)
# =========================================================================
def main():
    st.set_page_config(page_title="Alpha Screener", page_icon="📈", layout="wide")
    st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)
    
    config, tickers = load_settings()
    
    st.title("🏛️ Institutional Multi-Factor Strategy Screening Core")
    st.sidebar.header("Control Panel Grid")
    selected_ticker = st.sidebar.selectbox("Select Target Asset Vector", tickers)
    
    # Run quantitative workflow pipes
    raw = YahooFinanceIngestionEngine.fetch_historical_data(selected_ticker)
    if raw.empty:
        st.error("Error processing data extraction via Yahoo Finance networks.")
        return
        
    sig_engine = QuantitativeSignalEngine(config)
    dec_engine = InstitutionalDecisionEngine(config)
    tester = PortfolioRiskBacktester(config)
    
    processed = sig_engine.calculate_signals(raw)
    decided = dec_engine.execute_and_alert(processed, selected_ticker)
    final_df = tester.run(decided)
    
    # Structural Key Matrix Tiles
    last_row = final_df.iloc[-1]
    tot_ret = ((final_df['portfolio_value'].iloc[-1] / config['capital_base']) - 1) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current State Allocation", last_row['execution_signal'])
    c2.metric("Market Unit Valuation", f"${round(last_row['close'], 2)}")
