import os
import tweepy
from dotenv import load_dotenv
from pycoingecko import CoinGeckoAPI
from etherscan import Etherscan
from bscscan import BscScan
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd
import time
import praw
from telegram import Bot  # For Telegram integration
import hyperliquid  # Assuming pip install async-hyperliquid or from GitHub
from web3 import Web3  # For Aster on BNB Chain
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense  # For AI integration
from keras.callbacks import EarlyStopping, ModelCheckpoint  # For optimization
from keras.optimizers import AdamW  # For better optimization
import tensorflow as tf  # For mixed precision and gradient clipping
import optuna  # For Bayesian Optimization

# Load env vars
load_dotenv()
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
BNBSCAN_API_KEY = os.getenv('BNBSCAN_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'crypto_bot/1.0')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
HYPERLIQUID_PRIVATE_KEY = os.getenv('HYPERLIQUID_PRIVATE_KEY')  # For live trading
ASTER_PRIVATE_KEY = os.getenv('ASTER_PRIVATE_KEY')  # Wallet private key for BNB Chain
BNB_RPC = 'https://bsc-dataseed.binance.org/'  # BNB Chain RPC

class CryptoTradingBot:
    def __init__(self, cryptos=['bitcoin', 'ethereum', 'solana'], eth_address=None, bnb_address=None, reddit_subreddit='cryptocurrency', dex='hyperliquid'):
        self.cryptos = [crypto.lower() for crypto in cryptos]
        self.cg = CoinGeckoAPI()
        self.eth = Etherscan(ETHERSCAN_API_KEY) if eth_address else None
        self.bsc = BscScan(BNBSCAN_API_KEY) if bnb_address else None
        self.eth_address = eth_address
        self.bnb_address = bnb_address
        self.auth = tweepy.OAuth1UserHandler(TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
        self.api = tweepy.API(self.auth)
        self.sentiment_threshold_positive = 0.05
        self.sentiment_threshold_negative = -0.05
        self.capital = {'usdt': 10000.0}  # Profits in USDT
        self.positions = {crypto: 0 for crypto in self.cryptos}

        # Load HuggingFace model for sentiment
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.config = AutoConfig.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)

        # Reddit integration
        self.reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                                  client_secret=REDDIT_CLIENT_SECRET,
                                  user_agent=REDDIT_USER_AGENT)
        self.subreddit = self.reddit.subreddit(reddit_subreddit)

        # Telegram integration
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else None

        # DEX integration
        self.dex = dex.lower()
        if self.dex == 'hyperliquid':
            self.hyperliquid_client = hyperliquid.Client(private_key=HYPERLIQUID_PRIVATE_KEY) if HYPERLIQUID_PRIVATE_KEY else None
        elif self.dex == 'aster':
            self.web3 = Web3(Web3.HTTPProvider(BNB_RPC))
            self.aster_contract_address = '0xYourAsterContractAddress'  # Replace with actual
            self.wallet_address = '0xYourWalletAddress'
            self.private_key = ASTER_PRIVATE_KEY

        # Moving Average params
        self.short_ma_period = 50  # e.g., 50-day SMA
        self.long_ma_period = 200  # e.g., 200-day SMA
        # RSI params
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30

        # Grid Trading params
        self.grid_levels = 5  # Number of grid levels above/below price
        self.grid_spacing = 0.01  # 1% spacing

        # Market Making params
        self.spread = 0.005  # 0.5% spread for quotes

        # Scalp Trading params
        self.scalp_target = 0.002  # 0.2% profit target

        # AI Model for price prediction (optimized)
        self.ai_model = self.build_ai_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_ai_model(self, hidden_size=96, num_layers=2, dropout=0.2):
        # Optimized LSTM: Fewer neurons, dropout, AdamW optimizer with gradient clipping
        model = Sequential()
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(60, 1), dropout=dropout, recurrent_dropout=dropout))
        for _ in range(num_layers - 1):
            model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout))
        model.add(Dense(1))
        optimizer = AdamW(learning_rate=0.001, clipnorm=1.0)  # Gradient clipping
        model.compile(optimizer=optimizer, loss='huber')  # Huber loss for robustness
        return model

    def tune_lstm_hyperparams(self, X_train, y_train):
        def objective(trial):
            hidden_size = trial.suggest_int('hidden_size', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

            model = self.build_ai_model(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            optimizer = AdamW(learning_rate=learning_rate, clipnorm=1.0)
            model.compile(optimizer=optimizer, loss='huber')

            early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0, callbacks=[early_stop])

            # Validation (use last 20% as val)
            val_size = int(len(X_train) * 0.2)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            return val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)  # 20 trials for balance
        best_params = study.best_params
        print(f"Best LSTM Params: {best_params}")

        # Rebuild with best params
        self.ai_model = self.build_ai_model(hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], dropout=best_params['dropout'])

    def train_ai_model(self, historical_prices):
        if len(historical_prices) < 60:
            return
        scaled_data = self.scaler.fit_transform(historical_prices['price'].values.reshape(-1, 1))
        X_train = []
        y_train = []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Tune hyperparameters with Bayesian Optimization
        self.tune_lstm_hyperparams(X_train, y_train)

        # Train with optimized model
        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        checkpoint = ModelCheckpoint('lstm_model.h5', save_best_only=True)
        self.ai_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0, callbacks=[early_stop, checkpoint])  # Increased batch size

    def predict_next_price(self, historical_prices):
        if len(historical_prices) < 60:
            return 0
        scaled_data = self.scaler.transform(historical_prices['price'].values.reshape(-1, 1))
        last_60 = scaled_data[-60:]
        last_60 = np.reshape(last_60, (1, 60, 1))
        predicted = self.ai_model.predict(last_60)
        return self.scaler.inverse_transform(predicted)[0][0]

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def fetch_tweets(self, query, count=100):
        try:
            tweets = self.api.search_tweets(q=query, lang='en', count=count, tweet_mode='extended')
            return [tweet.full_text for tweet in tweets]
        except tweepy.TweepyException as e:
            print(f"Twitter API error: {e}")
            return []

    def fetch_reddit_posts(self, query, limit=50):
        try:
            posts = []
            for submission in self.subreddit.search(query, limit=limit):
                posts.append(submission.title + " " + submission.selftext)
            return posts
        except Exception as e:
            print(f"Reddit API error: {e}")
            return []

    def analyze_sentiment(self, texts):
        if not texts:
            return 0
        preprocessed_texts = [self.preprocess(text) for text in texts]
        encoded_input = self.tokenizer(preprocessed_texts, return_tensors='pt', truncation=True, max_length=512, padding=True)
        output = self.model(**encoded_input)
        logits = output.logits.detach().numpy()
        probs = softmax(logits, axis=1)
        scores = probs[:, 0] * (-1) + probs[:, 1] * 0 + probs[:, 2] * 1
        avg_score = np.mean(scores)
        return avg_score

    def fetch_historical_prices(self, crypto, days=365):
        try:
            data = self.cg.get_coin_market_chart_by_id(id=crypto, vs_currency='usd', days=days)
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            return prices
        except Exception as e:
            print(f"CoinGecko historical error for {crypto}: {e}")
            return pd.DataFrame()

    def compute_moving_averages(self, prices):
        short_ma = prices['price'].rolling(window=self.short_ma_period).mean().iloc[-1]
        long_ma = prices['price'].rolling(window=self.long_ma_period).mean().iloc[-1]
        previous_short_ma = prices['price'].rolling(window=self.short_ma_period).mean().iloc[-2]
        previous_long_ma = prices['price'].rolling(window=self.long_ma_period).mean().iloc[-2]
        if short_ma > long_ma and previous_short_ma <= previous_long_ma:
            return 'buy'  # Golden cross
        elif short_ma < long_ma and previous_short_ma >= previous_long_ma:
            return 'sell'  # Death cross
        return 'hold'

    def compute_rsi(self, prices):
        delta = prices['price'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean().iloc[-1]
        avg_loss = loss.rolling(window=self.rsi_period).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 0
        return rsi

    def fetch_price_action(self, crypto):
        try:
            data = self.cg.get_coin_market_chart_by_id(id=crypto, vs_currency='usd', days=1)
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['return'] = prices['price'].pct_change()
            trend = 'up' if prices['return'].mean() > 0 else 'down'
            return prices['price'].iloc[-1], trend
        except Exception as e:
            print(f"CoinGecko error for {crypto}: {e}")
            return 0, 'neutral'

    def fetch_on_chain_data(self, crypto):
        on_chain_activity = 0
        if crypto == 'ethereum' and self.eth and self.eth_address:
            try:
                txns = self.eth.get_normal_txs_by_address(self.eth_address, startblock=0, endblock=99999999, sort='desc')
                on_chain_activity += len(txns)
            except:
                pass
        # Add for other chains if needed
        return 'high' if on_chain_activity > 10 else 'low'

    def send_telegram_update(self, message):
        if self.telegram_bot:
            try:
                self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
                print("Telegram update sent.")
            except Exception as e:
                print(f"Telegram error: {e}")

    def execute_trade_on_dex(self, crypto, decision, price):
        if decision == 'buy':
            amount_usdt = self.capital.get('usdt', 0) * 0.5  # Risk 50% for simplicity
            amount_crypto = amount_usdt / price
            if self.dex == 'hyperliquid' and self.hyperliquid_client:
                order = self.hyperliquid_client.place_order(symbol=f"{crypto.upper()}-USD", side='buy', size=amount_crypto, price=price)
                print(f"Placed buy order on Hyperliquid: {order}")
                self.positions[crypto] += amount_crypto
                self.capital['usdt'] -= amount_usdt
            elif self.dex == 'aster' and self.private_key:
                tx = {
                    'from': self.wallet_address,
                    'to': self.aster_contract_address,
                    'data': '0x...'  # Encoded buy function call
                }
                signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
                print(f"Placed buy tx on Aster: {tx_hash.hex()}")
                self.positions[crypto] += amount_crypto
                self.capital['usdt'] -= amount_usdt
            return f"BUY: {amount_crypto:.4f} {crypto.upper()} at ${price:.2f}"
        elif decision == 'sell':
            amount_crypto = self.positions.get(crypto, 0)
            if amount_crypto > 0:
                amount_usdt = amount_crypto * price
                if self.dex == 'hyperliquid' and self.hyperliquid_client:
                    order = self.hyperliquid_client.place_order(symbol=f"{crypto.upper()}-USD", side='sell', size=amount_crypto, price=price)
                    print(f"Placed sell order on Hyperliquid: {order}")
                elif self.dex == 'aster' and self.private_key:
                    tx = {
                        'from': self.wallet_address,
                        'to': self.aster_contract_address,
                        'data': '0x...'  # Encoded sell function
                    }
                    signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    print(f"Placed sell tx on Aster: {tx_hash.hex()}")
                profit = amount_usdt - (self.capital.get('usdt', 10000) * (amount_crypto / sum(self.positions.values() or [1])))
                self.capital['usdt'] += amount_usdt
                self.positions[crypto] = 0
                return f"SELL: Profit/Loss ${profit:.2f}. New USDT: ${self.capital['usdt']:.2f}"
            return "No position to sell"

    def execute_grid_trade(self, crypto, price):
        # Placeholder for grid trading: Place buy/sell orders in grid
        for i in range(1, self.grid_levels + 1):
            buy_price = price * (1 - i * self.grid_spacing)
            sell_price = price * (1 + i * self.grid_spacing)
            # Execute small orders at these levels (adapt to DEX API)
            print(f"Grid order for {crypto}: Buy at {buy_price:.2f}, Sell at {sell_price:.2f}")

    def execute_market_making(self, crypto, price):
        bid = price * (1 - self.spread / 2)
        ask = price * (1 + self.spread / 2)
        # Place limit orders (adapt to DEX)
        print(f"Market making for {crypto}: Bid {bid:.2f}, Ask {ask:.2f}")

    def execute_scalp_trade(self, crypto, price, predicted_price):
        if predicted_price > price * (1 + self.scalp_target):
            # Buy and set sell target
            print(f"Scalp buy {crypto} at {price:.2f}, target {price * (1 + self.scalp_target):.2f}")
        elif predicted_price < price * (1 - self.scalp_target):
            # Short/sell
            print(f"Scalp sell {crypto} at {price:.2f}, target {price * (1 - self.scalp_target):.2f}")

    def decide_trade(self):
        update_msg = ""
        for crypto in self.cryptos:
            tweets = self.fetch_tweets(f"{crypto} sentiment OR price OR news", count=50)
            reddit_posts = self.fetch_reddit_posts(f"{crypto} sentiment OR price OR news", limit=50)
            all_texts = tweets + reddit_posts
            sentiment_score = self.analyze_sentiment(all_texts)
            price, price_trend = self.fetch_price_action(crypto)
            on_chain = self.fetch_on_chain_data(crypto)

            # Moving Average Crossover
            historical_prices = self.fetch_historical_prices(crypto, days=self.long_ma_period + 1)
            ma_signal = self.compute_moving_averages(historical_prices) if not historical_prices.empty else 'hold'

            # RSI
            rsi = self.compute_rsi(historical_prices) if not historical_prices.empty else 50

            # AI Prediction
            self.train_ai_model(historical_prices)
            predicted_price = self.predict_next_price(historical_prices) if not historical_prices.empty else price

            crypto_msg = f"{crypto.upper()} - Sentiment: {sentiment_score:.2f}, Price: ${price:.2f}, Trend: {price_trend}, On-Chain: {on_chain}, MA Signal: {ma_signal}, RSI: {rsi:.2f}, Predicted Price: ${predicted_price:.2f}"
            print(crypto_msg)
            update_msg += crypto_msg + "\n"

            # Refined signal combination logic
            sentiment_val = sentiment_score
            price_val = 1 if price_trend == 'up' else -1 if price_trend == 'down' else 0
            on_chain_val = 1 if on_chain == 'high' else 0
            ma_val = 1 if ma_signal == 'buy' else -1 if ma_signal == 'sell' else 0
            rsi_val = 1 if rsi < self.rsi_oversold else -1 if rsi > self.rsi_overbought else 0
            ai_val = 1 if predicted_price > price else -1 if predicted_price < price else 0

            # Weighted combined score (sentiment 30%, MA 15%, price 10%, on-chain 15%, RSI 15%, AI 15%)
            combined_score = (sentiment_val * 0.3) + (ma_val * 0.15) + (price_val * 0.1) + (on_chain_val * 0.15) + (rsi_val * 0.15) + (ai_val * 0.15)

            if combined_score > 0.5:
                decision = 'buy'
            elif combined_score < -0.5:
                decision = 'sell'
            else:
                decision = 'hold'

            trade_msg = self.execute_trade_on_dex(crypto, decision, price)
            if trade_msg:
                print(trade_msg)
                update_msg += f"\n{trade_msg}"

            # Additional strategies if hold or for diversification
            if decision == 'hold':
                self.execute_grid_trade(crypto, price)
                self.execute_market_making(crypto, price)
            self.execute_scalp_trade(crypto, price, predicted_price)

        # Send update to Telegram
        self.send_telegram_update(update_msg)

    def run(self, intervals=5, sleep=60):
        for _ in range(intervals):
            self.decide_trade()
            time.sleep(sleep)

if __name__ == "__main__":
    bot = CryptoTradingBot(cryptos=['bitcoin', 'ethereum', 'solana'], eth_address='0xab5801a7d398351b8be11c439e05c5b3259aec9b', bnb_address='0xab5801a7d398351b8be11c439e05c5b3259aec9b', dex='hyperliquid')  # or 'aster'
    bot.run(intervals=3, sleep=300)