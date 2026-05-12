# G27 Stock Price Predictor

A stock price prediction tool using an LSTM (Long Short-Term Memory) neural network built with PyTorch. It trains on historical stock data (CSV files) and forecasts future prices.

## Features

- LSTM model trained on historical High price data
- Saves and reloads the trained model (`LSTM_model.pth`) across runs
- Forecasts future stock prices beyond the historical dataset
- Optional matplotlib plots for training results and future predictions
- GPU support via CUDA (falls back to CPU automatically)

## Project Structure

```
Ai_project-main/
├── Main.py              # Entry point — loads CSVs and runs training
├── ML_01.py             # LSTM model definition and training logic
├── LSTM_model.pth       # Saved model weights (generated after first run)
├── Data/
│   ├── HDFCBANK.NS.csv
│   └── RELIANCENS.csv
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Usage

Place your stock CSV files inside the `Data/` folder. Each CSV must contain a `High` column.

Run the program:

```bash
python Main.py
```

You will be prompted:

```
show Output Graph press: 0
without showing Graph press: 1
Enter your choice here:
```

- Press `0` to train and view prediction plots
- Press `1` to train without displaying plots

## How It Works

1. Reads all `.csv` files from the `Data/` folder
2. Scales the `High` price column using MinMaxScaler
3. Creates sliding windows of size 10 as input sequences
4. Trains a 2-layer LSTM with hidden size 64 using the Adam optimizer and Huber loss
5. Saves the model weights after each training window
6. Predicts on the test split and optionally plots actual vs predicted prices
7. Forecasts the next 10 price points beyond the historical data

## Dataset

Currently includes:
- `HDFCBANK.NS.csv` — HDFC Bank historical stock data
- `RELIANCENS.csv` — Reliance Industries historical stock data
