import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# folder_path = "C:\\Users\\anude\\Documents\\Programming\\Mechian_Learning\\Python\\Data"

def ProcessUnit(fileName,folder_path,user):
    
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # time_period = "minute"
    col = "High" # which column you want to read the program

    data_path = fileName # file path you point to your .csv file

    file_path = os.path.join(folder_path,data_path)

    model_path = "LSTM_model.pth" # No change require

    # adjust with how much data is giving as Input it control the memory overflow
    stepSize = 1.0 #->value varies from 1 to 0.01 , for show the best output put value equel to 1

    # it is the batch size 
    window_size = 10 # No change require

    # loop times
    epochs = 50 # No change require


    try:
        df = pd.read_csv(file_path)
        df[col] = df[col].astype(float)
        dataRaw = df[[col]]
    except Exception as e:
        print("Error loading dataset:", e)
        exit()


    print("Size of Dataset: ",len(dataRaw))
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)


    class CryptoLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super(CryptoLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.linear(out[:, -1, :])
            return out

    model = CryptoLSTM().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from disk")
    else:
        print("Starting with fresh model")

    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    start = 0
    end = stepSize
    

    while end <= 1.0:
        print(f"\nTraining on window: {start:.2f} → {end:.2f}")
        Dstart = int(len(dataRaw) * start)
        Dend = int(len(dataRaw) * end)
        data = dataRaw[Dstart:Dend]

        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(data)
        X, y = create_sequences(scaled_close, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


        model.train()
        for epoch in range(epochs):
            output = model(X_train)
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (epoch+1) % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
       

        if torch.isnan(loss):
            print("Model skipping due to NaN loss")
        else:
            torch.save(model.state_dict(), model_path)
            print("\nModel saved")

        start = end
        end += stepSize
        if end>1.0 and end<2.0 and count==0:
            print(f"Entered: {end}")
            count = 1
            end = 1.0
        else:
            pass

        

    model.eval()
    with torch.no_grad():
        predicted = model(X_test).cpu().numpy()
        predicted = scaler.inverse_transform(predicted)
        actual = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))



    def PLot():
        plt.figure(figsize=(12, 5))
        plt.plot(actual, label="Actual Price")
        plt.plot(predicted, label="Predicted Price")

        plt.xticks(np.linspace(0, len(actual)-1, 10, dtype=int)) 
        plt.xlim(0, len(actual)-1)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        plt.xlabel("Data Point Index")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.title(f"training section of {fileName}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        def forecast_future(model, data, steps=200, window_size=10):
            model.eval()
            forecast = []
            last_window = data[-window_size:].copy()
            for _ in range(steps):
                input_seq = torch.tensor(last_window.reshape(1, window_size, 1), dtype=torch.float32).to(device)
                with torch.no_grad():
                    next_value = model(input_seq).cpu().numpy()
                forecast.append(next_value[0][0])
                last_window = np.vstack([last_window[1:], next_value])
            return np.array(forecast).reshape(-1, 1)

        future_scaled = forecast_future(model, scaled_close, steps=10, window_size=window_size)
        future_prices = scaler.inverse_transform(future_scaled)


        total_points = len(scaled_close) + len(future_prices)
        x_values = np.arange(total_points)  

        plt.figure(figsize=(12, 5))
        plt.plot(x_values[:len(scaled_close)], scaler.inverse_transform(scaled_close.reshape(-1, 1)), 
                 label="Historical Price")
        plt.plot(x_values[len(scaled_close):], future_prices, 
                 label="Future Forecast", color='green', linewidth=2)

        # Set explicit ticks for full range
        plt.xticks(np.linspace(0, total_points-1, 10, dtype=int))
        plt.xlim(0, total_points-1)

        # Format y-axis
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.xlabel("Data Point Index")
        plt.ylabel("Price ($)")
        plt.title(f"Prediction from {fileName} dataset")

        # Add vertical line at forecast point
        plt.axvline(x=len(scaled_close), color='red', linestyle='--', alpha=0.5, label='Forecast Start')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if user==0:
        PLot()
    else:
        pass


# ProcessUnit("3IINFOTECH.BO.csv") #->file name inside with .csv