import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Activation Functions (Vectorized) ===
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

def ReLU(x): return np.maximum(0, x)
def ReLU_derivative(x): return np.where(x > 0, 1, 0)

def Leaky_ReLU(x, epsilon=0.01): return np.where(x > 0, x, epsilon * x)
def Leaky_ReLU_derivative(x, epsilon=0.01): return np.where(x > 0, 1, epsilon)

def Tanh(x): return np.tanh(x)
def Tanh_derivative(x): return 1 - np.tanh(x)**2

def linear(x): return x
def linear_derivative(x): return np.ones_like(x)

def softplus(x): return np.log1p(np.exp(x))
def softplus_derivative(x): return 1 / (1 + np.exp(-x))


# === Neural Network Class ===
class ClimateNN:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.activation_name = activation
        self.activations = {
            'linear': (linear, linear_derivative),
            'softplus': (softplus, softplus_derivative),
            'ReLU': (ReLU, ReLU_derivative),
            'leaky_relu': (Leaky_ReLU, Leaky_ReLU_derivative),
            'sigmoid': (sigmoid, sigmoid_derivative),
            'tanh': (Tanh, Tanh_derivative)
        }

    def forward(self, X):
        act, _ = self.activations[self.activation_name]
        self.Z1 = np.dot(X, self.w1) + self.b1
        self.A1 = act(self.Z1)
        self.Z2 = np.dot(self.A1, self.w2) + self.b2
        return self.Z2

    def backward(self, X, y, output, learning_rate):
        _, d_func = self.activations[self.activation_name]
        m = y.shape[0]

        dZ2 = output - y
        dw2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.w2.T) * d_func(self.A1)
        dw1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate=0.01, epochs=1000, verbose=False):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if verbose and epoch % 200 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)


# === Helper Functions ===
def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0), np.mean(X, axis=0), np.std(X, axis=0)

def denormalize_data(X, mean, std):
    return X * std + mean

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# === Full Training Wrapper ===
def train_and_predict(df, activation, hidden_size=5, learning_rate=0.01, epochs=800):
    features = ["TMAX", "TMIN", "PRCP"]
    
    X_raw = df[features].values[:-1]
    y_raw = df[['TMAX', 'TMIN']].values[1:]

    X, mean_X, std_X = normalize_data(X_raw)
    y, mean_y, std_y = normalize_data(y_raw)

    nn = ClimateNN(input_size=len(features), hidden_size=hidden_size, 
                   output_size=2, activation=activation)

    nn.train(X, y, learning_rate, epochs, verbose=False)

    last_day = normalize_data(df[features].values[-1:])[0]
    prediction = nn.predict(last_day)
    prediction = denormalize_data(prediction, mean_y, std_y)

    return prediction


# === MAIN PROGRAM ===
def main():

    df = pd.read_csv("data.csv")

    df = df.rename(columns={
        "TMAX (Degrees Fahrenheit)": "TMAX",
        "TMIN (Degrees Fahrenheit)": "TMIN",
        "TAVG (Degrees Fahrenheit)": "TAVG",
        "PRCP (Inches)": "PRCP"
    })

    df = df.dropna()

    print("\nDataset Successfully Loaded:\n", df.head())

    activations = ['linear','softplus','ReLU','leaky_relu','sigmoid','tanh']
    loss_table = {}

    actual_next_day = df[['TMAX','TMIN']].values[1]

    print("\n=== Training & Evaluating Activation Functions ===")

    for act in activations:
        pred = train_and_predict(df, activation=act)
        error = mse(actual_next_day, pred)
        loss_table[act] = error
        print(f"{act} → MSE: {error:.4f} | Predicted: {pred[0]}")

    best = min(loss_table, key=loss_table.get)

    # Plot results
    plt.figure(figsize=(9,5))
    plt.bar(loss_table.keys(), loss_table.values(), color='skyblue')
    plt.title("Activation Function Performance")
    plt.ylabel("Mean Squared Error")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

    print(f"\nBest performing activation function: **{best.upper()}** with MSE = {loss_table[best]:.4f}")


if __name__ == "__main__":
    main()
