**NN-Weather**

NN-Weather is a simple neural network model built in Python to predict the next day's maximum and minimum temperatures based on historical daily temperature data. The model uses the previous 7 days of temperatures as input and evaluates multiple activation functions to determine the most accurate predictions.

**Features**

- Predicts next-day **TMAX** and **TMIN** using the last 7 days of data.
- Supports multiple activation functions: linear, softplus, ReLU, leaky_relu, sigmoid, tanh.
- Automatically normalizes input and output data for better model stability.
- Provides a bar chart comparing mean squared error (MSE) for different activation functions.
- Returns the best performing Activation function so the user can view the difference in performances across each one

**Requirements**

- Python 3.8 or higher
- Packages:
  - numpy
  - pandas
  - matplotlib

**Usage**

The script will:

- Load the dataset and normalize data.
- Train a neural network on the last 7 days of temperature data.
- Evaluate multiple activation functions and report their mean squared error (MSE).
- Predict the next day's temperatures using the best performing activation function.
- Display a bar chart of MSEs for each activation function.

**Dataset**

The data I used in this project was pulled from [**https://www.ncei.noaa.gov/access/past-weather/40.987647187338084,-73.94764870907557,40.55523303389395,-72.97697877927584**](https://www.ncei.noaa.gov/access/past-weather/40.987647187338084,-73.94764870907557,40.55523303389395,-72.97697877927584)

I trimmed the dataset to only include maximum and minimum temperatures along with the date of the collected data.

This dataset is extremely reliable as it is from a government domain.

**Sample output**

Dataset Successfully Loaded:

Date TMAX TMIN

0 7/17/48 74.0 64.0

1 7/18/48 81.0 70.0

2 7/19/48 85.0 70.0

3 7/20/48 84.0 69.0

4 7/21/48 85.0 71.0

Training & Evaluating Activation Functions

linear → MSE: 7.0541 | Predicted: \[52.20575148 38.29984288\]

softplus → MSE: 7.1333 | Predicted: \[52.47634031 38.4561771 \]

ReLU → MSE: 6.0439 | Predicted: \[51.86850198 37.74674037\]

leaky_relu → MSE: 6.0818 | Predicted: \[51.88051699 37.76971421\]

sigmoid → MSE: 6.5549 | Predicted: \[51.5875338 37.6999791\]

tanh → MSE: 6.8746 | Predicted: \[50.95183309 37.11137598\]

Best performing activation function: \*\*RELU\*\* with MSE = 6.0439

**Notes**

- The model works best with continuous, daily temperature data. Missing values should be cleaned before training.
- Predictions are more accurate if the dataset contains enough historical data to capture trends and seasonal patterns.
- The dataset used only represents weather from JFK International airport from the dates 7/17/1948 to 11/15/2025
- This project can be repeatable with any set of temperature data from any region, happy modeling :).