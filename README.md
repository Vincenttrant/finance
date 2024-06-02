# Finance Dashboard and Forecasting

## [Dashboard](https://futurefinance.streamlit.app/)

## Overview

This project is designed to provide a comprehensive finance dashboard and future financial forecasting using various machine learning techniques and the Prophet model for time series forecasting. The application includes transaction categorization, visualization of financial data, and forecasting future balances.

## Features

- **Transaction Categorization**: Uses a Naive Bayes classifier to categorize transactions into predefined categories such as Food, Shopping, Transportation, Finance, and Miscellaneous.
- **Visualization**: Interactive visualizations of running balances, transaction frequencies, and spending categories using Plotly and Streamlit.
- **Future Forecasting**: Forecast future financial balances using the Prophet model with various hyperparameters for precise predictions.
- **Database Integration**: SQLite database to store and manage transaction data.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/finance.git
    ```
2. Navigate to the project directory:
    ```bash
    cd finance
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    streamlit run machineLearning.py
    ```
2. Upload your monthly bank statement in CSV format.
3. Explore the Finance Dashboard for insights into your financial data.
4. Navigate to the Future Finance section to see predictions for your future balances.

## Files

- **database.py**: Handles the creation, saving, and loading of transactions from a SQLite database.
- **machineLearning.py**: Contains the main application logic including data preprocessing, categorization, visualization, and forecasting.

## Example

![Finance Dashboard Screenshot](https://i.gyazo.com/2ae2f95eb65285081b9c9b0e0b465aad.png)
![Future Forecasting Screenshot](https://i.gyazo.com/e445507381d51068a02476357ec8f44a.png)
## Dependencies

- `pandas`
- `matplotlib`
- `nltk`
- `prophet`
- `scikit-learn`
- `plotly`
- `streamlit`
- `sqlite3`
