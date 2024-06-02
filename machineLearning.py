import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from prophet.diagnostics import performance_metrics, cross_validation
from prophet.plot import plot_cross_validation_metric, plot_components_plotly
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import plotly.express as px
import streamlit as st
from prophet import Prophet
import re
import database

nltk.download('stopwords')

# manual categorization of transactions for training the model
categories = {
    "Food": ["FOODS", "EATS", "TACO DINER", "NICK & SAM'S", "LOCHLANDS", "WHISKEY CAKE", "IN-N-OUT BURGER",
             "GRAND LUX CAFE", "THE RUSTIC", "EL FENIX", "CANTINA LAREDO", "FOGO DE CHAO", "P.F. CHANG'S",
             "CHICKEN N PICKLE", "BURGER KING", "SUBWAY", "STARBUCKS", "BLUE BOTTLE COFFEE", "OLIVE GARDEN", "KFC",
             "DUNKIN' DONUTS", "PEET'S COFFEE", "TAQUERIA TAXCO", "IHOP", "ORACL*WAFFLE HOUS", "85C BAKERY CAFE",
             "DAVES HOT CHICKEN", "TACO BELL", "BA LE RESTAURANT", "PIRANHA KILLER", "HAPPY LEMON", "CHICKEN EXPRESS",
             "SQ *KONA ICE", "BON K BBQ", "NINJARAMENANDROYA", "WHATABURGER", "RAISING CANES", "PORT OF SUBS",
             "MCDONALD'S", "CHIPOTLE", "TOUCH OF TOAST", "PANDA EXPRESS", "CHIPOTLE", "DINGTEA", "WICKED SNOW",
             "WINGSTOP", "PANDA EXPRESS", "CHICK-FIL-A", "210 BRAUMS STORE", "10402 CAVA CHAMPI"],
    "Shopping": ["CUT", "HAIRCUT", "CHINGONCUT", "NEIMAN MARCUS", "LOUIS VUITTON", "GALLERIA", "WILLIAMS SONOMA",
                 "THE SHOPS AT LEGACY", "THE NORTH FACE", "DICK'S SPORTING GOODS", "SPOTIFY", "URBAN OUTFITTERS", "GAP",
                 "NORDSTROM", "PEETS COFFEE", "ASOS", "H&M", "NETFLIX.COM", "HOME DEPOT", "COSTCO", "PRIME VIDEO",
                 "KROGER", "HULU.COM", "WALGREENS", "LOWE'S", "EBAY", "eBayCommerce", "Alibaba.com", "APPLE",
                 "Amazon.com", "Nike.com", "SP WATC STUDIO", "EDFINITY.COM", "MUSINSA", "WL *STEAM", "Zara.com",
                 "UTA BOOKSTORE", "WAL Wal-Mart", "WM SUPERCENTER", "TARGET", "BEST BUY", "APPLE.COM/BILL", "Nike.com",
                 "Birkenstock", "TARGET", "AMZN DIGITAL", "AMZN Mktp"],
    "Transportation": ["GAS", "CAR", "BUC-EE'S", "UBER RIDE", "LYFT RIDE", "FRONTIER", "QT", "MURPHY EXPRESS",
                       "7-ELEVEN", "Upside", "UTA PARK TRANS"],
    "Finance": ["VENMO", "PAYPAL", "CHASE BANK", "GOLDMAN SACHS", "Cash App", "DISCOVER", "TRANSFER", "BKOFAMERICA",
                "UTA ARL MYMAV", "UT ARLINGTON"],
    "Misc": ["Zelle payment", "EXPRT TA", "Beginning balance"]
}


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))

    # Remove stopwords like 'the', 'and', 'is'...
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def categorize_transaction(description):
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in description.lower():
                return category
    return "Misc"


# automated categorization of transactions using the trained model (Naive Bayes classifier)
def machine_learning(data, data2):
    # Preprocess the text data
    data['Description'] = data['Description'].apply(preprocess_text)
    data2['Description'] = data2['Description'].apply(preprocess_text)

    # Training data
    X_train = data['Description']
    y_train = data['Category']
    X_test = data2['Description']

    # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    data2['Category'] = y_pred

    print(classification_report(y_train, model.predict(X_train)))


def main():
    # Initialize the database
    # Add categorization to the data for testing model
    # data = pd.read_csv("Data/testData.csv", skiprows=6)
    # df = pd.DataFrame(data, columns=["Date", "Description", "Amount", "Running_Bal"])
    #
    # df['Category'] = df['Description'].apply(categorize_transaction)
    # df['Amount'] = df['Amount'].replace({',': ''}, regex=True).astype(float)
    # df['Date'] = pd.to_datetime(df['Date'])
    # df['Running_Bal'] = df['Running_Bal'].replace({',': ''}, regex=True).astype(float)

    # database.save_transaction(df)

    data = database.load_transactions()

    st.set_page_config(page_title="Finance Dashboard", layout='wide')

    page = "Finance Dashboard"
    with st.sidebar:
        # create two buttons for the user to select 1. Dashboard 2. Future Output
        st.title('Navigation :heavy_exclamation_mark:')
        page = st.selectbox('Go to :car:', ['Finance Dashboard', 'Future Finance'])

    st.title(f'{page} :chart_with_upwards_trend:')

    file = st.file_uploader("Upload your bank :red[montly] statement", type=["csv"])

    if file is not None:
        df = pd.read_csv(file, skiprows=6)

        machine_learning(data, df)

        # Apply categorization function to the Description column
        df['Amount'] = df['Amount'].replace({',': ''}, regex=True).astype(float)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Running_Bal'] = df['Running_Bal'].replace({',': ''}, regex=True).astype(float)

        st.subheader('Raw Data')
        st.dataframe(df.iloc[1:], use_container_width=True)

        if page == 'Finance Dashboard':
            col1, col2 = st.columns(2)

            with col1:
                # Plotting a line plot of the running balance using Plotly
                fig = px.line(
                    df,
                    x='Date',
                    y='Running_Bal',
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )

                st.subheader('Total Running Balance')
                st.plotly_chart(fig, use_container_width=True, height=400)

            with col2:
                transaction_freq_by_month = df.groupby(df['Date'].dt.to_period('M')).size()
                df_freq_by_month = pd.DataFrame({
                    'Date': transaction_freq_by_month.index.strftime('%B'),
                    'Frequency': transaction_freq_by_month.values
                })

                fig = px.bar(
                    df_freq_by_month,
                    x='Date',
                    y='Frequency',
                )

                st.subheader('Transaction Frequency')
                st.plotly_chart(fig, use_container_width=True, height=400)

            st.markdown("---")

            col1, col2 = st.columns([0.4, 0.5])

            with col1:
                # Plotting a pie chart of the spending categories using Plotly
                spendingCategoriesTotal = {}
                for category in categories.keys():
                    if df[df['Category'] == category]['Amount'].sum() < 0:
                        spendingCategoriesTotal[category] = -df[df['Category'] == category]['Amount'].sum()

                fig = px.pie(
                    values=spendingCategoriesTotal.values(),
                    names=spendingCategoriesTotal.keys(),
                    color=spendingCategoriesTotal.keys(),
                    color_discrete_sequence=px.colors.qualitative.Safe
                )

                st.subheader('Spending Categories')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Plotting a line plot of the running totals for each category using Plotly
                category_running_totals = {category: [] for category in categories.keys()}
                running_totals = {category: 0 for category in categories.keys()}
                for index, row in df.iterrows():
                    category = row['Category']
                    amount = row['Amount']
                    if amount < 0:
                        running_totals[category] += -amount

                    for key in category_running_totals.keys():
                        category_running_totals[key].append(running_totals[key])

                df_running_totals = pd.DataFrame(category_running_totals)
                fig = px.line(
                    df_running_totals,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )

                st.subheader('Running Totals for Each Category')
                st.plotly_chart(fig, use_container_width=True)

        elif page == 'Future Finance':
            # Prepare data for forecasting
            df_forecast = df[['Date', 'Running_Bal']].rename(columns={'Date': 'ds', 'Running_Bal': 'y'})


            # Initialize and fit the Prophet model
            model = Prophet(
                changepoint_prior_scale=0.1,
                # seasonality_mode='multiplicative',
                # seasonality_prior_scale=2,
                holidays_prior_scale=10,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            model.fit(df_forecast)

            # Create future dates dataframe
            future_dates = model.make_future_dataframe(periods=30)


            forecast = model.predict(future_dates)

            # Adjust the forecast to replace any negative values with the last non-negative value
            last_valid_value = None
            for i in range(len(forecast)):
                if forecast.loc[i, 'yhat'] < 0:
                    forecast.loc[i, 'yhat'] = last_valid_value
                else:
                    last_valid_value = forecast.loc[i, 'yhat']

            future_forecast = forecast[forecast['ds'] > df_forecast['ds'].max()]

            fig1 = px.line(
                future_forecast,
                x='ds',
                y='yhat'
            )

            # Plotting the forecast and components using Plotly
            st.subheader('Next Month Balance Forecast')
            st.markdown(f'Your balance after 30 days is expected to be :red[${future_forecast["yhat"].iloc[-1]:.2f}]')
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = plot_components_plotly(model, future_forecast)
            st.subheader('Forecast Components')
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
