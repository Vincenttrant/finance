import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import plotly.express as px
import streamlit as st
from prophet import Prophet

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


def categorize_transaction(description):
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in description.lower():
                return category
    return "Misc"


# automated categorization of transactions using the trained model (Naive Bayes classifier)
def machine_learning(data, data2):
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
    # Add categorization to the data for testing model
    data = pd.read_csv("Data/testData.csv", skiprows=6)
    df = pd.DataFrame(data, columns=["Date", "Description", "Amount", "Running Bal."])

    df['Category'] = df['Description'].apply(categorize_transaction)
    df['Amount'] = df['Amount'].replace({',': ''}, regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Running Bal.'] = df['Running Bal.'].replace({',': ''}, regex=True).astype(float)

    # Save the categorized data to a new CSV file (optional)
    df.to_csv("Data/testCategorizedData.csv", index=False)

    data = pd.read_csv("Data/testCategorizedData.csv")

    st.set_page_config(page_title="Finance Dashboard", layout='wide')

    page = "Finance Dashboard"
    with st.sidebar:
        # create two buttons for the user to select 1. Dashboard 2. Future Output
        st.title('Navigation :heavy_exclamation_mark:')
        page = st.selectbox('Go to :car:', ['Finance Dashboard', 'Future Finance'])

    st.title(f'{page} :chart_with_upwards_trend:')

    file = st.file_uploader("Upload your bank statement", type=["csv"])

    if file is not None:
        df = pd.read_csv(file, skiprows=6)

        machine_learning(data, df)
        # Apply categorization function to the Description column
        df['Amount'] = df['Amount'].replace({',': ''}, regex=True).astype(float)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Running Bal.'] = df['Running Bal.'].replace({',': ''}, regex=True).astype(float)

        st.subheader('Raw Data')
        st.dataframe(df.iloc[1:], use_container_width=True)

        if page == 'Finance Dashboard':
            col1, col2 = st.columns(2)

            with col1:
                # Plotting a line plot of the running balance using Plotly
                fig = px.line(
                    df,
                    x='Date',
                    y='Running Bal.',
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )

                st.subheader('Total Running Balance')
                st.plotly_chart(fig, use_container_width=True, height=400)

            with col2:
                transaction_freq_by_month = df.groupby(df['Date'].dt.to_period('M')).size()
                df_freq_by_month = pd.DataFrame({
                    'Date': transaction_freq_by_month.index.strftime('%y-%m'),
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
            df_forecast = df[['Date', 'Running Bal.']].rename(columns={'Date': 'ds', 'Running Bal.': 'y'})

            # Initialize and fit the Prophet model
            model = Prophet()
            model.fit(df_forecast)

            # Create future dates dataframe
            future_dates = model.make_future_dataframe(periods=30)  # Forecast for the next 30 days

            # Predict future values
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

            st.subheader('Next Month Balance Forecast')
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = model.plot_components(future_forecast)
            st.subheader('Forecast Components')
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
