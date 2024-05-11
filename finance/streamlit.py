import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Define transaction categories and corresponding keywords
categories = {
    "food": ["ORACL*WAFFLE HOUS", "85C BAKERY CAFE", "DAVES HOT CHICKEN", "TACO BELL", "BA LE RESTAURANT", "PIRANHA KILLER", "HAPPY LEMON", "CHICKEN EXPRESS", "SQ *KONA ICE", "BON K BBQ", "NINJARAMENANDROYA", "WHATABURGER", "RAISING CANES", "PORT OF SUBS", "MCDONALD'S", "CHIPOTLE", "TOUCH OF TOAST", "PANDA EXPRESS", "CHIPOTLE", "DINGTEA", "WICKED SNOW", "WINGSTOP", "PANDA EXPRESS", "CHICK-FIL-A", "210 BRAUMS STORE", "10402 CAVA CHAMPI"],
    "shopping": ["APPLE", "Amazon.com", "Nike.com", "SP WATC STUDIO", "EDFINITY.COM", "MUSINSA", "WL *STEAM", "Zara.com", "UTA BOOKSTORE", "WAL Wal-Mart", "WM SUPERCENTER", "TARGET", "BEST BUY", "APPLE.COM/BILL", "Nike.com", "Birkenstock", "TARGET", "AMZN DIGITAL", "AMZN Mktp"],
    "transportation": ["QT", "MURPHY EXPRESS", "7-ELEVEN", "Upside", "UTA PARK TRANS"],
    "finance": ["GOLDMAN SACHS", "Cash App", "DISCOVER", "TRANSFER", "BKOFAMERICA", "UTA ARL MYMAV", "UT ARLINGTON"],
    "misc": ["Zelle payment"]
}

def categorize_transaction(description):
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in description.lower():
                return category
    return "misc"


def main():
    st.title('Financial Data Analysis')

    file = st.file_uploader("Upload your bank statement", type=["csv"])

    if file is not None:
        df = pd.read_csv(file, skiprows=6)

        # Apply categorization function to the Description column
        df['Category'] = df['Description'].apply(categorize_transaction)
        df['Amount'] = df['Amount'].replace({',': ''}, regex=True).astype(float)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Running Bal.'] = df['Running Bal.'].replace({',': ''}, regex=True).astype(float)

        st.dataframe(df)

        # Group transactions by month and count the frequency
        transaction_freq_by_month = df.groupby(df['Date'].dt.to_period('M')).size()
        df_freq_by_month = pd.DataFrame({
            'Date': transaction_freq_by_month.index.strftime('%y-%m'),
            'Frequency': transaction_freq_by_month.values
        })

        st.write('Transaction Frequency')
        st.bar_chart(df_freq_by_month.set_index('Date'), use_container_width=True, color='#FF5733')



if __name__ == "__main__":
    main()
