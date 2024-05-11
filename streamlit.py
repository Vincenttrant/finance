import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Define transaction categories and corresponding keywords
categories = {
    "Food": ["ORACL*WAFFLE HOUS", "85C BAKERY CAFE", "DAVES HOT CHICKEN", "TACO BELL", "BA LE RESTAURANT", "PIRANHA KILLER", "HAPPY LEMON", "CHICKEN EXPRESS", "SQ *KONA ICE", "BON K BBQ", "NINJARAMENANDROYA", "WHATABURGER", "RAISING CANES", "PORT OF SUBS", "MCDONALD'S", "CHIPOTLE", "TOUCH OF TOAST", "PANDA EXPRESS", "CHIPOTLE", "DINGTEA", "WICKED SNOW", "WINGSTOP", "PANDA EXPRESS", "CHICK-FIL-A", "210 BRAUMS STORE", "10402 CAVA CHAMPI"],
    "Shopping": ["APPLE", "Amazon.com", "Nike.com", "SP WATC STUDIO", "EDFINITY.COM", "MUSINSA", "WL *STEAM", "Zara.com", "UTA BOOKSTORE", "WAL Wal-Mart", "WM SUPERCENTER", "TARGET", "BEST BUY", "APPLE.COM/BILL", "Nike.com", "Birkenstock", "TARGET", "AMZN DIGITAL", "AMZN Mktp"],
    "Transportation": ["QT", "MURPHY EXPRESS", "7-ELEVEN", "Upside", "UTA PARK TRANS"],
    "Finance": ["GOLDMAN SACHS", "Cash App", "DISCOVER", "TRANSFER", "BKOFAMERICA", "UTA ARL MYMAV", "UT ARLINGTON"],
    "Misc": ["Zelle payment"]
}

def categorize_transaction(description):
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in description.lower():
                return category
    return "Misc"


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

        st.write('Transaction Frequency')
        st.plotly_chart(fig)




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

        st.write('Spending Categories')
        st.plotly_chart(fig)

        # Plotting a line plot of the running balance using Plotly
        fig = px.line(
            df,
            x='Date',
            y='Running Bal.',
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.write('Total Running Balance')
        st.plotly_chart(fig)

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

        st.write('Running Totals for Each Category')
        st.plotly_chart(fig)





if __name__ == "__main__":
    main()
