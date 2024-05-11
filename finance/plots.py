import pandas as pd
import seaborn as sns
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

def spending_sum(df, category):
    if df[df['Category'] == category]['Amount'].sum() < 0:
        return -df[df['Category'] == category]['Amount'].sum()

def main():
    # Load the data from Data/aprilData.csv into a DataFrame
    data = pd.read_csv("Data/yearData.csv", skiprows=6)
    df = pd.DataFrame(data, columns=["Date", "Description", "Amount", "Running Bal."])

    # Apply categorization function to the Description column
    df['Category'] = df['Description'].apply(categorize_transaction)
    df['Amount'] = df['Amount'].replace({',': ''}, regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Running Bal.'] = df['Running Bal.'].replace({',': ''}, regex=True).astype(float)



    # Plotting histplot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Date', color='lightcoral', kde=True)
    plt.title('Transaction Frequency by Month')
    plt.xlabel('Months')
    plt.ylabel('Transactions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plotting a pie chart of the spending categories using Matplotlib
    spendingCategoriesTotal = {}
    for category in categories.keys():
        if spending_sum(df, category) is not None:
            spendingCategoriesTotal[category] = spending_sum(df, category)

    plt.figure(figsize=(10, 10))
    plt.pie(spendingCategoriesTotal.values(), labels=spendingCategoriesTotal.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Spending Categories')
    plt.axis('equal')
    plt.show()

    # Plotting a line plot of the running balance using Seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Running Bal.', color='lightcoral')
    plt.fill_between(df['Date'], df['Running Bal.'], color='lightcoral', alpha=0.3)
    plt.title('Total Running Balance')
    plt.xlabel('Date')
    plt.ylabel('Running Balance (Total)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plotting a line plot of the running totals for each category using Seaborn
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
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_running_totals, dashes=False)
    plt.title('Running Totals for Each Category')
    plt.xlabel('Transaction Index')
    plt.ylabel('Running Total')
    plt.tight_layout()
    plt.show()

    # Streamlit
    st.title('Financial Data Analysis')
    st.write('This is a financial data analysis dashboard.')

    st.header('Transaction Frequency by Month')

    file = st.file_uploader("Upload your bank statement", type=["csv"])

    print("file input: ", file)

    if file is None:
        st.write("Please upload a file")
    else:
        df = pd.read_csv(file)
        st.write(df)




if __name__ == "__main__":
    main()
