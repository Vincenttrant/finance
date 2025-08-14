# Finance Dashboard & Forecasting

🔗 **Live Demo:** [futurefinance.streamlit.app](https://futurefinance.streamlit.app/)  

---

## Overview
The **Finance Dashboard** is an interactive web app for analyzing personal finances and forecasting future balances.  
It enables users to upload bank statements, automatically categorize transactions, visualize spending patterns, and project future balances up to **12 months ahead**.

The project integrates:
- **Machine Learning** for automated transaction categorization
- **Interactive Data Visualization** for trends and insights
- **Time Series Forecasting** for forward-looking financial planning

---

## Key Features
- **Automated Transaction Categorization**  
  - Implemented a **Naive Bayes classifier** for categorizing expenses into *Food, Shopping, Transportation, Finance, Miscellaneous*.
  - Achieved **~90% accuracy** on hundreds of test transactions.

- **Dynamic Data Visualization**  
  - Interactive charts for **running balances**, **transaction frequencies**, and **category spending** using **Plotly + Streamlit**.
  - Filter data by time range or category.

- **Future Balance Forecasting**  
  - Integrated **Facebook Prophet** to predict balances up to 12 months ahead.
  - Forecast dynamically updates with new uploads.

- **Persistent Data Storage**  
  - Utilized **SQLite** to store and manage transaction data.

---

## Technical Highlights
- **Tech Stack:** Python, Pandas, Scikit-learn, Prophet, Streamlit, Plotly, SQLite
- **Machine Learning:** Naive Bayes classifier with NLTK-based text preprocessing
- **Visualization:** Custom dashboards with Plotly inside Streamlit
- **Database:** SQLite backend with modular read/write functions

---

## Installation & Setup
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/finance-dashboard.git
cd finance-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run machineLearning.py
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


## 📌 Interview Reference Sheet (For Personal Use)

**Project Name:** Finance Dashboard & Forecasting  
**Tech Stack:** Python, Pandas, Scikit-learn, Prophet, Streamlit, Plotly, SQLite  

---

### **Problem & Motivation**
> Many people track their spending but lack an easy way to categorize transactions automatically and forecast their balances. I built a tool to solve that.

---

### **Solution**
- Upload bank statements in CSV format  
- Automatic expense categorization using **Naive Bayes classifier (~90% accuracy)**  
- Interactive dashboards for **spending trends, balances, and categories**  
- Balance forecasting for up to **12 months** using **Facebook Prophet**  

---

### **Technical Challenges**
- **Messy Transaction Data** → Used NLP preprocessing to clean merchant names/descriptions before classification  
- **Limited Labeled Data for Testing** → Manually created a labeled dataset of categorized transactions to evaluate and tune the classifier.  
- **Forecasting Accuracy** → Tuned Prophet hyperparameters to improve predictions  
- **Real-Time Updates** → Streamlit app dynamically updates with new uploads  

---

### **Key Metrics**
- **90%** classification accuracy on test dataset  
- **< 2 sec** average transaction categorization speed for hundreds of rows  
- Forecast horizon: **12 months**, auto-updating  

---

### **Impact**
- Saves time by automating categorization of large transaction lists  
- Empowers users to plan ahead based on accurate, visual forecasts  

---

### **Scalability**
- Modular architecture allows swapping the classifier for more advanced ML models (e.g., Random Forest, Gradient Boosting).  
- Database layer can migrate from SQLite to a cloud solution (e.g., AWS RDS) for multi-user access.  
- Current accuracy is limited by inconsistent and incomplete bank statement descriptions — adding an enrichment pipeline (e.g., matching transactions to merchant databases) could improve future categorization accuracy.  

---

### **STAR Interview Format**
**S – Situation:**  
While tracking my own expenses, I noticed that most bank statements have messy, inconsistent descriptions, and manually categorizing them is time-consuming. I also found that most budgeting tools focus on historical data but don’t give actionable forecasts.  

**T – Task:**  
I set out to build an interactive tool that would:
1. Automatically categorize uploaded bank statement transactions with high accuracy.  
2. Provide visual insights into spending trends and balances.  
3. Forecast future balances for up to 12 months to help with planning.  

**A – Actions:**  
- **Data Collection & Preprocessing:** Gathered sample bank statements, cleaned transaction descriptions using NLP techniques (tokenization, stopword removal, normalization).  
- **Model Development:** Chose a Naive Bayes classifier for speed and interpretability, trained it on a manually labeled dataset to ensure reliable evaluation, and iterated to reach ~90% accuracy.  
- **Visualization:** Used Plotly and Streamlit to create dynamic charts for running balances, category spending, and transaction frequencies, with filtering options.  
- **Forecasting Engine:** Integrated Facebook Prophet, tuned seasonal and trend parameters, and designed the system to update predictions dynamically with new uploads.  
- **Backend & Persistence:** Built a lightweight SQLite database layer to store historical transactions and reduce repeated processing.  
- **UI/UX Design:** Designed a clean, responsive dashboard that updates in real-time and requires minimal user input.  

**R – Results:**  
- Delivered a fully functional, browser-accessible finance dashboard.  
- Achieved ~90% classification accuracy, automating categorization for hundreds of transactions in under 2 seconds.  
- Forecasting feature projects balances 12 months ahead, enabling proactive financial planning.  
- Built a modular system that can scale with better ML models and improved transaction metadata.  
 
