from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

application = Flask(__name__, template_folder="C:/Users/Mitesh/CustomerSegmentationML/templates")
model = pickle.load(open('FinalYear_Project.pk1', 'rb'))

# def load_and_clean_data(file_path):
#     #Load data
#     data = pd.read_csv(file_path, sep=",", encoding="utf-8-sig", header = 0)
#     print(data.columns)  # Debugging line

#     #Convert CustomerID to string and create Amount column 
#     data['CustomerID'] = data['CustomerID'].astype(str)
#     data['Amount'] = data['Quantity']*data['UnitPrice']

#     #Computing RFM metrics
#     mt_rfm = data.groupby('CustomerID')['Amount'].sum().reset_index()
#     f_rfm = data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
#     f_rfm.columns = ['CustomerID','Frequency']
#     data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
#     date_max = max(data['InvoiceDate'])
#     data['Difference'] = date_max - data['InvoiceDate']
#     r_rfm = data.groupby('CustomerID')['Difference'].min().reset_index()
#     r_rfm['Difference'] = r_rfm['Difference'].dt.days
#     rfm = pd.merge(mt_rfm, f_rfm , on='CustomerID', how='inner')
#     rfm = pd.merge(rfm, r_rfm, on='CustomerID', how='inner')
#     rfm.columns = ['CustomerID' , 'Amount' , 'Frequency' , 'Recency']

#     #Removing outliers
#     Q1 = rfm.quantile(0.05)
#     Q2 = rfm.quantile(0.95)
#     IQR = Q2 - Q1
#     rfm= rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q2 + 1.5*IQR)]
#     rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q2 + 1.5*IQR)]
#     rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q2 + 1.5*IQR)]

#     return rfm
def load_and_clean_data(file_path):
    # Load data
    data = pd.read_csv(file_path, sep=",", encoding="utf-8-sig", header=0)
    print("Columns in dataset:", data.columns)  # Debugging line

    # Convert CustomerID to string and create Amount column
    data['CustomerID'] = data['CustomerID'].astype(str)
    data['Amount'] = data['Quantity'] * data['UnitPrice']

    # Computing RFM metrics
    mt_rfm = data.groupby('CustomerID')['Amount'].sum().reset_index()
    f_rfm = data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    f_rfm.columns = ['CustomerID', 'Frequency']
    
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    date_max = data['InvoiceDate'].max()
    data['Difference'] = (date_max - data['InvoiceDate']).dt.days

    r_rfm = data.groupby('CustomerID')['Difference'].min().reset_index()
    r_rfm.columns = ['CustomerID', 'Recency']

    # Merge RFM data
    rfm = mt_rfm.merge(f_rfm, on='CustomerID').merge(r_rfm, on='CustomerID')

    # 🔥 Force numeric conversion for all RFM columns 🔥
    rfm[['Amount', 'Frequency', 'Recency']] = rfm[['Amount', 'Frequency', 'Recency']].apply(pd.to_numeric, errors='coerce')

    # Drop NaN values after conversion
    rfm.dropna(inplace=True)

    # Debugging: Print data types to confirm numeric conversion
    print("Data types after conversion:\n", rfm.dtypes)

    # 🔥 Debugging Step: Print unique values of each column before quantile 🔥
    print("Unique values in Amount:", rfm['Amount'].unique()[:10])
    print("Unique values in Frequency:", rfm['Frequency'].unique()[:10])
    print("Unique values in Recency:", rfm['Recency'].unique()[:10])

    # Removing outliers
    Q1 = rfm[['Amount', 'Frequency', 'Recency']].quantile(0.05)
    Q2 = rfm[['Amount', 'Frequency', 'Recency']].quantile(0.95)
    IQR = Q2 - Q1

    rfm = rfm[
        (rfm['Amount'] >= Q1['Amount'] - 1.5 * IQR['Amount']) & (rfm['Amount'] <= Q2['Amount'] + 1.5 * IQR['Amount']) &
        (rfm['Recency'] >= Q1['Recency'] - 1.5 * IQR['Recency']) & (rfm['Recency'] <= Q2['Recency'] + 1.5 * IQR['Recency']) &
        (rfm['Frequency'] >= Q1['Frequency'] - 1.5 * IQR['Frequency']) & (rfm['Frequency'] <= Q2['Frequency'] + 1.5 * IQR['Frequency'])
    ]

    return rfm



def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    df_rfm = rfm[['Amount','Frequency','Recency']]
    scaler = StandardScaler()
    #Fit Transform
    df_rfm_scaled = scaler.fit_transform(df_rfm)
    df_rfm_scaled = pd.DataFrame(df_rfm_scaled)
    df_rfm_scaled.columns = ['Amount', 'Frequency', 'Recency']

    return rfm,df_rfm_scaled

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df = preprocess_data(file_path) [1] 
    results_df = model.predict(df) 

    #results_df = pd. DataFrame(results_df)
    df_with_id = preprocess_data(file_path)[0]
    df_with_id['Cluster_Id'] = results_df

    # Generate the images and save them
    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id, hue='Cluster_Id')
    recency_img_path = 'static\ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id, hue='Cluster_Id')
    frequency_img_path = 'static\ClusterId_Frequency.png'
    plt.savefig(frequency_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Amount', data=df_with_id, hue='Cluster_Id')
    amount_img_path = 'static\ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()

    response = {'recency_img': recency_img_path,
                'frequency_img': frequency_img_path,
                'amount_img': amount_img_path}
    
    return json.dumps(response)

if __name__=="__main__":
    application.run(debug=True, port=5500)
