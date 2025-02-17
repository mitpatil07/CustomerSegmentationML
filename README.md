# RFM Customer Segmentation using Flask

## 📌 Overview
This mini-project implements **RFM (Recency, Frequency, Monetary) Customer Segmentation** using Flask. It allows users to upload a dataset, preprocess the data, apply a trained **K-Means clustering model**, and visualize the results with **Seaborn plots**.

## 🚀 Features
- Upload a CSV file containing customer transaction data.
- Clean and preprocess the data (handle missing values, remove outliers, etc.).
- Compute **RFM metrics** (Recency, Frequency, and Monetary Value).
- Normalize the data and apply **K-Means clustering**.
- Generate visualizations for customer segments.
- Serve results as JSON response.

## 🛠 Tech Stack
- **Backend:** Flask, Pandas, NumPy, scikit-learn
- **Visualization:** Seaborn, Matplotlib
- **Modeling:** K-Means Clustering
- **Deployment:** Flask server

## 📂 Project Structure
```
D:/ME/
│── application.py  # Main Flask app
│── FinalYear_Project.pk1  # Trained K-Means model (Pickle file)
│── templates/
│   ├── index.html  # Frontend template
│── static/
│   ├── ClusterId_Recency.png  # Generated plots
│   ├── ClusterId_Frequency.png
│   ├── ClusterId_Amount.png
│── README.md  # Project Documentation
```

## 🔧 Setup and Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/mitpatil07/CustomerSegmentationML.git
cd CustomerSegmentationML
```

### 2️⃣ Install Dependencies
Make sure you have Python installed. Then install the required libraries:
```sh
pip install flask pandas numpy seaborn matplotlib scikit-learn
```

### 3️⃣ Run the Flask Application
```sh
python application.py
```
The server will start at `http://127.0.0.1:5500/`.

## 🏗 API Endpoints
### ➤ **Home Page**
**`GET /`**  
Renders the **index.html** page.

### ➤ **Predict Customer Segments**
**`POST /predict`**  
Uploads a CSV file, processes the data, applies clustering, and returns image paths for visualization.

#### **Request Format:**
- `file`: CSV file containing customer transaction data.

#### **Response Format:**
```json
{
  "recency_img": "D:/ME/static/ClusterId_Recency.png",
  "frequency_img": "D:/ME/static/ClusterId_Frequency.png",
  "amount_img": "D:/ME/static/ClusterId_Amount.png"
}
```

## 📊 Sample Data Format (CSV)
```
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/1/2010 8:26,2.55,17850,United Kingdom
536365,71053,WHITE METAL LANTERN,6,12/1/2010 8:26,3.39,17850,United Kingdom
```

## 📸 Visualizations
The model generates **three strip plots** for better insights:
- **Recency vs Cluster ID**
- **Frequency vs Cluster ID**
- **Monetary Value vs Cluster ID**

These are saved inside the **static/** directory and returned in the API response.

## 🔥 Troubleshooting & Debugging
- If you get a `KeyError`, check if your CSV has the correct column names.
- If `TypeError: unsupported operand type(s) for -: 'str' and 'str'` occurs, ensure numeric columns are correctly formatted.
- Always run `pip install -r requirements.txt` to ensure dependencies are installed.

## 📌 Future Enhancements
- Add a frontend UI for easy interaction.
- Extend to support other clustering models like DBSCAN or Hierarchical Clustering.
- Deploy the Flask app using Docker or AWS.

---
📌 **Author(s):** 
- Aniket Pendhari [Linkedin](https://www.linkedin.com/in/aniket-pendhari)
- Mitesh Patil [Linkedin](https://www.linkedin.com/in/mitpatil07)

📧 **Contact:**   
🚀 **Happy Coding!**

