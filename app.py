# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten 

import datetime as dt

st.set_page_config(page_title = "Unsupervised ML Platform", layout = "wide")
st.markdown("End to tend Customer, Text and Image analytics")

# Side bar

st.sidebar.header("Choose Analysis")
option = st.sidebar.selectbox(
    "Select Module",
    (
        "Customer Segmentation",
        "Market Basket Analysis",
        "Text Clustering",
        "Image Clustering"
    )
)

# Customer Segmentattion
if option == "Customer Segmentation":
    st.header("Customer Segmentation (RFM + Kmeans)")

    path_seg = r'C:\Users\Admin\Desktop\Omdena\Bhutan\MLOPS\Advanced Training\Supervised_Unsupervised ML\Capstone\Data\online_retail_II.xlsx'
    df = pd.read_excel(path_seg)
    df.dropna()
    df = df[(df['Quantity']>0) & (df['Price']>0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby(['Customer ID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'count',
        'Price': "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    k = st.slider("Number of clusters", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = model.fit_predict(rfm_scaled)

    st.subheader("CLuster Summary")
    st.dataframe(rfm.groupby('Cluster').mean())

    fig, ax = plt.subplots()
    sns.scatterplot(
        x = rfm['Recency'],
        y = rfm['Frequency'],
        hue = rfm['Cluster'],
        palette = 'viridis',
        ax = ax
    
    )
    ax.set_title("Customer Clusters")
    st.pyplot(fig)

# Market Basket Analysis
elif option == "Market Basket Analysis":
    st.header("Market Basket Analysis")
    path_seg = r'C:\Users\Admin\Desktop\Omdena\Bhutan\MLOPS\Advanced Training\Supervised_Unsupervised ML\Capstone\Data\online_retail_II.xlsx'
    df = pd.read_excel(path_seg)
    df.dropna()
    df = df[(df['Quantity']>0) & (df['Price']>0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    basket = (
        df.groupby(['Invoice', 'Description'])['Quantity']
        .sum()
        .unstack()
        .fillna(0)

    )
    basket = basket.applymap(lambda x: 1 if x>0 else 0)

    support = st.slider("Minimum Support", 0.01, 0.1, 0.02)

    frequent_itemsets = apriori(basket, min_support=support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1).sort_values("lift", ascending=False)

    st.subheader("Top Association Rules")
    st.dataframe(
        rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
        )
    

#------------------------------------------------------------
#     TEXT CLUSTERING
#------------------------------------------------------------
elif option == "Text Clustering":
    st.header("Text Clustering")

    

    def load_bbc_data(root_path):
        data = []
        # Loop through each folder (e.g. sport, tech...)
        for category in os.listdir(root_path):
            category_path = os.path.join(root_path, category)

            # Ensure we are looking at a directory not a hidden file
            if os.path.isdir(category_path):
                print(f"Processing category: {category}")
                
                # loop through every .txt file in that category
                for filename in os.listdir(category_path):
                    if filename.endswith(".txt"):
                        file_path = os.path.join(category_path, filename)
                        
                        # latin1 encode is safer
                        with open(file_path, 'r', encoding = 'latin1') as f:
                            content = f.read()
                            data.append({
                                "text": content,
                                "category": category,
                                "filename": filename
                            })

        return pd.DataFrame(data)
    

    root_path = r"C:\Users\Admin\Desktop\Omdena\Bhutan\MLOPS\Advanced Training\Supervised_Unsupervised ML\Capstone\Data\bbc"
    df = load_bbc_data(root_path)

    # texts = df['text']
    df_text = df.copy()
    texts = df_text['text']
            

    # path_bbc = r'C:\Users\Admin\Desktop\Omdena\Bhutan\MLOPS\Advanced Training\Supervised_Unsupervised ML\Capstone\Data\bbc_news_data.csv'

    # df_text = pd.read_csv(path_bbc)
    # texts = df_text['text']

    max_features = st.slider("Max TF-IDF Features", 1000, 10000, 5000)
    clusters = st.slider("Number of Clusters", 2, 10, 5)

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features
    )
    X_tfidf = vectorizer.fit_transform(texts)
    
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)


    # df_text = texts.copy()
    df_text['Cluster'] = labels
    st.dataframe(df_text[['text', 'Cluster']].head())

    st.subheader("Top Terms per cluster")
    terms = vectorizer.get_feature_names_out()

    for i in range(clusters):
        top_terms = kmeans.cluster_centers_[i].argsort()[-10:]
        st.markdown(f"**Cluster {i}**: " + ", ".join([terms[j] for j in top_terms]))
           


#------------------------------------------------------------
# IMAGE CLUSTERING
#------------------------------------------------------------
# elif option == "Image Clustering":
#     st.header("Image Clustering")

#     (X_train, _), (_, _) = fashion_mnist.load_data()
#     X_train = X_train/255.0

#     model = Sequential([
#         Flatten(input_shape=(28,28)),  
#     ])

#     embeddings = model.predict(X_train[:5000])

#     k = st.slider("Number of clusters", 2, 15, 10)
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(embeddings)

#     st.subheader("Sample Images per Cluster")
#     for cluster in range(min(k, 5)):
#         idx = np.where(labels == cluster)[0][:5]

#         fig, ax = plt.subplot(1, 5, figsize = (10, 3))
#         for i, img_idx in enumerate(idx):
#             ax[i].imshow(X_train[img_idx], cmap='gray')
#             ax[i].axis('off')
#         # st.pyplot(fig)



#------------------------------------------------------------
st.markdown('---')

    




    



    