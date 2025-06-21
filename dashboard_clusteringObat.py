import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.title("Analisis Segmentasi Penjualan Obat Menggunakan Metode K-Means")

# --- Load data obat ---
url_obat = "https://docs.google.com/spreadsheets/d/188yRPLfbuGmT3A6WIJe-8pAUtBTljdutDD7HxG-iGSI/export?format=csv"
data = pd.read_csv(url_obat)

# --- Preprocessing ---
selected_columns = ['Invoice Date', 'Item', 'Qty', 'Item Amount', 'Supplier', 'Use']
data_selected = data[selected_columns]
data_selected = data_selected[
    ~data_selected['Item'].str.contains('Racikan', case=False, na=False)
]
data_selected = data_selected[data_selected['Qty'] > 0]
data_selected['Month'] = pd.to_datetime(data_selected['Invoice Date']).dt.month

qty_bulanan = data_selected.groupby(['Item', 'Month'])['Qty'].sum().reset_index()
stabilitas = qty_bulanan.groupby('Item')['Qty'].agg(['mean', 'std']).reset_index()
stabilitas['CV (%)'] = (stabilitas['std'] / stabilitas['mean']) * 100
stabilitas = stabilitas[['Item', 'CV (%)']]
bulan_aktif = qty_bulanan.groupby('Item')['Month'].nunique().reset_index()
bulan_aktif.rename(columns={'Month': 'Jumlah Bulan Muncul'}, inplace=True)
stabilitas = stabilitas.merge(bulan_aktif, on='Item', how='left')
data_selected = data_selected.merge(stabilitas, on='Item', how='left')
data_selected['CV (%)'] = data_selected['CV (%)'].fillna(80)

data_grouped = data_selected.drop(columns=['Month']).groupby(
    ['Item', 'Supplier', 'Use', 'CV (%)', 'Jumlah Bulan Muncul'],
    as_index=False
).agg({'Qty': 'sum', 'Item Amount': 'sum'})

# --- Transformasi Log + Normalisasi ---
data_grouped['Qty_log'] = np.log1p(data_grouped['Qty'])
data_grouped['Item Amount_log'] = np.log1p(data_grouped['Item Amount'])
data_grouped['CV (%)_log'] = np.log1p(data_grouped['CV (%)'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_grouped[['Qty_log', 'Item Amount_log', 'CV (%)_log', 'Jumlah Bulan Muncul']])

# --- KMeans clustering ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
data_grouped['Cluster'] = labels + 1

# --- Evaluasi ---
silhouette_avg = silhouette_score(X_scaled, labels)
dbi = davies_bouldin_score(X_scaled, labels)

# --- Tampilkan hasil clustering ---
st.subheader("Hasil Clustering")
cluster_counts = data_grouped['Cluster'].value_counts().sort_index()
st.write("Distribusi Cluster:")
st.write(cluster_counts)

fig1, ax1 = plt.subplots()
ax1.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

st.write("Rata-rata Fitur per Cluster")
mean_data = data_grouped.groupby('Cluster').agg({
    'Qty': 'mean',
    'Item Amount': 'mean',
    'CV (%)': 'mean',
    'Jumlah Bulan Muncul': 'mean'
}).reset_index()
st.dataframe(mean_data)

st.write(f"Silhouette Score: {silhouette_avg:.4f}")
st.write(f"Davies-Bouldin Index: {dbi:.4f}")

# --- Load curah hujan ---
url_hujan = "https://docs.google.com/spreadsheets/d/1iV-HQsqU36-r3pjR-zUu33a4wM2LTk49nGLN_V6o1oY/export?format=csv"
data_hujan = pd.read_csv(url_hujan)
data_hujan['TANGGAL'] = pd.to_datetime(data_hujan['TANGGAL'])
data_hujan['Month'] = data_hujan['TANGGAL'].dt.month
monthly_sum = data_hujan.groupby('Month')['RR'].sum().reset_index()
monthly_sum.columns = ['Month', 'RR_BULAN']

# Merge data cluster + bulan + curah hujan
item_month = data_selected[['Item', 'Month']].drop_duplicates()
data_final = data_grouped.merge(item_month, on='Item', how='left')
data_final = data_final.merge(monthly_sum, on='Month', how='left')

# --- Analisis curah hujan ---
st.subheader("Analisis Curah Hujan per Cluster")
cluster_pick = st.selectbox("Pilih Cluster", sorted(data_final['Cluster'].unique()))
kategori = st.selectbox("Pilih Kategori Curah Hujan", ['Rendah', 'Menengah', 'Tinggi', 'Sangat Tinggi'])

if kategori == 'Rendah':
    df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR_BULAN'] < 100)]
elif kategori == 'Menengah':
    df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR_BULAN'] >= 100) & (data_final['RR_BULAN'] <= 300)]
elif kategori == 'Tinggi':
    df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR_BULAN'] > 300) & (data_final['RR_BULAN'] <= 500)]
else:
    df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR_BULAN'] > 500)]

if not df_filtered.empty:
    use_qty = df_filtered.groupby('Use')['Qty'].sum().reset_index().sort_values('Qty', ascending=False).head(10)
    st.write("Top 10 Fungsi Obat (Use)")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.barplot(data=use_qty, x='Use', y='Qty', palette='viridis', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
    st.dataframe(use_qty)

    top3_list = []
    for use in use_qty['Use']:
        top3 = df_filtered[df_filtered['Use'] == use].groupby(['Item', 'Supplier'])['Qty'].sum().reset_index().sort_values('Qty', ascending=False).head(3)
        top3['Use'] = use
        top3_list.append(top3)
    top3_df = pd.concat(top3_list)
    st.write("Top 3 Item + Supplier per Fungsi Obat")
    st.dataframe(top3_df)
else:
    st.info("Tidak ada data untuk kombinasi ini.")
