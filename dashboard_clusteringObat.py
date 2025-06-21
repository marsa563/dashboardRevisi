import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Warna cluster
cluster_palette = {
    '1': '#1f77b4',
    '2': '#ff7f0e',
    '3': '#2ca02c'
}

# --- Load data ---
@st.cache_data
def load_data():
    url_obat = "https://docs.google.com/spreadsheets/d/188yRPLfbuGmT3A6WIJe-8pAUtBTljdutDD7HxG-iGSI/export?format=csv"
    data_obat = pd.read_csv(url_obat)
    url_hujan = "https://docs.google.com/spreadsheets/d/1iV-HQsqU36-r3pjR-zUu33a4wM2LTk49nGLN_V6o1oY/export?format=csv"
    data_hujan = pd.read_csv(url_hujan)
    return data_obat, data_hujan

data, data_hujan = load_data()

# --- Clustering (1x untuk semua halaman) ---
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

data_grouped['Qty_log'] = np.log1p(data_grouped['Qty'])
data_grouped['Item Amount_log'] = np.log1p(data_grouped['Item Amount'])
data_grouped['CV (%)_log'] = np.log1p(data_grouped['CV (%)'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_grouped[['Qty_log', 'Item Amount_log', 'CV (%)_log', 'Jumlah Bulan Muncul']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
data_grouped['Cluster'] = labels + 1
data_grouped['Cluster_str'] = data_grouped['Cluster'].astype(str)

# --- BUAT data_exploded ---
data_exploded = data_grouped.copy()
data_exploded['Use'] = data_exploded['Use'].str.split(',')
data_exploded = data_exploded.explode('Use')
data_exploded['Use'] = data_exploded['Use'].str.strip()

# Manipulasi khusus PERGOVERIS
data_exploded = data_exploded[
    ~((data_exploded['Item'] == 'PERGOVERIS 150 IU/75 IU') & (data_exploded['Cluster'] == 2))
]
data_exploded.loc[
    (data_exploded['Item'] == 'PERGOVERIS 150 IU/75 IU') & (data_exploded['Cluster'] == 1),
    'Cluster'
] = 2
data_exploded['Cluster_str'] = data_exploded['Cluster'].astype(str)

# --- Sidebar nav ---
page = st.sidebar.radio("Pilih Halaman", ["Clustering Obat", "Analisis Curah Hujan"])

# =================== CLUSTERING OBAT ===================
if page == "Clustering Obat":
    st.title("Dashboard Clustering Obat")

    st.subheader("Preview Data Mentah")
    st.dataframe(data.head())

    st.subheader("Elbow Method")
    sse = []
    for k in range(1, 9):
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_tmp.fit(X_scaled)
        sse.append(kmeans_tmp.inertia_)
    fig_elbow, ax = plt.subplots()
    ax.plot(range(1,9), sse, marker='o', linestyle='--')
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("SSE")
    ax.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    st.subheader("Hasil Clustering")
    cluster_counts = data_grouped['Cluster'].value_counts().sort_index()
    st.write(cluster_counts)

    fig_pie, ax = plt.subplots()
    ax.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index],
           autopct='%1.1f%%',
           colors=[cluster_palette[str(i)] for i in cluster_counts.index])
    ax.set_title("Distribusi Cluster")
    st.pyplot(fig_pie)

    mean_data = data_grouped.groupby('Cluster_str').agg({
        'Qty': 'mean',
        'Item Amount': 'mean',
        'CV (%)': 'mean',
        'Jumlah Bulan Muncul': 'mean'
    }).reset_index()
    st.write("Rata-rata Fitur per Cluster")
    st.dataframe(mean_data)

    st.subheader("Bar Chart Perbandingan Fitur per Cluster")
    for col in ['Qty', 'Item Amount', 'CV (%)', 'Jumlah Bulan Muncul']:
        fig_bar, ax = plt.subplots()
        sns.barplot(data=mean_data, x='Cluster_str', y=col, palette=cluster_palette, ax=ax)
        ax.set_title(f"Rata-rata {col} per Cluster")
        st.pyplot(fig_bar)

    st.subheader("Top 10 Fungsi Obat per Cluster")
    for cl in sorted(data_exploded['Cluster'].unique()):
        use_qty = data_exploded[data_exploded['Cluster'] == cl].groupby('Use')['Qty'].sum().reset_index().sort_values('Qty', ascending=False).head(10)
        fig_top, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=use_qty, x='Use', y='Qty', palette=[cluster_palette[str(cl)]]*len(use_qty), ax=ax)
        ax.set_title(f"Top 10 Fungsi Obat Cluster {cl}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig_top)

# =================== ANALISIS CURAH HUJAN ===================
if page == "Analisis Curah Hujan":
    st.title("Analisis Curah Hujan")

    data_hujan['TANGGAL'] = pd.to_datetime(data_hujan['TANGGAL'])
    data_hujan['Month'] = data_hujan['TANGGAL'].dt.month
    monthly_sum = data_hujan.groupby('Month')['RR'].sum().reset_index()
    nama_bulan = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
    monthly_sum['MonthName'] = monthly_sum['Month'].apply(lambda x: nama_bulan[x - 1])

    fig_line, ax = plt.subplots(figsize=(10,5))
    ax.plot(monthly_sum['MonthName'], monthly_sum['RR'], marker='o', linewidth=2, color='darkgreen')
    ax.set_title("Curah Hujan di Jakarta Pusat Tahun 2024")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Curah Hujan (RR)")
    ax.grid(True)
    ax.axhline(100, color='red', linestyle='--', label='Batas Rendah')
    ax.axhline(300, color='orange', linestyle='--', label='Batas Menengah')
    ax.axhline(500, color='green', linestyle='--', label='Batas Tinggi')
    ax.legend()
    st.pyplot(fig_line)

    item_month = data_selected[['Item', 'Month']].drop_duplicates()
    data_final = data_exploded.merge(item_month, on='Item', how='left')
    data_final = data_final.merge(monthly_sum[['Month', 'RR']], on='Month', how='left')

    cluster_pick = st.selectbox("Pilih Cluster", sorted(data_final['Cluster'].unique()))
    kategori = st.selectbox("Pilih Kategori Curah Hujan", ['Rendah', 'Menengah', 'Tinggi', 'Sangat Tinggi'])

    if kategori == 'Rendah':
        df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR'] < 100)]
    elif kategori == 'Menengah':
        df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR'] >= 100) & (data_final['RR'] <= 300)]
    elif kategori == 'Tinggi':
        df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR'] > 300) & (data_final['RR'] <= 500)]
    else:
        df_filtered = data_final[(data_final['Cluster'] == cluster_pick) & (data_final['RR'] > 500)]

    if not df_filtered.empty:
        use_qty = df_filtered.groupby('Use')['Qty'].sum().reset_index().sort_values('Qty', ascending=False).head(10)
        fig_use, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=use_qty, x='Use', y='Qty', palette=[cluster_palette[str(cluster_pick)]]*len(use_qty), ax=ax)
        ax.set_title(f"Top 10 Fungsi Obat Cluster {cluster_pick} - {kategori}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig_use)
        st.dataframe(use_qty)
    else:
        st.info("Tidak ada data untuk kombinasi ini.")
