import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Palet warna cluster
cluster_palette = {
    1: '#1f77b4',
    2: '#ff7f0e',
    3: '#2ca02c'
}

# --- LOAD DATA ---
@st.cache_data
def load_data():
    url_obat = "https://docs.google.com/spreadsheets/d/188yRPLfbuGmT3A6WIJe-8pAUtBTljdutDD7HxG-iGSI/export?format=csv"
    url_hujan = "https://docs.google.com/spreadsheets/d/1iV-HQsqU36-r3pjR-zUu33a4wM2LTk49nGLN_V6o1oY/export?format=csv"
    data_obat = pd.read_csv(url_obat)
    data_hujan = pd.read_csv(url_hujan)
    data_hujan['TANGGAL'] = pd.to_datetime(data_hujan['TANGGAL'])
    return data_obat, data_hujan

data, data_hujan = load_data()

# --- PREPROCESSING + CLUSTERING ---
data = data[~data['Item'].str.contains('Racikan', case=False, na=False)]
data = data[data['Qty'] > 0]
data['Month'] = pd.to_datetime(data['Invoice Date']).dt.month

qty_bulanan = data.groupby(['Item', 'Month'])['Qty'].sum().reset_index()
stabilitas = qty_bulanan.groupby('Item')['Qty'].agg(['mean', 'std']).reset_index()
stabilitas['CV (%)'] = (stabilitas['std'] / stabilitas['mean']) * 100
stabilitas = stabilitas[['Item', 'CV (%)']]
bulan_aktif = qty_bulanan.groupby('Item')['Month'].nunique().reset_index()
bulan_aktif.rename(columns={'Month': 'Jumlah Bulan Muncul'}, inplace=True)
stabilitas = stabilitas.merge(bulan_aktif, on='Item', how='left')
data = data.merge(stabilitas, on='Item', how='left')
data['CV (%)'] = data['CV (%)'].fillna(80)

data_grouped = data.groupby(
    ['Item', 'Supplier', 'Use', 'CV (%)', 'Jumlah Bulan Muncul'],
    as_index=False
)[['Qty', 'Item Amount']].sum()

data_grouped['Qty_log'] = np.log1p(data_grouped['Qty'])
data_grouped['Item Amount_log'] = np.log1p(data_grouped['Item Amount'])
data_grouped['CV (%)_log'] = np.log1p(data_grouped['CV (%)'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_grouped[['Qty_log', 'Item Amount_log', 'CV (%)_log', 'Jumlah Bulan Muncul']])

# --- KMeans ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
data_grouped['Cluster'] = labels + 1

# Silhouette & DBI
silhouette_avg = silhouette_score(X_scaled, labels)
dbi = davies_bouldin_score(X_scaled, labels)

# --- DATA EXPLODED ---
data_final = pd.merge(
    data,
    data_grouped[['Item', 'Cluster']],
    on='Item',
    how='left'
)

data_grouped_clustered = data_final.groupby(
    ['Item', 'Supplier', 'Use', 'Month', 'CV (%)', 'Cluster'],
    as_index=False
)[['Qty', 'Item Amount']].sum()

data_exploded = data_grouped_clustered.copy()
data_exploded['Use'] = data_exploded['Use'].str.split(',')
data_exploded = data_exploded.explode('Use')
data_exploded['Use'] = data_exploded['Use'].str.strip()

# PERGOVERIS rules
data_exploded = data_exploded[
    ~((data_exploded['Item'] == 'PERGOVERIS 150 IU/75 IU') & (data_exploded['Cluster'] == 2))
]
data_exploded.loc[
    (data_exploded['Item'] == 'PERGOVERIS 150 IU/75 IU') & (data_exploded['Cluster'] == 1),
    'Cluster'
] = 2

# Merge curah hujan
monthly_sum = data_hujan.groupby(data_hujan['TANGGAL'].dt.month)['RR'].sum().reset_index()
monthly_sum.columns = ['Month', 'RR_BULAN']
data_exploded = data_exploded.merge(monthly_sum, on='Month', how='left')

# --- SIDEBAR ---
page = st.sidebar.radio("Pilih Halaman", ["Clustering Obat", "Analisis Curah Hujan"])

# ==================== CLUSTERING OBAT ====================
if page == "Clustering Obat":
    st.title("Clustering Obat")

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
    ax.pie(cluster_counts, 
           labels=[f"Cluster {i}" for i in cluster_counts.index],
           autopct='%1.1f%%',
           colors=[cluster_palette[int(i)] for i in cluster_counts.index])
    ax.set_title("Distribusi Cluster")
    st.pyplot(fig_pie)

    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
    st.write(f"Davies-Bouldin Index: {dbi:.4f}")

    mean_data = data_grouped.groupby('Cluster').agg({
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
        sns.barplot(data=mean_data, x='Cluster', y=col,
                    palette=[cluster_palette[int(i)] for i in mean_data['Cluster']], ax=ax)
        ax.set_title(f"Rata-rata {col} per Cluster")
        st.pyplot(fig_bar)

    st.subheader("Top 10 Fungsi Obat per Cluster")
    for cl in sorted(data_exploded['Cluster'].unique()):
        use_qty = data_exploded[data_exploded['Cluster'] == cl].groupby('Use')['Qty'].sum().reset_index().sort_values('Qty', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(data=use_qty, x='Use', y='Qty', color=cluster_palette[cl], ax=ax)
        ax.set_title(f"Top 10 Fungsi Obat Cluster {cl}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

# ==================== ANALISIS CURAH HUJAN ====================
if page == "Analisis Curah Hujan":
    st.title("Analisis Curah Hujan")

    cluster_pick = st.selectbox("Pilih Cluster", sorted(data_exploded['Cluster'].unique()))
    kategori = st.selectbox("Pilih Kategori Curah Hujan", ['RENDAH', 'MENENGAH', 'TINGGI', 'SANGAT TINGGI'])

    if kategori == 'RENDAH':
        df_filtered = data_exploded[(data_exploded['Cluster'] == cluster_pick) & (data_exploded['RR_BULAN'] <= 100)]
    elif kategori == 'MENENGAH':
        df_filtered = data_exploded[(data_exploded['Cluster'] == cluster_pick) & (data_exploded['RR_BULAN'] > 100) & (data_exploded['RR_BULAN'] <= 300)]
    elif kategori == 'TINGGI':
        df_filtered = data_exploded[(data_exploded['Cluster'] == cluster_pick) & (data_exploded['RR_BULAN'] > 300) & (data_exploded['RR_BULAN'] <= 500)]
    else:
        df_filtered = data_exploded[(data_exploded['Cluster'] == cluster_pick) & (data_exploded['RR_BULAN'] > 500)]

    if not df_filtered.empty:
        use_qty = df_filtered.groupby('Use')['Qty'].sum().reset_index().sort_values('Qty', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(data=use_qty, x='Use', y='Qty', color=cluster_palette[cluster_pick], ax=ax)
        ax.set_title(f"Top 10 Fungsi Obat Cluster {cluster_pick} - {kategori}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

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
