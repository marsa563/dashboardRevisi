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
stabilitas['CV'] = (stabilitas['std'] / stabilitas['mean']) * 100
stabilitas = stabilitas[['Item', 'CV']]
bulan_aktif = qty_bulanan.groupby('Item')['Month'].nunique().reset_index()
bulan_aktif.rename(columns={'Month': 'Jumlah Bulan Muncul'}, inplace=True)
stabilitas = stabilitas.merge(bulan_aktif, on='Item', how='left')
data = data.merge(stabilitas, on='Item', how='left')
data['CV'] = data['CV'].fillna(80)

data_grouped = data.groupby(
    ['Item', 'Supplier', 'Use', 'CV', 'Jumlah Bulan Muncul'],
    as_index=False
)[['Qty', 'Item Amount']].sum()

data_grouped['Qty_log'] = np.log1p(data_grouped['Qty'])
data_grouped['Item Amount_log'] = np.log1p(data_grouped['Item Amount'])
data_grouped['CV_log'] = np.log1p(data_grouped['CV'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_grouped[['Qty_log', 'Item Amount_log', 'CV_log', 'Jumlah Bulan Muncul']])

# --- KMeans ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
data_grouped['Cluster'] = labels + 1

# --- DATA EXPLODED ---
data_final = pd.merge(
    data,
    data_grouped[['Item', 'Cluster']],
    on='Item',
    how='left'
)

data_grouped_clustered = data_final.groupby(
    ['Item', 'Supplier', 'Use', 'Month', 'CV', 'Cluster'],
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
page = st.sidebar.radio("Pilih Halaman", ["Hasil Klasterisasi", "Analisis Curah Hujan"])
st.sidebar.markdown("---")
st.sidebar.markdown("Marsa Nabila | 2110512048")

# ==================== CLUSTERING OBAT ====================
if page == "Hasil Klasterisasi":
    st.title("Dashboard Analisis Segmentasi Penjualan Obat Di RSU YPK Mandiri Menggunakan Metode K-Means")

    st.subheader("Preview Data")
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

    st.markdown("""
        Gambar grafik dari hasil metode elbow, dapat dilihat bahwa titik siku terjadi pada k = 3.
        """)
    
    st.subheader("Hasil Clustering")
    # Hitung jumlah data per cluster
    cluster_counts = data_grouped['Cluster'].value_counts().sort_index()
    # Ubah ke DataFrame
    cluster_df = cluster_counts.reset_index()
    cluster_df.columns = ['Cluster', 'Jumlah Item']
    
    # Tambahkan kolom karakteristik
    karakteristik_dict = {
        1: 'Seasonal or Irregular Moving Items',
        2: 'Fast Moving Items',
        3: 'Slow Moving Items'
    }
    cluster_df['Karakteristik'] = cluster_df['Cluster'].map(karakteristik_dict)
    st.write(cluster_df)

    fig_pie, ax = plt.subplots()
    ax.pie(cluster_counts, 
           labels=[f"Cluster {i}" for i in cluster_counts.index],
           autopct='%1.1f%%',
           colors=[cluster_palette[int(i)] for i in cluster_counts.index])
    ax.set_title("Distribusi Cluster")
    st.pyplot(fig_pie)

    st.subheader("Data per Cluster")
    # Menampilkan data dari setiap cluster
    for i in range(1, 4):
        cluster_df = data_grouped[data_grouped['Cluster'] == i].copy()
        cluster_df = cluster_df[['Item', 'Qty', 'Item Amount', 'Supplier', 'Use', 'CV', 'Jumlah Bulan Muncul', 'Cluster']]
        st.markdown(f"### Data untuk Cluster {i}")
        st.dataframe(cluster_df)
    
    mean_data = data_grouped.groupby('Cluster').agg({
        'Qty': 'mean',
        'Item Amount': 'mean',
        'CV': 'mean',
        'Jumlah Bulan Muncul': 'mean'
    }).reset_index()

    st.subheader("Bar Chart Perbandingan Fitur per Cluster")
    for col in ['Qty', 'Item Amount', 'CV', 'Jumlah Bulan Muncul']:
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

# Rekomendasi
    st.subheader("Rekomendasi")
    st.markdown("""
       Cluster 1 (Seasonal or Irregular Moving Items) : kelompok obat dengan pola penggunaan yang fluktuatif dan tidak konsisten. Obat-obatan dalam cluster ini cenderung dibutuhkan dalam waktu atau kondisi tertentu, seperti saat musim penyakit infeksi meningkat atau ketika terjadi lonjakan kasus. Pengadaan untuk cluster ini perlu disesuaikan dengan tren musiman dan pemantauan kondisi aktual di lapangan.
       
       Cluster 2 (Fast Moving Items) : kelompok obat dengan permintaan tinggi, frekuensi penggunaan rutin, dan permintaan yang konsisten sepanjang tahun. Obat dalam kelompok ini umumnya digunakan untuk pengobatan penyakit kronis atau pemeliharaan kesehatan jangka panjang, seperti vitamin, obat hipertensi, dan diabetes. Cluster ini perlu menjadi prioritas utama dalam perencanaan pengadaan dan pengelolaan stok agar selalu tersedia dan menghindari kekosongan. Selain itu, rumah sakit dapat memprioritaskan alokasi anggaran untuk obat-obatan dalam cluster ini karena permintaannya tinggi dan rutin, lakukan pembelian dalam jumlah besar dan terjadwal secara berkala untuk memastikan ketersediaan stok yang stabil sepanjang tahun untuk menghindari kekosongan stok (stockout). Disarankan menggunakan kontrak jangka panjang dengan supplier untuk menjamin kontinuitas pasokan dan memperoleh harga yang kompetitif.
       
       Cluster 3 (Slow Moving Items) : kelompok obat dengan permintaan jarang digunakan namun memiliki pola permintaan yang stabil. Biasanya digunakan untuk kondisi medis yang lebih spesifik. Item dalam cluster ini hanya dibeli sesuai permintaan, lakukan evaluasi berkala untuk menghindari penyimpanan berlebihan dan tetap sediakan stok minimum untuk kebutuhan khusus.
        """)

# Saran
    st.subheader("Saran")
    st.markdown("""
        Hasil analisis dan visualisasi ini berdasarkan dataset transaksi bulan Januari-Desember 2024.
        """)

# ==================== ANALISIS CURAH HUJAN ====================
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

    st.markdown("""
        Sumber: https://dataonline.bmkg.go.id.
        """)
    
    st.title("Analisis Curah Hujan per Cluster")
    
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
        st.write("Rekomendasi Supplier")
        st.dataframe(top3_df)
    else:
        st.info("Tidak ada data untuk kombinasi ini.")
