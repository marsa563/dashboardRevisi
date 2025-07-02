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
page = st.sidebar.radio("Pilih Halaman", ["Hasil Klasterisasi", "Optimalisasi"])
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

    # Ukuran figure diperlebar untuk beri ruang legend
    fig_pie, ax = plt.subplots(figsize=(8, 6))
    
    # Pie chart
    wedges, texts, autotexts = ax.pie(
        cluster_counts,
        labels=[f"Cluster {i}" for i in cluster_counts.index],
        autopct='%1.1f%%',
        colors=[cluster_palette[int(i)] for i in cluster_counts.index]
    )
    ax.set_title("Distribusi Cluster")
    
    # Tambahkan legend kustom
    cluster_labels = {
        1: "Seasonal or irregular moving items",
        2: "Fast moving items",
        3: "Slow moving items"
    }
    
    # Buat legend di pojok kanan atas, tidak menimpa chart
    legend_labels = [f"Cluster {i}: {cluster_labels[i]}" for i in cluster_counts.index]
    ax.legend(
        wedges,
        legend_labels,
        title="Cluster",
        loc='upper right',
        bbox_to_anchor=(1.4, 1)  # posisi lebih ke kanan
    )
    
    # Tampilkan di Streamlit
    st.pyplot(fig_pie)



    st.markdown("""
        Gambar di atas menunjukkan hasil segmentasi item penjualan menggunakan metode K-Means Clustering yang dibagi menjadi tiga kelompok berdasarkan karakteristik pergerakan item, yaitu Fast Moving Items, Seasonal or Irregular Moving Items, dan Slow Moving Items.

        Cluster 2 merupakan kelompok dengan jumlah item terbanyak, yaitu sebanyak 740 item atau sekitar 57,5% dari total item. Kelompok ini dikategorikan sebagai Fast Moving Items, yaitu item yang memiliki tingkat penjualan tinggi dan penjualan cukup stabil.

        Cluster 1 terdiri dari 498 item atau sekitar 38,7%. Kelompok ini diklasifikasikan sebagai Seasonal or Irregular Moving Items, yaitu item dengan pola penjualan yang fluktuatif atau musiman.
        
        Cluster 3 hanya mencakup 50 item atau sekitar 3,9%. Kelompok ini merupakan Slow Moving Items, yaitu item yang jarang terjual. Item dalam kelompok ini perlu dievaluasi secara berkala untuk menghindari penumpukan stok atau pemborosan anggaran.
        """)

    st.subheader("Data per Cluster")
    st.markdown("""
        Berikut merupakan daftar item per cluster:
        """)
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
    
        # Tambahkan keterangan setelah grafik
        if col == 'Qty':
            st.markdown("Grafik diatas merupakan hasil visualisasi rata-rata jumlah penjualan (Qty) pada masing-masing cluster yang terbentuk dari hasil K-Means. Cluster 1 menunjukkan rata-rata paling rendah dengan nilai 34,33, yang menunjukkan bahwa kelompok ini berisi item dengan tingkat penjualan yang sangat rendah dibandingkan cluster lain. Sebaliknya, Cluster 2 menunjukkan rata-rata tertinggi dengan nilai 1374,61, yang menunjukkan bahwa kelompok ini berisi item dengan tingkat penjualan yang sangat tinggi dibandingkan cluster lain. Sementara itu, Cluster 3 menunjukkan rata-rata menengah dengan nilai 66,72, yang menunjukkan bahwa kelompok ini berisi item dengan tingkat penjualan menengah.")
        elif col == 'Item Amount':
            st.markdown("Grafik diatas merupakan hasil visualisasi rata-rata nilai transaksi (Item Amount) pada masing-masing cluster yang terbentuk dari hasil K-Means. Cluster 1 menunjukkan rata-rata paling rendah dengan nilai 1.701.851, yang menunjukkan bahwa kelompok ini berisi item dengan nilai transaksi yang sangat rendah dibandingkan cluster lain. Sebaliknya, Cluster 2 menunjukkan rata-rata tertinggi dengan nilai 37.309.363, yang menunjukkan bahwa kelompok ini berisi item dengan nilai transaksi yang sangat tinggi dibandingkan cluster lain. Sementara itu, Cluster 3 menunjukkan rata-rata menengah dengan nilai 2.412.678, yang menunjukkan bahwa kelompok ini berisi item dengan nilai transaksi menengah.")
        elif col == 'CV':
            st.markdown("Grafik diatas merupakan hasil visualisasi rata-rata koefisien variasi (CV) pada masing-masing cluster yang terbentuk dari hasil K-Means. Nilai CV mencerminkan tingkat kestabilan penjualan obat, dimana semakin rendah nilai koefisien variasi, semakin stabil pola pemakaiannya dari waktu ke waktu (Chaniago, 2022).  Cluster 1 menunjukkan rata-rata tertinggi dengan nilai 75,83, yang menunjukkan bahwa kelompok ini berisi item dengan penjualan fluktuatif. Cluster 2 menunjukkan rata-rata menengah dengan nilai 51,30, yang menunjukkan bahwa kelompok ini berisi item dengan penjualan cukup stabil. Sementara itu, Cluster 3 menunjukkan rata-rata paling rendah dengan nilai 0,16, yang menunjukkan bahwa kelompok ini berisi item dengan penjualan sangat stabil.")
        elif col == 'Jumlah Bulan Muncul':
            st.markdown("Gambar 4.16 merupakan hasil visualisasi rata-rata jumlah bulan muncul pada masing-masing cluster yang terbentuk dari hasil K-Means. Cluster 1 menunjukkan rata-rata paling menengah dengan nilai 3,18, yang menunjukkan bahwa kelompok ini berisi item dengan penjualan kurang konsisten dan cenderung bersifat musiman. Cluster 2 menunjukkan rata-rata tertinggi dengan nilai 10,76, yang menunjukkan bahwa kelompok ini berisi item dengan penjualan konsisten sepanjang tahun dibandingkan cluster lain. Sementara itu, Cluster 3 menunjukkan rata-rata paling rendah dengan nilai 2,74, yang menunjukkan bahwa kelompok ini berisi item yang jarang terjual atau kebutuhan khusus.")


    st.subheader("Top 10 Fungsi Obat per Cluster")
    
    for cl in sorted(data_exploded['Cluster'].unique()):
        use_qty = data_exploded[data_exploded['Cluster'] == cl] \
            .groupby('Use')['Qty'].sum() \
            .reset_index() \
            .sort_values('Qty', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=use_qty, x='Use', y='Qty', color=cluster_palette[cl], ax=ax)
        ax.set_title(f"Top 10 Fungsi Obat Cluster {cl}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        
        # Tambahkan narasi atau penjelasan untuk tiap cluster
        if cl == 1:
            st.markdown(
                "Cluster 1 : Seasonal or Irregular Moving Items"
                "Berdasarkan asil visualisasi top 10 Fungsi Obat (Use) pada Cluster 1, dapat dilihat bahwa hipertensi dan diabetes mellitus tipe 2 tetap menjadi dua fungsi utama dalam Cluster 1, namun dengan jumlah permintaan yang jauh lebih rendah dibandingkan Cluster 2, karena kemungkinan adanya perbedaan dalam merek, dosis, atau bentuk sediaan yang digunakan. Selanjutnya, terdapat fungsi obat untuk penyakit musiman dan infeksi umum seperti batuk, infeksi bakteri, serta antibiotik yang muncul. Ini menunjukkan bahwa Cluster 1 banyak mencakup obat-obat yang digunakan ketika terjadi lonjakan kasus, misalnya saat musim hujan (flu, batuk, ISPA), atau saat terjadi penyebaran infeksi. Fungsi-fungsi lain yang muncul seperti kolesterol, stimulasi ovarium, infeksi saluran kemih, dan zat besi menunjukkan adanya keberagaman kondisi medis yang tidak selalu rutin terjadi. Hal ini sangat selaras dengan karakteristik Cluster 1, dengan jumlah pembelian yang lebih rendah, serta frekuensi kemunculan yang sedikit, dan permintaan yang fluktuatif, sehingga perlu dikelola secara dinamis dan menyesuaikan dengan tren penyakit musiman atau permintaan temporer."
            )
        elif cl == 2:
            st.markdown(
                "Cluster 2 : Fast Moving Items"
                "Berdasarkan hasil visualisasi top 10 Fungsi Obat (Use) pada Cluster 2, dapat dilihat bahwa Vitamin D menempati posisi teratas dalam Cluster 2 yang menandakan bahwa suplemen ini memiliki tingkat konsumsi tinggi dan berkelanjutan, kemungkinan besar digunakan dalam program pemeliharaan kesehatan umum, terutama untuk pasien dengan risiko defisiensi vitamin, seperti lansia, ibu hamil, atau pasien dengan penyakit kronis. Fungsi lain yang menempati urutan atas dalam grafik ini adalah obat hipertensi, terapi hormon, infeksi bakteri, dan diabetes melitus tipe 2. Fungsi-fungsi tersebut berkaitan erat dengan penyakit kronis atau kebutuhan terapi jangka panjang, yang menuntut penggunaan obat secara konsisten dan berulang. Hal ini sangat selaras dengan karakteristik Cluster 2, dengan jumlah pembelian yang sangat tinggi, serta frekuensi kemunculan yang merata hampir setiap bulan. Dengan demikian, item dalam Cluster 2 bisa diidentifikasi sebagai obat prioritas tinggi yang perlu dijaga ketersediaannya secara stabil dalam stok rumah sakit."
            )
        elif cl == 3:
            st.markdown(
                "Cluster 3 : Slow Moving Items"
                "Berdasarkan hasil visualisasi top 10 Fungsi Obat (Use) pada Cluster 3, dapat dilihat bahwa Vitamin kehamilan mendominasi secara signifikan di Cluster 3, diikuti oleh item dengan fungsi yang lebih spesifik termasuk gangguan metabolisme protein, pembesaran prostat, hiperplasia, dan GERD/tukak lambung, dimana fungsi-fungsi obat ini merupakan kondisi medis khusus dan biasanya ditangani dengan terapi jangka panjang namun intensitas rendah. Meskipun jumlah kuantitas setiap fungsi di cluster ini relatif kecil, hal ini sejalan dengan karakteristik Cluster 3, yaitu kelompok item yang muncul hanya dalam beberapa bulan, namun memiliki pola permintaan yang sangat stabil. Fungsi-fungsi dalam cluster ini cenderung terkait penggunaan khusus. Strategi pengadaan yang digunakan bisa berupa stok minimum yang tetap tersedia, dengan penyesuaian berdasarkan siklus perawatan pasien tertentu."
            )


# Kesimpulan
    st.subheader("Kesimpulan")
    st.markdown("""
       Cluster 1 (Seasonal or Irregular Moving Items) : kelompok obat dengan pola penggunaan yang fluktuatif dan tidak konsisten. Obat-obatan dalam cluster ini cenderung dibutuhkan dalam waktu atau kondisi tertentu, seperti saat musim penyakit infeksi meningkat atau ketika terjadi lonjakan kasus.
       
       Cluster 2 (Fast Moving Items) : kelompok obat dengan permintaan tinggi, frekuensi penggunaan rutin, dan permintaan yang konsisten sepanjang tahun. Obat dalam kelompok ini umumnya digunakan untuk pengobatan penyakit kronis atau pemeliharaan kesehatan jangka panjang, seperti vitamin, obat hipertensi, dan diabetes.
       
       Cluster 3 (Slow Moving Items) : kelompok obat dengan permintaan jarang digunakan namun memiliki pola permintaan yang stabil. Biasanya digunakan untuk kondisi medis yang lebih spesifik.
        """)
    
# Rekomendasi
    st.subheader("Rekomendasi")
    st.markdown("""
       Cluster 1 (Seasonal or Irregular Moving Items) : Pengadaan untuk cluster ini perlu disesuaikan dengan tren musiman dan pemantauan kondisi aktual di lapangan.
       
       Cluster 2 (Fast Moving Items) : Cluster ini perlu menjadi prioritas utama dalam perencanaan pengadaan dan pengelolaan stok agar selalu tersedia dan menghindari kekosongan. Selain itu, rumah sakit dapat memprioritaskan alokasi anggaran untuk obat-obatan dalam cluster ini karena permintaannya tinggi dan rutin, lakukan pembelian dalam jumlah besar dan terjadwal secara berkala untuk memastikan ketersediaan stok yang stabil sepanjang tahun untuk menghindari kekosongan stok (stockout). Disarankan menggunakan kontrak jangka panjang dengan supplier untuk menjamin kontinuitas pasokan dan memperoleh harga yang kompetitif.
       
       Cluster 3 (Slow Moving Items) : Item dalam cluster ini hanya dibeli sesuai permintaan, lakukan evaluasi berkala untuk menghindari penyimpanan berlebihan dan tetap sediakan stok minimum untuk kebutuhan khusus.
        """)

# ==================== OPTIMALISASI ====================
if page == "Optimalisasi":
    st.title("Curah Hujan")

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
    st.write("Gambar diatas merupakan visualisasi curah hujan bulanan di Jakarta Pusat selama tahun 2024 yang menunjukkan pola fluktuasi yang cukup jelas antara musim hujan dan musim kemarau. Curah hujan tertinggi tercatat pada bulan Februari yang dikategorikan sebagai sangat tinggi, sementara bulan Januari, Maret, dan Desember berada dalam kategori tinggi. Kondisi ini mengindikasikan bahwa periode awal dan akhir tahun merupakan puncak musim hujan. Sebaliknya, curah hujan terendah terjadi pada bulan Mei dan Agustus yang tergolong dalam kategori rendah, mencerminkan masa puncak musim kemarau. Bulan-bulan lain seperti April, Juni, Juli, September, dan November menunjukkan curah hujan menengah, yang dapat diasosiasikan sebagai periode transisi antara musim hujan dan kemarau.")
    
    st.title("Optimalisasi Pengadaan Stok Obat")

    st.subheader("Top 10 Item per Cluster, Curah Hujan, dan Bulan")
    st.markdown("""
        Berikut merupakan daftar 10 besar item per Cluster berdasarkan Curah Hujan dan Bulan:
    """)
    
    # Peta nama bulan
    bulan_map = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    
    df = data_grouped_clustered.copy()
    
    # Tambahkan kolom Curah Hujan berdasarkan kolom 'Month'
    def kategori_curah_hujan(bulan):
        if bulan == 2:
            return 'Sangat Tinggi'
        elif bulan in [1, 3, 12]:
            return 'Tinggi'
        elif bulan in [4, 6, 7, 9, 11]:
            return 'Menengah'
        elif bulan in [5, 8, 10]:
            return 'Rendah'
        else:
            return 'Tidak Diketahui'
    
    df['Curah Hujan'] = df['Month'].apply(kategori_curah_hujan)
    df['Month'] = df['Month'].map(bulan_map)
    
    # Filter kolom yang diperlukan
    df_filtered = df[['Cluster', 'Curah Hujan', 'Month', 'Item', 'Supplier', 'Use', 'Qty']].copy()
    
    # Tambahkan Rank
    df_filtered['Rank'] = df_filtered.groupby(['Cluster', 'Curah Hujan', 'Month'])['Qty'].rank(method='first', ascending=False)
    
    # Filter hanya top 10
    df_top10 = df_filtered[df_filtered['Rank'] <= 10].drop(columns='Rank')
    
    # ================== FILTER INTERAKTIF ================== #
    # Pilihan filter
    cluster_options = sorted(df_top10['Cluster'].unique())
    bulan_options = df_top10['Month'].unique()
    
    selected_clusters = st.multiselect("Pilih Cluster:", options=cluster_options, default=cluster_options)
    selected_bulan = st.multiselect("Pilih Bulan:", options=bulan_options, default=bulan_options)
    
    # Terapkan filter ke df_top10
    df_top10_filtered = df_top10[
        (df_top10['Cluster'].isin(selected_clusters)) &
        (df_top10['Month'].isin(selected_bulan))
    ]
    
    # Urutkan data
    df_top10_filtered = df_top10_filtered.sort_values(by=['Cluster', 'Curah Hujan', 'Month', 'Qty'], ascending=[True, True, True, False])
    
    # Tampilkan hasil
    st.dataframe(df_top10_filtered.reset_index(drop=True), use_container_width=True)
    
    # ================== RINGKASAN TOTAL PENJUALAN ================== #
    st.subheader("Rekapitulasi Total Penjualan (Qty) Obat Perbulan")
    st.write("Berikut ini merupakan jumlah total permintaan obat berdasarkan hasil cluster, dikategorikan menurut curah hujan dan bulan:")
    
    # Group data berdasarkan Cluster, Curah Hujan, dan Bulan
    cluster_month_summary = (
        df_top10_filtered.groupby(['Cluster', 'Curah Hujan', 'Month'])['Qty']
        .sum()
        .reset_index()
        .sort_values(by=['Cluster', 'Curah Hujan', 'Qty'], ascending=[True, True, False])
    )
    
    # Tampilkan hasil
    st.dataframe(cluster_month_summary, use_container_width=True)


    st.subheader("Ringkasan")

    st.markdown("""
        Cluster 1:
        Pada Cluster 1, pada kondisi curah hujan menengah, penjualan tertinggi terjadi pada bulan Juli, diikuti oleh November dan Juni. Hal ini menunjukkan bahwa bulan Juni merupakan waktu yang ideal untuk melakukan pengadaan agar dapat mengantisipasi lonjakan permintaan di bulan berikutnya. Pada saat curah hujan rendah, puncak penjualan tercatat pada Mei dan Oktober, sehingga pengadaan sebaiknya dilakukan mulai dari April hingga awal Mei. Sementara itu, saat curah hujan tinggi, jumlah penjualan melonjak drastis di Januari, yang menandakan bahwa pengadaan harus diprioritaskan pada bulan Desember. Untuk kategori sangat tinggi, puncak permintaan terjadi di Februari, sehingga pengadaan optimal dilakukan pada Januari.

        Cluster 2:
        Pada Cluster 2, tren penjualan juga menunjukkan pola yang konsisten. Saat curah hujan menengah, penjualan tertinggi terjadi pada November, disusul oleh bulan April dan Juli. Maka, bulan Oktober menjadi waktu yang strategis untuk pengadaan. Saat curah hujan rendah, penjualan memuncak di Mei, sehingga pengadaan sebaiknya dilakukan pada bulan sebelumnya, yaitu April. Untuk kategori curah hujan tinggi, permintaan tertinggi terjadi pada Maret, yang mengindikasikan bahwa Februari adalah waktu yang tepat untuk mempersiapkan stok. Di sisi lain, saat curah hujan sangat tinggi, puncak penjualan kembali terjadi di Februari, yang memperkuat pentingnya pengadaan pada bulan Januari.

        Cluster 3:
        Pada Cluster 3, pola permintaan cenderung lebih rendah dibandingkan dua cluster sebelumnya, namun tren bulanan tetap dapat diamati. Saat curah hujan menengah, penjualan tertinggi terjadi pada November, menunjukkan bahwa pengadaan ideal dilakukan pada Oktober. Dalam kondisi curah hujan rendah, puncak permintaan tercatat di Oktober, sehingga stok perlu disiapkan mulai dari September. Untuk curah hujan kategori tinggi, penjualan tertinggi tercapai pada Desember, diikuti oleh Januari dan Maret. Ini menunjukkan bahwa pengadaan sebaiknya dilakukan mulai November. Sementara saat curah hujan sangat tinggi, kembali terjadi lonjakan permintaan di Februari, dan pengadaan idealnya dilakukan pada Januari.
    """)

    st.subheader("Kesimpulan")
    st.markdown("""
        Secara keseleuruhan, bulan Januari, Februari, Mei, dan November merupakan periode dengan permintaan tertinggi di berbagai cluster dan kategori curah hujan. Oleh karena itu, strategi pengadaan stok obat di rumah sakit sebaiknya mempertimbangkan karakteristik masing-masing cluster untuk menghindari kekurangan stok dan memastikan ketersediaan obat saat dibutuhkan.
    """)
