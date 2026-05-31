import streamlit as st
import ebooklib
import re
import google.generativeai as genai
from ebooklib import epub
from bs4 import BeautifulSoup

# Mengambil API Key secara aman dari Streamlit Secrets (Kesiapan Step 8)
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Inisialisasi Model Gemini
model = genai.GenerativeModel('gemini-2.5-flash')

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="EPUB Arab Translator",
    page_icon="📖",
    layout="wide" 
)

# Judul Utama
st.title("📖 EPUB Arabic to Text Translator")
st.markdown("Aplikasi web pintar untuk mengekstrak dan menerjemahkan buku EPUB bahasa Arab.")
st.divider()

# 2. Persiapan Layout Kolom
kolom_kontrol1, kolom_kontrol2 = st.columns([2, 1])
kolom_kiri, kolom_kanan = st.columns(2)

# Ambil input bahasa dari user
with kolom_kontrol2:
    DAFTAR_BAHASA = [
        "Bahasa Indonesia", "English", "日本語 (Jepang)", "韓国어 (Korea)", 
        "العربية (Arab)", "Français (Prancis)", "Deutsch (Jerman)", 
        "Español (Spanyol)"
    ]
    bahasa_pilihan = st.selectbox("🎯 Pilih Bahasa Target Terjemahan:", DAFTAR_BAHASA)

# Ambil input file EPUB dari user
with kolom_kontrol1:
    berkas_diunggah = st.file_uploader("📂 Unggah Buku EPUB Bahasa Arab Anda:", type=["epub"])

# Fungsi Optimasi Regex menggunakan Blok Unicode Arab (Mencegah Language Leakage)
def bersihkan_dan_baca_arab(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    teks_bersih = soup.get_text()
    
    # Menjaga karakter khusus Arab murni (termasuk harakat dan angka Arab)
    # \u0600-\u06FF adalah blok utama aksara Arab
    karakter_arab = re.findall(r'[\u0600-\u06FF\s]+', teks_bersih)
    teks_arab_murni = " ".join(karakter_arab)
    
    # Merapikan spasi ganda hasil ekstraksi
    teks_arab_murni = re.sub(r'\s+', ' ', teks_arab_murni)
    return teks_arab_murni.strip()

# Fungsi RAG Chunking: Memecah teks besar menjadi bagian-bagian logis berdasarkan jumlah kata
def buat_potongan_teks(teks, batas_kata=300):
    kata = teks.split()
    potongan = []
    for i in range(0, len(kata), batas_kata):
        potongan.append(" ".join(kata[i:i+batas_kata]))
    return potongan

# 3. Proses Utama Aplikasi (Jika File Sudah Diunggah)
if berkas_diunggah is not None:
    # Simpan sementara file yang diunggah ke dalam disk lokal server
    with open("temp_book.epub", "wb") as f:
        f.write(berkas_diunggah.getbuffer())
        
    # Membaca buku EPUB
    buku = epub.read_epub("temp_book.epub")
    
    # Ekstraksi Bab/Item dokumen dokumen di dalam EPUB yang berisi teks (Dokumen HTML)
    daftar_bab = []
    for item in buku.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            daftar_bab.append(item)
            
    # Buat dropdown list di UI kontrol untuk memilih bab buku
    with kolom_kontrol1:
        pilihan_bab_index = st.selectbox(
            "📖 Pilih Bab / Halaman Buku yang Ingin Dianalisis:",
            range(len(daftar_bab)),
            format_func=lambda x: f"Bab {x + 1} - {daftar_bab[x].get_name()}"
        )
        
    # Ambil konten HTML dari bab yang dipilih user
    bab_terpilih = daftar_bab[pilihan_bab_index]
    html_konten = bab_terpilih.get_content().decode('utf-8')
    
    # Jalankan fungsi optimasi pembersihan data Arab
    teks_arab_final = bersihkan_dan_baca_arab(html_konten)
    
    if not teks_arab_final:
        st.warning("⚠️ Tidak ditemukan teks beraksara Arab murni pada bab ini. Silakan pilih bab lain.")
    else:
        # Implementasi RAG Dasar: Pecah teks Arab menjadi beberapa chunk utuh
        potongan_rag = buat_potongan_teks(teks_arab_final, batas_kata=250)
        
        # Kontrol Navigasi Chunk di kolom kontrol
        with kolom_kontrol2:
            chunk_terpilih = st.selectbox(
                "🧩 Bagian Teks (RAG Chunk):",
                range(len(potongan_rag)),
                format_func=lambda x: f"Bagian {x + 1} dari {len(potongan_rag)}"
            )
            
        teks_chunk_aktif = potongan_rag[chunk_terpilih]
        
        # Tampilkan Teks Sumber di UI Kiri
        with kolom_kiri:
            st.subheader("📝 Teks Sumber Arab (Hasil Filter)")
            with st.container(height=500):
                st.write(teks_chunk_aktif)
                
        # Tombol Eksekusi AI Terjemahan
        mulai_terjemah = st.button("🚀 Terjemahkan & Analisis dengan Gemini AI", use_container_width=True)
        
        if mulai_terjemah:
            with st.spinner("Sedang memproses dokumen akademis Anda melalui Gemini 2.5-Flash..."):
                # System Instruction & Prompting Engineering Tingkat Lanjut (Step 4)
                prompt = f"""
                Kamu adalah seorang asisten akademis profesional, ahli filologi klasik, dan penerjemah kitab bahasa Arab berwawasan luas.
                
                TUGAS MUTLAK:
                1. Terjemahkan teks Arab murni di bawah ini ke dalam {bahasa_pilihan}.
                2. JELASKAN juga intisari akademis, kontekstual, dan analisis struktural dari teks tersebut secara komprehensif.
                3. WAJIB GUNAKAN {bahasa_pilihan} UNTUK SELURUH BALASANMU. Jangan gunakan bahasa lain!
                
                ATURAN FORMATTING VISUAL:
                1. Gunakan format Markdown yang rapi.
                2. DILARANG menggunakan Heading 1 (#) atau Heading 2 (##). Gunakan Heading 3 (###) atau teks tebal (**) saja untuk judul bagian.
                3. Pisahkan bagian "Terjemahan" dan "Penjelasan/Intisari" dengan sangat jelas menggunakan garis pemisah.
                4. LANGSUNG berikan hasil analisis. DILARANG memberikan salam pembuka, ramah tamah, atau basa-basi teks sistem.
                
                Berikut adalah potongan teks Arab murni yang harus dianalisis (RAG Chunk):
                {teks_chunk_aktif}
                """
                
                # Panggil Gemini API
                response = model.generate_content(prompt)
                
                # Tampilkan Hasil Analisis AI di UI Kanan
                with kolom_kanan:
                    st.subheader("🤖 Hasil Terjemahan & Analisis AI")
                    with st.container(height=500):
                        st.write(response.text)
                    st.success("Proses terjemahan dan analisis berhasil diselesaikan.")