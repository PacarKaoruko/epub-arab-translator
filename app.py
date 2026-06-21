# Patch untuk ChromaDB khusus di Streamlit Cloud (Lingkungan Linux)
# Menggunakan try-except agar tidak error saat dijalankan di lokal (Windows)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Abaikan dan gunakan sqlite3 bawaan jika dijalankan di komputer lokal

import os
import hashlib
import streamlit as st
from engine import GeminiService, GeminiEmbeddingFunction, EPubExtractor, RAGChunker, VectorDBManager

def main():
    # Konfigurasi Halaman Web
    st.set_page_config(page_title="EPUB Arab RAG Engine", page_icon="📖", layout="wide")
    
    # 1. Inisialisasi API Key & Engine
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("⚠️ Kunci 'GOOGLE_API_KEY' tidak ditemukan di dalam file .streamlit/secrets.toml")
        st.stop()

    gemini_service = GeminiService(api_key)
    chunker = RAGChunker(chunk_size=500, overlap=150)
    embedding_fn = GeminiEmbeddingFunction(gemini_service)

    # 2. Desain UI: Sidebar untuk Kontrol Utama
    with st.sidebar:
        st.title("⚙️ Panel Kontrol")
        st.markdown("Pengaturan sumber buku dan RAG.")
        st.divider()
        
        # Pengecekan folder dataset lokal
        dataset_dir = "dataset epub"
        if not os.path.exists(dataset_dir):
            st.error(f"Folder '{dataset_dir}' tidak ditemukan!")
            st.stop()
            
        # Mengambil file epub, dibatasi maksimal 3 file saja sesuai permintaan
        semua_file = [f for f in os.listdir(dataset_dir) if f.endswith('.epub')]
        epub_files = semua_file[:3] 
        
        if not epub_files:
            st.warning("Belum ada file buku di dalam folder dataset.")
            st.stop()
            
        buku_terpilih = st.selectbox("📂 Pilih Buku EPUB:", epub_files)
        file_path_lengkap = os.path.join(dataset_dir, buku_terpilih)

        DAFTAR_BAHASA = ["Bahasa Indonesia", "English", "日本語 (Jepang)", "العربية (Arab)"]
        bahasa_pilihan = st.selectbox("🎯 Target Terjemahan AI:", DAFTAR_BAHASA)

    # 3. Konten Utama Halaman
    st.title("📖 EPUB Arabic to Text RAG Analyzer")
    st.markdown("Demonstrasi Arsitektur RAG Berbasis Vector Database ChromaDB.")

    if buku_terpilih:
        # Menghindari ekstraksi ulang file fisik berkali-kali (State Management)
        if "extractor" not in st.session_state or st.session_state.get("current_file_name") != buku_terpilih:
            with st.spinner("Membongkar arsip dokumen EPUB..."):
                st.session_state.extractor = EPubExtractor(file_path_lengkap)
                st.session_state.current_file_name = buku_terpilih

        daftar_nama_bab = st.session_state.extractor.get_chapter_names()
        
        pilihan_bab_index = st.selectbox(
            "📑 Pilih Bab untuk Diindeks ke Vector Database:", 
            range(len(daftar_nama_bab)), 
            format_func=lambda x: daftar_nama_bab[x]
        )

        teks_arab_final = st.session_state.extractor.extract_pure_arabic(pilihan_bab_index)

        if not teks_arab_final:
            st.warning("⚠️ Tidak ditemukan teks Arab murni pada bab ini.")
            return

        # Membuat hash unik (MD5) dari nama buku agar ChromaDB tidak error karena karakter Arab
        buku_hash = hashlib.md5(buku_terpilih.encode('utf-8')).hexdigest()[:8]
        koleksi_nama = f"koleksi_buku_{buku_hash}_bab_{pilihan_bab_index}"
        
        # Proses indexing RAG berjalan di balik layar
        vdb_manager = VectorDBManager(koleksi_nama, embedding_fn)
        potongan_rag = chunker.generate_chunks(teks_arab_final)
        
        with st.spinner("⏳ Membangun indeks Vektor (Proses Batching)..."):
            vdb_manager.index_chunks(potongan_rag)
        st.success(f"✅ Vektor DB Siap! {len(potongan_rag)} potongan teks berhasil diindeks.")

        st.divider()
        
        # 4. Desain UI: Penggunaan st.tabs untuk visual yang lebih rapi
        st.subheader("🔍 Kueri Semantic Search")
        kueri_pengguna = st.text_input("Tanyakan sesuatu ke AI (Misal: 'Apa intisari dari mukadimah ini?'):")
        mulai_analisis = st.button("🚀 Analisis Menggunakan RAG", use_container_width=True)

        # Tab antarmuka Streamlit
        tab1, tab2, tab3 = st.tabs(["🤖 Analisis AI", "📄 Referensi Ditemukan (RAG)", "📝 Teks Sumber Asli"])
        
        with tab3:
            # Tab ini hanya untuk melihat teks asli jika pengguna penasaran
            st.write(teks_arab_final)
            
        if mulai_analisis and kueri_pengguna:
            with st.spinner("Menelusuri database vektor..."):
                # Menarik konteks dari ChromaDB
                teks_konteks = vdb_manager.search_semantic(kueri_pengguna, n_results=2)
                
                with tab2:
                    st.info("Konteks teks terdekat dari buku yang diambil oleh Vector Database:")
                    st.write(teks_konteks)
                    
                with tab1:
                    with st.spinner("Menghasilkan analisis akademis dengan Gemini..."):
                        prompt = f"""
                        Kamu adalah ahli filologi dan penerjemah akademis kitab bahasa Arab.
                        PERTANYAAN PENGGUNA: "{kueri_pengguna}"
                        
                        TUGAS MUTLAK:
                        1. Jawab pertanyaan pengguna HANYA berdasarkan konteks teks Arab yang disediakan.
                        2. Terjemahkan bagian relevan ke dalam {bahasa_pilihan}.
                        3. Jelaskan intisarinya.
                        
                        ATURAN FORMATTING VISUAL:
                        Gunakan format Markdown rapi. DILARANG menggunakan Heading 1 (#) atau Heading 2 (##). Gunakan Heading 3 (###).
                        LANGSUNG berikan hasil analisis tanpa basa-basi.
                        
                        KONTEKS TEKS (RAG Chunk):
                        {teks_konteks}
                        """
                        response = gemini_service.analyze_context(prompt)
                        st.write(response)

if __name__ == "__main__":
    main()