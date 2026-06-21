import os
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import google.generativeai as genai
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class GeminiService:
    """Mengelola integrasi dengan Google Gemini API untuk LLM dan Akses Embeddings."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Parameter deterministik (Temperature & Top P) untuk mencegah halusinasi akademis
        self.model = genai.GenerativeModel('gemini-2.5-flash', generation_config={
            "temperature": 0.2,
            "top_p": 0.5
        })

    def get_embeddings(self, texts: list) -> list:
        """Menghasilkan representasi vektor menggunakan model embedding terbaru Google."""
        result = genai.embed_content(
            model="models/gemini-embedding-001", 
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']

    def analyze_context(self, prompt: str) -> str:
        """Mengirimkan instruksi ke model Gemini untuk inferensi."""
        response = self.model.generate_content(prompt)
        return response.text


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Wrapper kustom agar ChromaDB dapat memanggil fungsi embedding Gemini secara otomatis."""
    
    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service

    def __call__(self, input: Documents) -> Embeddings:
        return self.gemini_service.get_embeddings(input)


class EPubExtractor:
    """Mengelola dekapsulasi arsip berkas EPUB dari folder lokal dan prapemrosesan teks."""
    
    def __init__(self, file_path: str):
        self.buku = epub.read_epub(file_path)
        self.bab_dokumen = [item for item in self.buku.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]

    def get_chapter_names(self) -> list:
        """Mengembalikan daftar nama bab dokumen HTML internal yang ada di dalam buku."""
        return [f"Bab {i + 1} - {chapter.get_name()}" for i, chapter in enumerate(self.bab_dokumen)]

    def extract_pure_arabic(self, chapter_index: int) -> str:
        """Melucuti elemen tag HTML dan memfilter hanya karakter Unicode blok Arab murni."""
        if chapter_index >= len(self.bab_dokumen):
            return ""
            
        bab_terpilih = self.bab_dokumen[chapter_index]
        html_konten = bab_terpilih.get_content().decode('utf-8')
        soup = BeautifulSoup(html_konten, 'html.parser')
        teks_bersih = soup.get_text()
        
        # Mencegah Language Leakage dengan Regex Blok Unicode Arab
        karakter_arab = re.findall(r'[\u0600-\u06FF\s]+', teks_bersih)
        teks_arab_murni = " ".join(karakter_arab)
        return re.sub(r'\s+', ' ', teks_arab_murni).strip()


class RAGChunker:
    """Arsitektur Sentence-Aware Chunking dengan Tumpang Tindih (Overlap)."""
    
    def __init__(self, chunk_size=500, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def generate_chunks(self, text: str) -> list:
        """Memecah teks Arab tanpa memutus konteks gramatikal antar paragraf."""
        kata = text.split()
        potongan = []
        step = self.chunk_size - self.overlap
        
        if len(kata) <= self.chunk_size:
            return [text]
            
        for i in range(0, len(kata), step):
            chunk = " ".join(kata[i:i+self.chunk_size])
            potongan.append(chunk)
            if i + self.chunk_size >= len(kata):
                break
        return potongan


class VectorDBManager:
    """Mengelola pangkalan data vektor ChromaDB lokal untuk Semantic Search."""
    
    def __init__(self, collection_name: str, embedding_function: EmbeddingFunction):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.collection = self._setup_collection()

    def _setup_collection(self):
        """Reset koleksi jika berganti buku/bab agar pencarian tidak tercampur."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        return self.client.create_collection(
            name=self.collection_name, 
            embedding_function=self.embedding_function
        )

    def index_chunks(self, chunks: list):
        """Memasukkan teks ke ruang vektor secara bertahap (batching) untuk menghindari batas API."""
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Google Gemini API membatasi maksimal 100 teks per satu kali proses embed.
        # Kita mencicil injeksi dokumen ini dalam batch berukuran 90 untuk keamanan penuh.
        batch_size = 90
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.collection.add(documents=batch_chunks, ids=batch_ids)

    def search_semantic(self, query: str, n_results=2) -> str:
        """Mencari referensi terdekat berdasarkan pertanyaan pengguna."""
        hasil = self.collection.query(query_texts=[query], n_results=n_results)
        if not hasil['documents'] or not hasil['documents'][0]:
            return ""
        return "\n\n---\n\n".join(hasil['documents'][0])