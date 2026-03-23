"""
Back-end para API de Busca por Similaridade
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

# Configuração para evitar problemas de threading
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Download do tokenizador do nltk
nltk.download('punkt', quiet=True)

# ==================== CONFIGURAÇÃO ====================
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ==================== CARREGAR MODELO E ÍNDICE ====================
class SearchEngineAPI:
    """Classe wrapper para o motor de busca"""
    
    def __init__(self):
        print("🤖 Carregando modelo de embeddings...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chunks = []
        self.embeddings = []
        self.urls = []
        self.is_loaded = False
        
        # Tenta carregar o índice existente
        self.load_index()
    
    def load_index(self):
        """Carrega o índice do disco"""
        try:
            if os.path.exists('search_index.pkl'):
                with open('search_index.pkl', 'rb') as f:
                    index_data = pickle.load(f)
                
                self.chunks = index_data['chunks']
                self.embeddings = index_data['embeddings']
                self.urls = index_data['urls']
                self.is_loaded = True
                print(f"✅ Índice carregado: {len(self.chunks)} chunks")
                print(f"   Shape dos embeddings: {self.embeddings.shape}")
            else:
                print("⚠️ Nenhum índice encontrado. Criando índice de exemplo...")
                self.create_sample_index()
        except Exception as e:
            print(f"❌ Erro ao carregar índice: {e}")
    
    def create_sample_index(self):
        """Cria um índice de exemplo com conteúdo básico"""
        sample_chunks = [
            "Inteligência Artificial (IA) é um campo da ciência da computação que se dedica ao estudo e desenvolvimento de máquinas inteligentes.",
            "Machine Learning é um subcampo da IA que permite que sistemas aprendam com dados sem serem explicitamente programados.",
            "Deep Learning é uma subárea do Machine Learning que utiliza redes neurais com múltiplas camadas.",
            "Redes Neurais Artificiais são modelos computacionais inspirados no cérebro humano.",
            "Processamento de Linguagem Natural (PLN) permite que computadores entendam e processem linguagem humana."
        ]
        
        self.chunks = sample_chunks
        self.urls = ["https://pt.wikipedia.org/wiki/Inteligência_artificial"] * len(sample_chunks)
        self.embeddings = self.model.encode(sample_chunks)
        self.is_loaded = True
        print(f"✅ Índice de exemplo criado: {len(self.chunks)} chunks")
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Busca os chunks mais similares à query"""
        if not self.is_loaded or len(self.embeddings) == 0:
            return []
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        urls_vistas = set()
        
        for idx in top_indices:
            url = self.urls[idx] if idx < len(self.urls) else "URL não disponível"
            if url in urls_vistas:
                continue
            urls_vistas.add(url)
            
            chunk_embedding = self.embeddings[idx].tolist()[:15]
            
            results.append({
                'chunk': self.chunks[idx],
                'score': float(similarities[idx]),
                'url': url,
                'embedding': chunk_embedding
            })
        
        return results
    
    def get_stats(self):
        """Retorna estatísticas do índice"""
        return {
            'loaded': self.is_loaded,
            'chunks': len(self.chunks),
            'embeddings_dim': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        }

# Instância global
search_engine = SearchEngineAPI()

# ==================== ROTAS ====================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/buscar', methods=['POST'])
def buscar():
    """Endpoint para busca"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'Query vazia'}), 400
        
        resultados = search_engine.search(query, top_k)
        
        # Embedding da query
        query_embedding = search_engine.model.encode([query]).tolist()[0][:15]
        
        return jsonify({
            'success': True,
            'query': query,
            'query_embedding': query_embedding,
            'embedding_dimension': 384,
            'results': resultados,
            'stats': search_engine.get_stats()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/estatisticas', methods=['GET'])
def estatisticas():
    """Retorna estatísticas"""
    return jsonify(search_engine.get_stats())

@app.route('/api/sugestoes', methods=['GET'])
def sugestoes():
    """Retorna sugestões"""
    sugestoes = [
        "O que é inteligência artificial?",
        "Como funciona uma rede neural?",
        "Qual a diferença entre IA e machine learning?",
        "O que são redes neurais convolucionais?",
        "Como funciona o aprendizado profundo?"
    ]
    return jsonify({'sugestoes': sugestoes})

# ==================== EXECUÇÃO ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)