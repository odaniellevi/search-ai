"""
Back-end para API de Busca por Similaridade
Servidor Flask que expõe endpoints para consulta
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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = []
        self.urls = []
        self.is_loaded = False
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
                print("⚠️ Nenhum índice encontrado. Execute app.py primeiro.")
        except Exception as e:
            print(f"❌ Erro ao carregar índice: {e}")
    
    def search(self, query: str, top_k: int = 5) -> list:
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
            
            chunk_embedding = self.embeddings[idx].tolist()
            
            results.append({
                'chunk': self.chunks[idx],
                'score': float(similarities[idx]),
                'url': url,
                'embedding': chunk_embedding[:20]
            })
        
        return results
    
    def get_stats(self):
        if not self.is_loaded:
            return {'loaded': False, 'chunks': 0}
        
        return {
            'loaded': True,
            'chunks': len(self.chunks),
            'embeddings_dim': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        }

search_engine = SearchEngineAPI()

# ==================== ROTAS ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/buscar', methods=['POST'])
def buscar():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'Query vazia'}), 400
        
        resultados = search_engine.search(query, top_k)
        query_embedding = search_engine.model.encode([query]).tolist()[0][:20]
        
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
    return jsonify(search_engine.get_stats())

@app.route('/api/sugestoes', methods=['GET'])
def sugestoes():
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
    print("\n" + "="*50)
    print("🚀 SERVIDOR DE BUSCA SEMÂNTICA")
    print("="*50)
    print(f"📊 Índice carregado: {'Sim' if search_engine.is_loaded else 'Não'}")
    print(f"📦 Chunks disponíveis: {len(search_engine.chunks)}")
    print(f"🌐 Acesse: http://localhost:{port}")
    print("🛑 Para parar: Ctrl+C")
    print("="*50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=port)