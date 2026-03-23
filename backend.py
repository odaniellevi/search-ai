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
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download do tokenizador do nltk
nltk.download('punkt', quiet=True)

# ==================== CONFIGURAÇÃO ====================
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Permite requisições de diferentes origens

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
                print("⚠️ Nenhum índice encontrado. Execute app.py primeiro para criar o índice.")
        except Exception as e:
            print(f"❌ Erro ao carregar índice: {e}")
    
    def search(self, query: str, top_k: int = 5) -> list:
        if not self.is_loaded or len(self.embeddings) == 0:
            return []
        
        # Gera embedding da query
        query_embedding = self.model.encode([query])
        
        # Calcula similaridades
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Pega os top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepara resultados
        results = []
        urls_vistas = set()
        
        for idx in top_indices:
            url = self.urls[idx] if idx < len(self.urls) else "URL não disponível"
            
            # Evitar URLs duplicadas
            if url in urls_vistas:
                continue
            urls_vistas.add(url)
            
            # Pegar o embedding do chunk
            chunk_embedding = self.embeddings[idx].tolist()  # Converte numpy para lista
            
            results.append({
                'chunk': self.chunks[idx],
                'score': float(similarities[idx]),
                'url': url,
                'embedding': chunk_embedding[:20]  # Mostra primeiros 20 valores
            })
        
        return results
    
    def get_stats(self):
        """Retorna estatísticas do índice"""
        if not self.is_loaded:
            return {'loaded': False, 'chunks': 0}
        
        return {
            'loaded': True,
            'chunks': len(self.chunks),
            'embeddings_dim': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        }

# Instância global do motor de busca
search_engine = SearchEngineAPI()

# ==================== ROTAS DA API ====================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/buscar', methods=['POST'])
def buscar():
    """
    Endpoint para busca
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        show_embeddings = data.get('show_embeddings', True)
        
        if not query:
            return jsonify({'error': 'Query vazia'}), 400
        
        # Realiza busca
        resultados = search_engine.search(query, top_k)
        
        # Gerar embedding da query também
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
    """Retorna estatísticas do sistema"""
    return jsonify(search_engine.get_stats())

@app.route('/api/sugestoes', methods=['GET'])
def sugestoes():
    """Retorna sugestões de perguntas"""
    sugestoes = [
        "O que é inteligência artificial?",
        "Como funciona uma rede neural?",
        "Qual a diferença entre IA e machine learning?",
        "O que são redes neurais convolucionais?",
        "Como funciona o aprendizado profundo?",
        "Quais as aplicações de visão computacional?",
        "O que é processamento de linguagem natural?",
        "Como funciona o aprendizado por reforço?"
    ]
    return jsonify({'sugestoes': sugestoes})

# ==================== EXECUÇÃO ====================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SERVIDOR DE BUSCA SEMÂNTICA")
    print("="*50)
    print(f"📊 Índice carregado: {'Sim' if search_engine.is_loaded else 'Não'}")
    print(f"📦 Chunks disponíveis: {len(search_engine.chunks)}")
    print("\n🌐 Acesse: http://localhost:5000")
    print("🛑 Para parar: Ctrl+C")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)