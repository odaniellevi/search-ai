from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

class SearchEngineAPI:
    def __init__(self):
        print("🤖 Carregando modelo...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = []
        self.urls = []
        self.is_loaded = False
        self.load_index()
    
    def load_index(self):
        try:
            if os.path.exists('search_index.pkl'):
                with open('search_index.pkl', 'rb') as f:
                    data = pickle.load(f)
                self.chunks = data['chunks']
                self.embeddings = data['embeddings']
                self.urls = data['urls']
                self.is_loaded = True
                print(f"✅ Índice carregado: {len(self.chunks)} chunks")
            else:
                print("⚠️ Nenhum índice encontrado. Usando dados de exemplo.")
                self.load_sample_data()
        except Exception as e:
            print(f"❌ Erro: {e}")
            self.load_sample_data()
    
    def load_sample_data(self):
        """Dados de exemplo caso não tenha índice"""
        self.chunks = [
            "Inteligência Artificial (IA) é um campo da ciência da computação.",
            "Machine Learning permite que sistemas aprendam com dados.",
            "Redes Neurais são inspiradas no cérebro humano.",
            "Deep Learning usa múltiplas camadas de redes neurais.",
            "Processamento de Linguagem Natural permite entender texto."
        ]
        self.urls = ["https://pt.wikipedia.org/wiki/IA"] * 5
        self.embeddings = self.model.encode(self.chunks)
        self.is_loaded = True
        print(f"✅ Dados de exemplo carregados: {len(self.chunks)} chunks")
    
    def search(self, query, top_k=5):
        if not self.is_loaded:
            return []
        
        query_emb = self.model.encode([query])
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx][:400],
                'score': float(similarities[idx]),
                'url': self.urls[idx]
            })
        return results
    
    def get_stats(self):
        return {'loaded': self.is_loaded, 'chunks': len(self.chunks)}

engine = SearchEngineAPI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/buscar', methods=['POST'])
def buscar():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Query vazia'}), 400
    
    results = engine.search(query, 5)
    return jsonify({
        'success': True,
        'query': query,
        'results': results,
        'stats': engine.get_stats()
    })

@app.route('/api/sugestoes', methods=['GET'])
def sugestoes():
    return jsonify({'sugestoes': [
        "O que é inteligência artificial?",
        "Como funciona uma rede neural?",
        "Qual a diferença entre IA e machine learning?",
        "O que é deep learning?",
        "Como funciona o processamento de linguagem natural?"
    ]})

@app.route('/api/estatisticas', methods=['GET'])
def estatisticas():
    return jsonify(engine.get_stats())

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'loaded': engine.is_loaded})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)