from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

class SearchEngineAPI:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.urls = []
        self.is_loaded = False
        self.load_index()
    
    def load_index(self):
        try:
            if os.path.exists('search_index.pkl'):
                with open('search_index.pkl', 'rb') as f:
                    index_data = pickle.load(f)
                
                self.chunks = index_data['chunks']
                self.embeddings = index_data['embeddings']
                self.urls = index_data['urls']
                self.is_loaded = True
                print(f"✅ Índice carregado: {len(self.chunks)} chunks")
            else:
                print("⚠️ Nenhum índice encontrado.")
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    def search(self, query: str, top_k: int = 5) -> list:
        if not self.is_loaded or len(self.embeddings) == 0:
            return []
        
        # Nota: Precisamos de um método para gerar embedding da query
        # Sem sentence-transformers, não temos como gerar embeddings
        # Isso é um problema - precisamos de uma solução alternativa
        
        return []

search_engine = SearchEngineAPI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/buscar', methods=['POST'])
def buscar():
    return jsonify({'error': 'Modelo não carregado. Recrie o índice primeiro.'}), 500

@app.route('/api/sugestoes', methods=['GET'])
def sugestoes():
    return jsonify({'sugestoes': ["O que é IA?", "Como funciona rede neural?"]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)