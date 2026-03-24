from flask import Flask, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Dados de exemplo
chunks = [
    "Inteligência Artificial (IA) é um campo da ciência da computação que estuda a criação de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana.",
    "Machine Learning é um subcampo da IA que permite que sistemas aprendam padrões a partir de dados, sem serem explicitamente programados.",
    "Redes Neurais Artificiais são modelos computacionais inspirados no cérebro humano, capazes de aprender padrões complexos.",
    "Deep Learning utiliza redes neurais com múltiplas camadas para aprender representações hierárquicas de dados.",
    "Processamento de Linguagem Natural (PLN) permite que computadores entendam, interpretem e gerem linguagem humana."
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/buscar', methods=['POST'])
def buscar():
    from flask import request
    data = request.get_json()
    query = data.get('query', '')
    
    # Retorna resultados fixos
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            'chunk': chunk,
            'score': 0.95 - (i * 0.03),
            'url': 'https://pt.wikipedia.org/wiki/Inteligência_artificial'
        })
    
    return jsonify({
        'success': True,
        'query': query,
        'results': results,
        'stats': {'loaded': True, 'chunks': len(chunks)}
    })

@app.route('/api/sugestoes', methods=['GET'])
def sugestoes():
    return jsonify({'sugestoes': [
        "O que é inteligência artificial?",
        "Como funciona uma rede neural?",
        "Qual a diferença entre IA e machine learning?"
    ]})

@app.route('/api/estatisticas', methods=['GET'])
def estatisticas():
    return jsonify({'loaded': True, 'chunks': len(chunks)})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)