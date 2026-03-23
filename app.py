"""
Aplicação de Busca por Similaridade em Textos sobre Inteligência Artificial
Versão Corrigida com:
- Limpeza de texto avançada
- Chunks com sobreposição
- Remoção de duplicatas
- Melhor formatação dos resultados
"""

# ==================== IMPORTAÇÕES ====================
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from typing import List, Tuple
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import re
from urllib.parse import urljoin, urlparse
import time
import warnings
warnings.filterwarnings('ignore')

# Download do tokenizador do nltk
print("Baixando recursos do NLTK...")
nltk.download('punkt')
nltk.download('punkt_tab')
print("✅ NLTK pronto!")


# ==================== CLASSE PRINCIPAL ====================

class SimilaritySearchEngine:
    """
    Motor de busca por similaridade usando embeddings de LLM
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', chunk_size: int = 4):
        """
        Inicializa o motor de busca
        
        Args:
            model_name: Nome do modelo SentenceTransformer
            chunk_size: Número de sentenças por chunk (aumentado para 4)
        """
        print(f"\n🤖 Carregando modelo {model_name}...")
        print("   (Isso pode levar alguns minutos na primeira execução)")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = []
        self.urls = []
        print(f"✅ Modelo carregado com sucesso!")
        print(f"   - Chunk size: {chunk_size} sentenças")
    
    def scrape_website(self, url: str, max_pages: int = 10) -> List[str]:
        """Realiza scraping do site com headers avançados"""
        texts = []
        visited_urls = set()
        urls_to_visit = [url]
        
        # Headers para simular navegador real
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
        print(f"\n🌐 Iniciando scraping do site: {url}")
        print(f"📄 Máximo de páginas: {max_pages}")
        
        while urls_to_visit and len(visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            try:
                print(f"  🔍 Processando: {current_url}")
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove elementos indesejados
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    element.decompose()
                
                # Extrai texto
                text = soup.get_text()
                text = self._clean_text(text)
                
                if text and len(text) > 200:
                    texts.append(text)
                    self.urls.append(current_url)
                    print(f"    ✅ Texto extraído ({len(text)} caracteres)")
                else:
                    print(f"    ⚠️ Texto muito curto, ignorado")
                
                visited_urls.add(current_url)
                
                # Encontra links internos
                domain = urlparse(url).netloc
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(current_url, link['href'])
                    parsed_next = urlparse(next_url)
                    
                    if (parsed_next.netloc == domain or parsed_next.netloc == '') and \
                       not any(skip in next_url.lower() for skip in ['login', 'cadastro', 'perfil', 'editar']):
                        if next_url not in visited_urls and next_url not in urls_to_visit:
                            urls_to_visit.append(next_url)
                
                time.sleep(1)  # Delay respeitoso
                
            except Exception as e:
                print(f"    ❌ Erro ao processar {current_url}: {e}")
                continue
        
        print(f"\n✅ Scraping concluído. {len(texts)} páginas processadas.")
        return texts
    
    def _clean_text(self, text: str) -> str:
        """Limpeza avançada de texto - remove lixo da Wikipedia"""
        # Remove referências
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[editar\]', '', text)
        text = re.sub(r'editar código-fonte', '', text)
        text = re.sub(r'editar', '', text)
        text = re.sub(r'ocultar', '', text)
        
        # Remove cabeçalhos de seção
        text = re.sub(r'^=+\s*.*?\s*=+$', '', text, flags=re.MULTILINE)
        
        # Remove linhas com "Ver também", "Referências", etc
        text = re.sub(r'^(Ver também|Referências|Ligações externas|Notas|Bibliografia).*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Normaliza espaços
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        
        # Remove caracteres especiais mantendo acentos
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\"\'\náàâãéèêíïóôõúç]', '', text)
        
        # Remove linhas muito curtas (lixo)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 40]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def chunk_text(self, texts: List[str]) -> List[str]:
        """Divide o texto em chunks inteligentes com sobreposição"""
        chunks = []
        chunk_urls = []
        
        print(f"\n✂️ Dividindo textos em chunks (cada chunk = {self.chunk_size} sentenças)...")
        
        for text, url in zip(texts, self.urls):
            # Primeiro, limpa o texto
            text = self._clean_text(text)
            
            # Tokeniza em sentenças
            try:
                sentences = sent_tokenize(text)
            except:
                continue
            
            # Remove sentenças muito curtas (lixo)
            sentences = [s for s in sentences if len(s) > 40]
            
            if len(sentences) < 2:
                continue
            
            # Cria chunks com sobreposição para manter contexto
            overlap = 1
            step = max(1, self.chunk_size - overlap)
            
            for i in range(0, len(sentences), step):
                chunk_sentences = sentences[i:i + self.chunk_size]
                if len(chunk_sentences) >= 2:
                    chunk = ' '.join(chunk_sentences)
                    # Remove texto residual
                    chunk = re.sub(r'\beditar\b|\bcódigo\b|\bfonte\b', '', chunk, flags=re.IGNORECASE)
                    chunk = re.sub(r'\s+', ' ', chunk).strip()
                    
                    if len(chunk) > 100 and len(chunk) < 3000:
                        chunks.append(chunk)
                        chunk_urls.append(url)
        
        # Remove chunks duplicados
        unique_chunks = []
        unique_urls = []
        seen = set()
        
        for chunk, url in zip(chunks, chunk_urls):
            chunk_key = chunk[:200]
            if chunk_key not in seen:
                seen.add(chunk_key)
                unique_chunks.append(chunk)
                unique_urls.append(url)
        
        print(f"✅ Texto dividido em {len(unique_chunks)} chunks únicos")
        self.chunks = unique_chunks
        self.urls = unique_urls
        
        return self.chunks
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Gera embeddings para os chunks de texto"""
        print(f"\n🧠 Gerando embeddings para {len(chunks)} chunks...")
        print("   Isso pode levar alguns minutos...")
        
        embeddings = self.model.encode(chunks, show_progress_bar=True, batch_size=16)
        
        print(f"✅ Embeddings gerados. Shape: {embeddings.shape}")
        
        self.embeddings = embeddings
        return embeddings
    
    def save_index(self, filename: str = 'search_index.pkl'):
        """Salva o índice em disco"""
        index_data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'urls': self.urls
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"\n💾 Índice salvo em {filename}")
        print(f"   - {len(self.chunks)} chunks")
        print(f"   - Shape dos embeddings: {self.embeddings.shape}")
    
    def load_index(self, filename: str = 'search_index.pkl'):
        """Carrega o índice do disco"""
        with open(filename, 'rb') as f:
            index_data = pickle.load(f)
        
        self.chunks = index_data['chunks']
        self.embeddings = index_data['embeddings']
        self.urls = index_data['urls']
        
        print(f"\n📂 Índice carregado com sucesso!")
        print(f"   - {len(self.chunks)} chunks disponíveis")
        print(f"   - Shape dos embeddings: {self.embeddings.shape}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Busca os chunks mais similares à query - com remoção de duplicatas"""
        if len(self.embeddings) == 0:
            raise ValueError("Índice vazio. Carregue ou crie um índice primeiro.")
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Pega mais resultados para filtrar duplicatas
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]
        
        results = []
        urls_vistas = set()
        chunks_vistos = set()
        
        for idx in top_indices:
            url = self.urls[idx] if idx < len(self.urls) else "URL não disponível"
            chunk = self.chunks[idx]
            
            # Evitar URLs duplicadas
            if url in urls_vistas:
                continue
            
            # Evitar chunks muito similares
            chunk_hash = hash(chunk[:150])
            if chunk_hash in chunks_vistos:
                continue
            
            urls_vistas.add(url)
            chunks_vistos.add(chunk_hash)
            
            results.append((
                chunk,
                float(similarities[idx]),
                url
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def build_index_from_url(self, start_url: str, max_pages: int = 10):
        """Pipeline completo: scraping -> chunking -> embeddings -> salvamento"""
        print("\n" + "="*60)
        print("🚀 INICIANDO PIPELINE COMPLETO")
        print("="*60)
        
        # Passo 1: Scraping
        texts = self.scrape_website(start_url, max_pages)
        
        if not texts:
            print("❌ Nenhum texto extraído. Pipeline interrompido.")
            return
        
        # Passo 2: Chunking
        chunks = self.chunk_text(texts)
        
        if not chunks:
            print("❌ Nenhum chunk gerado. Pipeline interrompido.")
            return
        
        # Passo 3: Embeddings
        self.generate_embeddings(chunks)
        
        # Passo 4: Salvar índice
        self.save_index()
        
        print("\n" + "="*60)
        print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*60)


# ==================== FUNÇÕES DE BUSCA ====================

def buscar_e_exibir(engine, query, top_k=5):
    """Realiza busca e exibe resultados formatados"""
    print("\n" + "="*80)
    print(f"🔍 BUSCA: \"{query}\"")
    print("="*80)
    
    try:
        resultados = engine.search(query, top_k=top_k)
        
        if not resultados:
            print("❌ Nenhum resultado encontrado.")
            return
        
        print(f"\n📊 TOP {len(resultados)} RESULTADOS MAIS RELEVANTES:\n")
        
        for i, (chunk, similaridade, url) in enumerate(resultados, 1):
            print(f"{'─'*80}")
            print(f"📌 RESULTADO {i}")
            print(f"{'─'*80}")
            print(f"🎯 Similaridade: {similaridade:.4f} ({similaridade*100:.2f}%)")
            print(f"🔗 Fonte: {url}")
            print(f"📝 Conteúdo:")
            
            # Formata o chunk para melhor leitura
            chunk = chunk.strip()
            chunk = re.sub(r'\s+', ' ', chunk)
            
            # Divide em parágrafos lógicos
            if len(chunk) > 400:
                # Mostra primeiras 3 frases
                frases = chunk.split('. ')
                for j, frase in enumerate(frases[:5]):
                    if frase.strip():
                        print(f"   {frase.strip()}.")
                if len(frases) > 5:
                    print("   ...")
            else:
                print(f"   {chunk}")
            
            print()
        
        # Avaliação da qualidade
        melhor_similaridade = resultados[0][1]
        if melhor_similaridade > 0.6:
            print("✅ Excelente correspondência encontrada!")
        elif melhor_similaridade > 0.45:
            print("👍 Boa correspondência encontrada.")
        elif melhor_similaridade > 0.3:
            print("⚠️ Correspondência moderada. Tente refinar sua pergunta.")
        else:
            print("⚠️ Baixa correspondência. Tente uma pergunta diferente.")
                
    except Exception as e:
        print(f"❌ Erro na busca: {e}")


# ==================== INTERFACE INTERATIVA ====================

def interface_interativa(engine):
    """Interface interativa para o usuário fazer perguntas"""
    print("\n" + "="*80)
    print("🎯 SISTEMA DE BUSCA POR SIMILARIDADE SEMÂNTICA")
    print("   Tema: Inteligência Artificial e Tecnologia")
    print("="*80)
    print("\n✨ Faça perguntas sobre Inteligência Artificial, Machine Learning,")
    print("   Deep Learning, Redes Neurais, e outros tópicos de tecnologia.")
    print("\n💡 Exemplos de perguntas:")
    print("   - O que é deep learning?")
    print("   - Como funciona uma rede neural?")
    print("   - Quais as aplicações de IA na medicina?")
    print("   - Diferença entre IA e machine learning")
    print("\n⚠️ Digite 'sair' para encerrar.")
    print("="*80)
    
    while True:
        print("\n" + "─"*80)
        query = input("🔍 Sua pergunta: ").strip()
        
        if query.lower() in ['sair', 'exit', 'quit', 'fim']:
            print("\n👋 Encerrando o sistema de busca. Até mais!")
            break
        
        if not query:
            print("⚠️ Por favor, digite uma pergunta válida.")
            continue
        
        buscar_e_exibir(engine, query, top_k=5)
        
        print("\n" + "─"*80)
        continuar = input("💭 Continuar buscando? (Enter para continuar, 'n' para sair): ").strip()
        if continuar.lower() in ['n', 'nao', 'não', 'no']:
            print("\n👋 Encerrando o sistema de busca. Até mais!")
            break


# ==================== URLs PARA SCRAPING ====================

urls_ia = [
    "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial",
    "https://pt.wikipedia.org/wiki/Aprendizado_de_m%C3%A1quina",
    "https://pt.wikipedia.org/wiki/Aprendizado_profundo",
    "https://pt.wikipedia.org/wiki/Rede_neural_artificial",
    "https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural",
    "https://pt.wikipedia.org/wiki/Visa%C3%A3o_computacional",
    "https://pt.wikipedia.org/wiki/Ci%C3%AAncia_de_dados",
]

print("\n📚 URLs disponíveis para scraping sobre Inteligência Artificial:")
for i, url in enumerate(urls_ia, 1):
    print(f"   {i}. {url}")


# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Função principal da aplicação"""
    
    print("\n" + "🚀"*30)
    print("APLICAÇÃO DE BUSCA POR SIMILARIDADE EM TEXTOS")
    print("Tema: Inteligência Artificial e Tecnologia")
    print("Versão: Corrigida e Otimizada")
    print("🚀"*30)
    
    print("\nOpções:")
    print("1. Criar novo índice (fazer scraping e gerar embeddings)")
    print("2. Carregar índice existente (se já tiver criado antes)")
    
    opcao = input("\nEscolha (1/2): ").strip()
    
    # Criar motor de busca com chunk_size=4 (melhor contexto)
    engine = SimilaritySearchEngine(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        chunk_size=4  # Aumentado para 4 sentenças por chunk
    )
    
    if opcao == "1":
        # Escolher URL inicial
        print("\nEscolha uma URL inicial (recomendado: 4 - Rede Neural):")
        for i, url in enumerate(urls_ia, 1):
            print(f"{i}. {url}")
        
        try:
            escolha = int(input("\nDigite o número da URL (1-7): "))
            if 1 <= escolha <= len(urls_ia):
                url_inicial = urls_ia[escolha - 1]
            else:
                url_inicial = urls_ia[3]  # Padrão: Rede Neural
                print(f"Opção inválida. Usando: {url_inicial}")
        except:
            url_inicial = urls_ia[3]  # Padrão: Rede Neural
            print(f"Usando URL padrão: {url_inicial}")
        
        # Número de páginas
        try:
            max_paginas = int(input("\nNúmero de páginas para processar (3-10, padrão 6): "))
            max_paginas = max(3, min(10, max_paginas))
        except:
            max_paginas = 6
            print(f"Usando padrão: {max_paginas} páginas")
        
        # Executar pipeline
        engine.build_index_from_url(url_inicial, max_pages=max_paginas)
        
    elif opcao == "2":
        try:
            engine.load_index()
        except FileNotFoundError:
            print("\n❌ Nenhum índice encontrado. Crie um novo primeiro (opção 1).")
            return
    else:
        print("Opção inválida!")
        return
    
    # Iniciar interface interativa
    interface_interativa(engine)


# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    main()