// Configuração da API
const API_URL = "/api";

// Elementos DOM
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const loading = document.getElementById("loading");
const resultsSection = document.getElementById("resultsSection");
const resultsGrid = document.getElementById("resultsGrid");
const resultsHeader = document.getElementById("resultsHeader");
const queryBadge = document.getElementById("queryBadge");
const emptyState = document.getElementById("emptyState");
const statsText = document.getElementById("statsText");
const suggestionChips = document.getElementById("suggestionChips");

// Variáveis de controle
let currentQuery = "";

// ==================== FUNÇÕES AUXILIARES ====================

// Formatar score como porcentagem
function formatScore(score) {
  return (score * 100).toFixed(2);
}

// Destacar termos da query no texto
function highlightQuery(text, query) {
  if (!query) return text;

  const words = query
    .toLowerCase()
    .split(" ")
    .filter((w) => w.length > 3);
  let highlighted = text;

  words.forEach((word) => {
    const regex = new RegExp(`(${word})`, "gi");
    highlighted = highlighted.replace(
      regex,
      '<mark class="result-highlight">$1</mark>',
    );
  });

  return highlighted;
}

// Truncar texto
function truncateText(text, maxLength = 400) {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + "...";
}

// ==================== CHAMADAS API ====================

// Carregar estatísticas
async function loadStats() {
  try {
    const response = await fetch(`${API_URL}/estatisticas`);
    const data = await response.json();

    if (data.loaded) {
      statsText.textContent = `${data.chunks} chunks indexados`;
    } else {
      statsText.textContent = "Índice não carregado";
    }
  } catch (error) {
    console.error("Erro ao carregar estatísticas:", error);
    statsText.textContent = "Erro ao carregar";
  }
}

// Carregar sugestões
async function loadSuggestions() {
  try {
    const response = await fetch(`${API_URL}/sugestoes`);
    const data = await response.json();

    suggestionChips.innerHTML = "";
    data.sugestoes.forEach((sugestao) => {
      const chip = document.createElement("div");
      chip.className = "suggestion-chip";
      chip.textContent = sugestao;
      chip.onclick = () => {
        searchInput.value = sugestao;
        performSearch();
      };
      suggestionChips.appendChild(chip);
    });
  } catch (error) {
    console.error("Erro ao carregar sugestões:", error);
  }
}

// Realizar busca
async function performSearch() {
  const query = searchInput.value.trim();

  if (!query) {
    showNotification("Por favor, digite uma pergunta", "warning");
    return;
  }

  currentQuery = query;

  loading.style.display = "block";
  resultsSection.style.display = "none";
  emptyState.style.display = "none";

  try {
    const response = await fetch(`${API_URL}/buscar`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        top_k: 5,
        show_embeddings: showEmbeddings, // Envia a preferência
      }),
    });

    const data = await response.json();

    if (data.success) {
      displayResults(data, query);
    } else {
      showNotification(data.error || "Erro na busca", "error");
      emptyState.style.display = "block";
    }
  } catch (error) {
    console.error("Erro na busca:", error);
    showNotification("Erro de conexão com o servidor", "error");
    emptyState.style.display = "block";
  } finally {
    loading.style.display = "none";
  }
}

// Exibir resultados
// Variável global para controlar exibição de embeddings
let showEmbeddings = true;

// Configurar o toggle
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("showEmbeddingsCheckbox");
  if (toggle) {
    toggle.addEventListener("change", (e) => {
      showEmbeddings = e.target.checked;
      // Recarregar resultados se houver uma busca atual
      if (currentQuery) {
        performSearch();
      }
    });
  }
});

// Função para formatar embedding como string legível
function formatEmbedding(embedding, maxValues = 15) {
  if (!embedding || embedding.length === 0) return "[]";

  const values = embedding.slice(0, maxValues);
  const formatted = values.map((v) => v.toFixed(4)).join(", ");

  if (embedding.length > maxValues) {
    return `[${formatted}, ... (${embedding.length - maxValues} mais)]`;
  }
  return `[${formatted}]`;
}

// Função para exibir resultados
function displayResults(data, query) {
  const results = data.results;
  const queryEmbedding = data.query_embedding;
  const embeddingDim = data.embedding_dimension || 384;

  if (!results || results.length === 0) {
    emptyState.style.display = "block";
    resultsSection.style.display = "none";
    return;
  }

  // Atualizar header
  queryBadge.textContent = `"${query}"`;
  resultsHeader.style.display = "flex";
  resultsSection.style.display = "block";
  emptyState.style.display = "none";

  // Gerar cards de resultados
  resultsGrid.innerHTML = "";

  // Mostrar embedding da query (se ativado)
  if (showEmbeddings && queryEmbedding) {
    const queryEmbeddingCard = document.createElement("div");
    queryEmbeddingCard.className = "embedding-card query-embedding";
    queryEmbeddingCard.innerHTML = `
            <div class="embedding-title">
                <i class="fas fa-question-circle"></i>
                EMBEDDING DA QUERY (${embeddingDim} dimensões)
                <span style="font-size: 0.6rem; margin-left: auto;">Primeiros 20 valores</span>
            </div>
            <div class="embedding-values">
                ${formatEmbedding(queryEmbedding, 20)}
            </div>
            <div style="font-size: 0.65rem; color: #666; margin-top: 0.5rem;">
                ⚡ Este vetor numérico representa a semântica da sua pergunta
            </div>
        `;
    resultsGrid.appendChild(queryEmbeddingCard);
  }

  results.forEach((result, index) => {
    const card = document.createElement("div");
    card.className = "result-card";

    const scoreClass =
      result.score > 0.6 ? "high" : result.score > 0.4 ? "medium" : "low";
    const highlightedContent = highlightQuery(result.chunk, query);
    const truncatedContent = truncateText(highlightedContent, 400);

    let embeddingHtml = "";
    if (showEmbeddings && result.embedding) {
      embeddingHtml = `
                <div class="embedding-card" style="margin-top: 1rem;">
                    <div class="embedding-title">
                        <i class="fas fa-chart-line"></i>
                        EMBEDDING DO CHUNK (${embeddingDim} dimensões)
                    </div>
                    <div class="embedding-values">
                        ${formatEmbedding(result.embedding, 15)}
                    </div>
                    <div style="font-size: 0.65rem; color: #666; margin-top: 0.5rem;">
                        🔢 Similaridade calculada via cosseno entre vetores
                    </div>
                </div>
            `;
    }

    card.innerHTML = `
            <div class="result-header">
                <div class="result-rank">${index + 1}</div>
                <div class="result-score ${scoreClass}">
                    <i class="fas fa-chart-simple"></i>
                    ${formatScore(result.score)}% relevância
                </div>
            </div>
            <a href="${result.url}" target="_blank" class="result-url">
                <i class="fas fa-link"></i>
                ${result.url.length > 60 ? result.url.substring(0, 60) + "..." : result.url}
            </a>
            <div class="result-content">
                ${truncatedContent}
            </div>
            ${embeddingHtml}
        `;

    resultsGrid.appendChild(card);
  });

  // Atualizar a chamada da API para incluir o resultado
  if (data.query_embedding) {
    const infoCard = document.createElement("div");
    infoCard.className = "result-card";
    infoCard.style.background = "rgba(16, 185, 129, 0.1)";
    infoCard.style.border = "1px solid rgba(16, 185, 129, 0.3)";
    infoCard.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas fa-info-circle" style="color: #10b981;"></i>
                <strong>📊 Informações Técnicas</strong>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.875rem;">
                • Dimensão dos embeddings: ${embeddingDim}<br>
                • Total de chunks no índice: ${data.stats?.chunks || "N/A"}<br>
                • Similaridade calculada: Cosseno entre vetores (quanto maior, mais similar)<br>
                • Embeddings gerados com: Sentence Transformer (paraphrase-multilingual-MiniLM-L12-v2)
            </div>
        `;
    resultsGrid.appendChild(infoCard);
  }
}

// Notificações
function showNotification(message, type = "info") {
  // Criar elemento de notificação
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
        <i class="fas ${type === "error" ? "fa-exclamation-circle" : "fa-info-circle"}"></i>
        <span>${message}</span>
    `;

  document.body.appendChild(notification);

  // Estilo da notificação
  notification.style.position = "fixed";
  notification.style.bottom = "20px";
  notification.style.right = "20px";
  notification.style.background = type === "error" ? "#ef4444" : "#10b981";
  notification.style.color = "white";
  notification.style.padding = "1rem 1.5rem";
  notification.style.borderRadius = "0.5rem";
  notification.style.display = "flex";
  notification.style.alignItems = "center";
  notification.style.gap = "0.5rem";
  notification.style.zIndex = "1000";
  notification.style.animation = "slideIn 0.3s ease";

  // Remover após 3 segundos
  setTimeout(() => {
    notification.remove();
  }, 3000);
}

// ==================== EVENT LISTENERS ====================

// Buscar ao clicar no botão
searchBtn.addEventListener("click", performSearch);

// Buscar ao pressionar Enter
searchInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    performSearch();
  }
});

// ==================== INICIALIZAÇÃO ====================

// Carregar dados iniciais
loadStats();
loadSuggestions();

// Animação de entrada
document.body.style.opacity = "0";
document.body.style.transition = "opacity 0.5s ease";
setTimeout(() => {
  document.body.style.opacity = "1";
}, 100);
