// ==================== CONFIGURAÇÃO ====================
const BACKEND_URL = "https://unscripted-musingly-dawne.ngrok-free.dev";

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
let showEmbeddings = true;

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

// ==================== CHAMADAS API ====================

// Carregar sugestões (usando o backend do Colab)
async function loadSuggestions() {
  try {
    const response = await fetch(`${BACKEND_URL}/sugestoes`);
    const data = await response.json();

    suggestionChips.innerHTML = "";
    if (data.sugestoes) {
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
    }
  } catch (error) {
    console.error("Erro ao carregar sugestões:", error);
    // Fallback: sugestões padrão
    const fallback = [
      "O que é inteligência artificial?",
      "Como funciona uma rede neural?",
      "Qual a diferença entre IA e machine learning?",
    ];
    fallback.forEach((sugestao) => {
      const chip = document.createElement("div");
      chip.className = "suggestion-chip";
      chip.textContent = sugestao;
      chip.onclick = () => {
        searchInput.value = sugestao;
        performSearch();
      };
      suggestionChips.appendChild(chip);
    });
  }
}

// Carregar estatísticas (do Colab)
async function loadStats() {
  try {
    const response = await fetch(`${BACKEND_URL}/`);
    const data = await response.json();

    if (data.chunks) {
      statsText.textContent = `${data.chunks} chunks indexados`;
    } else {
      statsText.textContent = "Índice carregado";
    }
  } catch (error) {
    console.error("Erro ao carregar estatísticas:", error);
    statsText.textContent = "Backend online";
  }
}

// Realizar busca no Colab
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
    const response = await fetch(`${BACKEND_URL}/buscar`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: query }),
    });

    const data = await response.json();

    if (data.results) {
      displayResults(data, query);
    } else {
      showNotification(data.error || "Erro na busca", "error");
      emptyState.style.display = "block";
    }
  } catch (error) {
    console.error("Erro na busca:", error);
    showNotification("Erro de conexão com o backend do Colab", "error");
    emptyState.style.display = "block";
  } finally {
    loading.style.display = "none";
  }
}

// Exibir resultados
function displayResults(data, query) {
  const results = data.results;
  const queryEmbedding = data.query_embedding;
  const embeddingDim = 384; // Dimensão do all-MiniLM-L6-v2

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

  results.forEach((result, index) => {
    const card = document.createElement("div");
    card.className = "result-card";

    const scoreClass =
      result.score > 0.6 ? "high" : result.score > 0.4 ? "medium" : "low";
    const highlightedContent = highlightQuery(result.chunk, query);
    const truncatedContent = truncateText(highlightedContent, 400);

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
    `;

    resultsGrid.appendChild(card);
  });

  // Informações técnicas
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
      • Modelo: all-MiniLM-L6-v2 (80MB)<br>
      • Total de chunks: ${data.stats?.chunks || results.length}<br>
      • Similaridade: Cosseno entre vetores (quanto maior, mais similar)<br>
      • Backend rodando no Google Colab com GPU
    </div>
  `;
  resultsGrid.appendChild(infoCard);
}

// Notificações
function showNotification(message, type = "info") {
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
    <i class="fas ${type === "error" ? "fa-exclamation-circle" : "fa-info-circle"}"></i>
    <span>${message}</span>
  `;

  document.body.appendChild(notification);

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

  setTimeout(() => {
    notification.remove();
  }, 3000);
}

// ==================== EVENT LISTENERS ====================

// Configurar toggle de embeddings
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("showEmbeddingsCheckbox");
  if (toggle) {
    toggle.addEventListener("change", (e) => {
      showEmbeddings = e.target.checked;
      if (currentQuery) {
        performSearch();
      }
    });
  }
});

searchBtn.addEventListener("click", performSearch);
searchInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    performSearch();
  }
});

// ==================== INICIALIZAÇÃO ====================

loadStats();
loadSuggestions();

// Animação de entrada
document.body.style.opacity = "0";
document.body.style.transition = "opacity 0.5s ease";
setTimeout(() => {
  document.body.style.opacity = "1";
}, 100);
