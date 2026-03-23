#!/bin/bash

# Baixar recursos do NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Iniciar o servidor
gunicorn backend:app --timeout 120 --workers 1