---
title: Sentiment EN/ES (XLM-R)
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

Demo multilingüe (EN/ES) de análisis de sentimiento con XLM-RoBERTa.
Carga el modelo desde Hugging Face Hub usando `from_pretrained`.

- **Modelo**: `${HF_REPO_ID}` (configurable en Settings → Variables)
- **Umbral positivo**: `${THRESHOLD}` (default 0.48)
- **Máx tokens**: `${MAX_LEN}` (default 224)

> Requisitos en `requirements.txt`. El Space descargará y cacheará los pesos al iniciar.