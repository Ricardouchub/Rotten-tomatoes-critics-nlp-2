# (2da parte del proyecto)

# Analizador de Sentimiento Bilingüe (EN/ES) con XLM-RoBERTa + Hugging Face Spaces

Este proyecto implementa un analizador de sentimiento para **reseñas de películas** en **inglés y español** usando **Transformers**.  
Se compara un baseline rápido (**DistilRoBERTa**) frente a un modelo **multilingüe** (**XLM-RoBERTa base**) y se despliega una **app web** con **Gradio** en **Hugging Face Spaces** con una salida amigable para usuarios.

---

### [Notebook](./Notebook.ipynb)
### [Aplicación](https://huggingface.co/spaces/Ricardouchub/sentiment-es-en)
### [Modelo en el Hub](https://huggingface.co/Ricardouchub/xlmr-sentiment-es-en)

## Dataset

Se utiliza el dataset de **críticas de Rotten Tomatoes** (reseñas de críticos profesionales), con más de **1 millón** de reseñas.

- **Fuente:** [Kaggle – Rotten Tomatoes Movies and Critic Reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

---

## Herramientas

- **Modelado:** Hugging Face **Transformers**, **Datasets**
- **Entrenamiento:** PyTorch (fp16/bf16 según GPU), `gradient_checkpointing`, `cosine` scheduler, `EarlyStopping`
- **Particionado:** `GroupShuffleSplit` por **película** (evita fuga entre títulos)
- **App & Deploy:** **Gradio** en **Hugging Face Spaces**
- **Métricas:** Accuracy, F1, Precision, Recall, **ROC-AUC**, Matriz de Confusión

---

## Procedimiento

### 1) Preparación de datos
- Auto-detección de columnas (texto/etiqueta/agrupador).
- Limpieza ligera y normalización de la etiqueta.
- **Split por grupos (película)** con `GroupShuffleSplit`.

### 2) Modelado
- **Baselines:** DistilRoBERTa (rápido, EN).
- **Modelo principal:** **XLM-RoBERTa base** (multilingüe EN/ES).
- Submuestreo para comparaciones reproducibles: **100k train / 10k test**.

### 3) Entrenamiento y evaluación
- Entrenamiento con fp16/bf16 (según hardware), `gradient_checkpointing`.
- Evaluación con métricas estándar y **barrido de umbral** para optimizar F1(1).

### 4) Despliegue
- App **Gradio** (Blocks) con:
  - pestaña **Texto único** y **Lote** (una reseña por línea),
  - **ejemplos precargados**,
  - output **amigable** (p. ej., *“Positiva 👍 (97.1%)”*),
  - **umbral fijo** en producción (por defecto `0.48`, configurable).
- **API**: el Space expone endpoints automáticos; consulta el botón **“Use via API”** en la página del Space para ver la ruta exacta y payload.

---

## Resultados (100k train / 10k test)

| Modelo               | Accuracy | F1     | Precision | Recall | AUC   | Umbral |
|----------------------|:-------:|:------:|:---------:|:------:|:-----:|:------:|
| DistilRoBERTa        | 0.8484  | 0.8882 | 0.8426    | **0.9390** | **0.9282** | 0.6046 |
| **XLM-RoBERTa (base)** | **0.8519** | 0.8876 | **0.8646** | 0.9119 | 0.9260 | **0.4800** |

**Matriz de confusión (umbral óptimo por F1 de cada modelo):**
- DistilRoBERTa (0.6046): `[[TN=2463, FP=1125], [FN=391, TP=6021]]`
- XLM-R (0.4800): `[[TN=2672, FP=916],  [FN=565, TP=5847]]`

**Lectura:** XLM-R logra **mayor precisión** (menos FP) y **ligera mejora en accuracy**, sacrificando algo de recall frente a DistilRoBERTa. Además, ofrece **bilingüismo** (EN/ES), por lo que es el modelo elegido para la app.

---

## Cómo usar la App (Space)

1. Escribe una reseña en inglés o español y pulsa **Analizar**.  
2. Para varias reseñas, usa la pestaña **Lote** (una por línea).  
3. El Space incluye un botón **“Use via API”** con el **endpoint** y un **ejemplo de payload** para integrar en tus proyectos.

**Variables de entorno útiles (Space Settings → Variables & secrets):**
- `HF_REPO_ID` → `Ricardouchub/xlmr-sentiment-es-en`
- `THRESHOLD` → `0.48` (puedes re-ajustar según validación)
- `HF_TOKEN` → solo si el repositorio del modelo es **privado**

---

## Conclusiones

XLM-RoBERTa base es el modelo de producción por su bilingüismo y mejor Precisión/Accuracy a umbral ≈ 0.48, manteniendo un recall alto.

El split por película evita fuga y hace la evaluación honesta.

La app web es simple de usar y cuenta con API para integración.

---

## Autor

**Ricardo Urdaneta** 
**[Linkedin](https://www.linkedin.com/in/ricardourdanetacastro/)**