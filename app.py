import os, torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd  

# =========================
# Configuración
# =========================
REPO_ID   = os.getenv("HF_REPO_ID", "Ricardouchub/xlmr-sentiment-es-en")
THRESHOLD = float(os.getenv("THRESHOLD", "0.48"))
MAX_LEN   = int(os.getenv("MAX_LEN", "224"))
DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
HF_TOKEN  = os.getenv("HF_TOKEN")  

# =========================
# Carga única del modelo
# =========================
tok = AutoTokenizer.from_pretrained(REPO_ID, use_fast=True, token=HF_TOKEN)
mdl = AutoModelForSequenceClassification.from_pretrained(REPO_ID, token=HF_TOKEN).to(DEVICE).eval()

labels = {0: "Negativa", 1: "Positiva"}
icons  = {0: "👎", 1: "👍"}

# =========================
# Funciones de inferencia
# =========================
@torch.inference_mode()
def predict_single(text: str):
    enc = tok([text], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
    probs = torch.softmax(mdl(**enc).logits, dim=-1)[:, 1]  # prob. clase positiva
    score = float(probs[0].item())
    label = 1 if score >= THRESHOLD else 0
    lbl = labels[label]
    conf_pct = round(score * 100, 1)
    emoji = icons[label]
    return f"**El sentimiento es {lbl} {emoji} con una confianza de {conf_pct}%.**"

@torch.inference_mode()
def predict_batch(raw_text: str):
    texts = [t.strip() for t in raw_text.splitlines() if t.strip()]
    if not texts:
        return pd.DataFrame(columns=["Texto", "Sentimiento", "Confianza (%)"])
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
    probs = torch.softmax(mdl(**enc).logits, dim=-1)[:, 1]
    rows = []
    for t, p in zip(texts, probs.tolist()):
        label = 1 if p >= THRESHOLD else 0
        lbl = labels[label]
        emoji = icons[label]
        rows.append({"Texto": t, "Sentimiento": f"{lbl} {emoji}", "Confianza (%)": round(p * 100, 1)})
    return pd.DataFrame(rows)

# =========================
# Interfaz Gradio
# =========================
with gr.Blocks(title="Analizador de Sentimiento Bilingüe de Reseña de Películas (EN/ES)") as demo:
    # Encabezado y métricas actualizadas (XLM-R 100k/10k)
    gr.Markdown(
        f"""
# Analizador de Sentimiento Bilingüe de Reseña de Películas (EN/ES)
Clasifica reseñas en **inglés o español** como *Positivas* o *Negativas* usando **XLM-RoBERTa base** (modelo multilingüe EN/ES).

**Modelo:** `{REPO_ID}` · **Dispositivo:** `{DEVICE}`  
**Resultados:** *Accuracy = **0.8519** · F1 = **0.8876** · Precision = **0.8646** · Recall = **0.9119** · AUC = **0.9260***  
**Umbral operativo:** **{THRESHOLD}**
"""
    )

    # Resumen del proyecto (dataset, split, entrenamiento, API)
    with gr.Tab("Texto único"):
        inp = gr.Textbox(
            label="Escribe una reseña en inglés o español",
            lines=4,
            placeholder="Excelente fotografía, pero la historia es floja… / Great acting, but the plot felt predictable."
        )
        gr.Examples(
            examples=[
                ["This movie was fantastic, a true masterpiece of cinema."],
                ["La película fue una completa pérdida de tiempo, muy aburrida."],
                ["It wasn't bad, but it didn't meet my expectations."],
                ["Me gustó la actuación, aunque la trama era un poco predecible."]
            ],
            inputs=[inp],
            label="Prueba con ejemplos rápidos"
        )
        btn = gr.Button("Analizar", variant="primary")
        out = gr.Markdown(label="Resultado")
        btn.click(predict_single, inputs=inp, outputs=out)

    with gr.Tab("Lote (una por línea)"):
        batch_inp = gr.Textbox(
            label="Pega varias reseñas (una por línea)",
            lines=8,
            placeholder="Escribe una reseña por línea…"
        )
        btn2 = gr.Button("Analizar Lote")
        out2 = gr.Dataframe(label="Resultados", wrap=True)
        btn2.click(predict_batch, inputs=batch_inp, outputs=out2)

    # Footer con autor y enlaces
    gr.Markdown("---")
    gr.Markdown("**Realizado por: Ricardo Urdaneta**")
    with gr.Row():
        gr.Button("LinkedIn", link="https://www.linkedin.com/in/ricardourdanetacastro", size="sm")
        gr.Button("GitHub",   link="https://github.com/Ricardouchub",                  size="sm")

if __name__ == "__main__":
    demo.launch()