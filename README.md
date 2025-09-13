Este es el segundo repositorio de un proyecto de NLP en dos partes, enfocado en el despliegue de un modelo bilingüe. Para ver el análisis exploratorio y el modelado inicial que condujo a este proyecto, revisa la Parte 1.

| Proyecto NLP | [Parte 1: Análisis Exploratorio y Predictivo](https://github.com/Ricardouchub/Rotten-tomatoes-critics-nlp) | [Parte 2: Modelo Robusto Bilingüe y Despliegue](https://github.com/Ricardouchub/Rotten-tomatoes-critics-nlp-2) |
| :--- | :--- | :--- |
| **Objetivo Principal** | Análisis predictivo e investigativo sobre las críticas | **Construir y desplegar una aplicación funcional** |
| **Idioma** | Inglés | **Bilingüe (EN/ES)** |
| **Modelo Clave** | Logistic Regression | **XLM-RoBERTa** |
| **Producto Final** | Análisis y hallazgos en un Notebook | **Aplicación Web Interactiva** |
| **Enfoque** | Exploración y Modelado | **Producción y Aplicación**  |

---

<p align="center">
  <img width="560" height="187" alt="image" src="https://github.com/user-attachments/assets/b017bc97-3ee9-4f29-9559-a5a0f23c1bdd" />
</p>


Este proyecto implementa un analizador de sentimiento para *reseñas de películas* en *inglés y español* usando **Transformers**.  
Se compara un baseline rápido `DistilRoBERTa` frente a un modelo **multilingüe** `XLM-RoBERTa`base y se despliega una **app web** en *Hugging Face Spaces*.

---

<p align="center">
  <a href="./Notebook.ipynb">Notebook</a> |
  <a href="https://huggingface.co/spaces/Ricardouchub/sentiment-es-en">Aplicación Web</a> |
  <a href="https://huggingface.co/Ricardouchub/xlmr-sentiment-es-en">Modelo</a>
</p>

---

<p align="center">
  <a href="https://huggingface.co/spaces/Ricardouchub/sentiment-es-en">
    <img
      src="https://github.com/user-attachments/assets/a7fec7f1-dd29-4ec2-bbaf-9cb9d2fb84eb"
      alt="Demo en Hugging Face Space: Sentiment EN/ES"
      width="1503"
      height="913"
    />
  </a>
</p>

---

## Características

* **Análisis Bilingüe:** Clasifica texto en inglés y español usando un único modelo.
* **Interfaz Interactiva:** App web simple con pestañas para análisis de texto único y por lotes.
* **API Accesible:** El *Space* expone un endpoint automático para integrar el modelo en otros proyectos.
* **Evaluación Robusta:** El particionado de datos por película (`GroupShuffleSplit`) previene la fuga de datos y garantiza la evaluación del rendimiento.
---

## Herramientas

* **Modelo de IA**
    * **Librerías:** *PyTorch* y *Hugging Face Transformers* para construir y manejar el modelo de lenguaje.
    * **Datos:** *Hugging Face Datasets* para gestionar la información de entrenamiento.

* **Técnicas de Entrenamiento**
    * **Eficiencia:** Se aplicaron `fp16`/`bf16` para acelerar el proceso y `gradient_checkpointing` para reducir el uso de memoria.
    * **Optimización:** Se usó un scheduler `cosine` para ajustar el aprendizaje y `EarlyStopping` para evitar el sobreentrenamiento.

* **Aplicación y Despliegue**
    * **Interfaz:** Se creó una aplicación web interactiva con *Gradio*.
    * **Plataforma:** App desplegada en *Hugging Face Spaces*.

* **Evaluación del Modelo**
    * **Métricas:** Accuracy, F1-Score, Precision, Recall, y ROC-AUC.

---

## Procedimiento

1.  **Preparación de Datos:** Se normalizó y limpió un dataset de reseñas de Rotten Tomatoes. Para evitar que reseñas de la misma película estuvieran en los conjuntos de entrenamiento y prueba, se utilizó `GroupShuffleSplit` agrupando por título de película.
2.  **Modelado y Experimentación:**
    * **Baseline:** Se entrenó un `DistilRoBERTa` (solo inglés) para establecer una línea base de rendimiento.
    * **Modelo Principal:** Se afinó un `XLM-RoBERTa-base` usando una submuestra de **100k para entrenamiento y 10k para prueba**, aprovechando su capacidad multilingüe.
3.  **Entrenamiento y Optimización:** El entrenamiento se realizó con técnicas de eficiencia (fp16/bf16). Post-entrenamiento, se realizó un **barrido de umbrales de decisión** para encontrar el punto que maximizaba el F1-Score para la clase positiva.
4.  **Despliegue:** Se construyó la interfaz con Gradio Blocks, separando la lógica de inferencia para análisis único y por lotes, y se desplegó en un *Space* público.


---

## Resultados

Se comparó el modelo bilingüe (XLM-R) con el baseline monolingüe (DistilRoBERTa).

| Modelo                 | Accuracy | F1     | Precision | Recall | AUC    | Umbral Óptimo |
| :--------------------- | :------: | :----: | :-------: | :----: | :----: | :-----------: |
| DistilRoBERTa          | 0.8484   | 0.8882 | 0.8426    | **0.9390** | **0.9282** | 0.6046        |
| **XLM-RoBERTa (base)** | **0.8519** | 0.8876 | **0.8646** | 0.9119 | 0.9260 | **0.4800** |

>  XLM-RoBERTa fue seleccionado como el modelo final. Aunque DistilRoBERTa tiene un Recall y AUC marginalmente superiores, **XLM-R ofrece mayor precisión** (menos falsos positivos), un **mejor accuracy general** y, fundamentalmente, la **capacidad de procesar texto en español**, lo que era un requisito clave del proyecto.

---

## Dataset

Se utiliza el dataset de **críticas de Rotten Tomatoes** (reseñas de críticos profesionales), con más de **1 millón** de reseñas.

- **Fuente:** [Kaggle – Rotten Tomatoes Movies and Critic Reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)


---

## Autor

**Ricardo Urdaneta** 

**[Linkedin](https://www.linkedin.com/in/ricardourdanetacastro/)**
