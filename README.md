# MetaScan AI | Precision Histopathology Suite 🔬🦾

> ### 🌐 [LIVE DEMO: ACCESS THE CLINICAL APP HERE](https://metascanai-precision-suite.streamlit.app/)
> *Experience the future of colorectal cancer detection in real-time. High-resolution biopsy analysis at your fingertips.*

---

## 🇺🇸 ENGLISH VERSION

### 📄 Project Overview
**MetaScan AI** is an advanced Clinical Decision Support System (CDSS) designed to assist pathologists in the automated detection and classification of **Colorectal Cancer (CRC)** through H&E-stained histological slides. By bridging the gap between raw pathological data and State-of-the-Art Deep Learning, this suite optimizes the diagnostic workflow with clinical-grade precision and interpretable results.

### 🛠️ Technology Stack & Architecture
This "Crown Jewel" architecture integrates a robust stack of technologies to ensure high performance and clinical reliability:
* **Deep Learning Framework:** `TensorFlow 2.15` & `Keras 3` (leveraging JAX/PyTorch backend capabilities).
* **Neural Backbone:** `EfficientNet-B0` — Chosen for its optimal balance between computational efficiency and feature extraction via compound scaling.
* **Explainable AI (XAI):** `Grad-CAM` (Gradient-weighted Class Activation Mapping) for visual interpretability of neural decisions.
* **Backend & Frontend:** `Streamlit` — High-performance framework for medical data visualization.
* **Computer Vision:** `OpenCV` (Advanced morphological filtering) and `Pillow` (Clinical-grade image handling for `.tif`, `.png`, `.jpg`).
* **Scientific Computing:** `NumPy` for high-speed matrix manipulation of histological tensors.

### 🧬 Scientific Evidence & Foundations
MetaScan AI is built upon validated histopathological research and standardized oncology datasets:
1.  **Dataset Foundation:** Trained and validated on the **Kather Multiclass Dataset**, a peer-reviewed collection of 100,000+ histological images from the **NCT Biobank (National Center for Tumor Diseases, Heidelberg)**.
2.  **Histopathological Standards:** Our 8-class classification system follows the morphological criteria for **Colorectal Cancer (CRC)** tissue differentiation: *Tumor, Stroma, Lymphocytes, Mucus, Mucosa, Adipose, Debris, and Smooth Muscle.*
3.  **Transfer Learning Methodology:** Utilized weights pre-trained on `ImageNet` with specialized **Fine-Tuning** on histological features, a methodology validated in journals like *Nature Medicine* and *The Lancet Digital Health*.
4.  **Clinical Reliability:** The system targets a high Sensitivity/Specificity balance, achieving a **93% Validation Accuracy**, specifically tuned to avoid common "Stroma-as-Tumor" false positives.

### 🌟 Key Features
* **Multiclass Differentiation:** Beyond binary detection, it classifies 8 distinct tissue types.
* **Explainable Diagnosis:** Integrated Grad-CAM highlights the exact morphological patterns driving the malignancy prediction.
* **Automated Quality Control (QC):** Laplacian-based blur detection to ensure slide diagnostic viability.
* **Bilingual Clinical Reporting:** Instant generation of PDF reports in English and Spanish.
* **Medical Standard Support:** Native handling of high-resolution `.tif` and `.tiff` files.

---

## 🇪🇸 VERSIÓN EN ESPAÑOL

### 📄 Descripción del Proyecto
**MetaScan AI** es un Sistema de Soporte a la Decisión Clínica (CDSS) avanzado, diseñado para asistir a patólogos en la detección y clasificación automatizada de **Cáncer Colorrectal (CRC)** a través de laminillas histológicas teñidas con H&E. Esta suite optimiza el flujo diagnóstico integrando datos patológicos con Deep Learning de última generación para obtener una precisión de grado clínico y resultados interpretables.

### 🛠️ Stack Tecnológico y Arquitectura
Esta arquitectura integra un robusto conjunto de tecnologías para garantizar un alto rendimiento y fiabilidad clínica:
* **Framework de Deep Learning:** `TensorFlow 2.15` y `Keras 3` (aprovechando las nuevas capacidades de backend JAX/PyTorch).
* **Backbone Neuronal:** `EfficientNet-B0` — Seleccionada por su equilibrio óptimo entre eficiencia computacional y capacidad de extracción de características.
* **IA Explicable (XAI):** `Grad-CAM` (Mapeo de Activación de Clases por Gradiente) para la interpretabilidad visual de las decisiones neuronales.
* **Backend y Frontend:** `Streamlit` — Framework de alto rendimiento para la visualización de datos médicos.
* **Visión Artificial:** `OpenCV` (Filtrado morfológico avanzado) y `Pillow` (Manejo de imágenes de grado clínico).
* **Computación Científica:** `NumPy` para la manipulación de tensores histológicos a alta velocidad.

### 🧬 Evidencia Científica y Fundamentos
MetaScan AI está construida sobre investigación histopatológica validada y conjuntos de datos oncológicos estandarizados:
1.  **Base del Dataset:** El modelo fue entrenado y validado con el **Kather Multiclass Dataset**, una colección revisada por pares de más de 100,000 imágenes del **NCT Biobank (Heidelberg, Alemania)**.
2.  **Estándares Histopatológicos:** Clasificación basada en criterios morfológicos para 8 tipos de tejido: *Tumor, Estroma, Linfocitos, Mucus, Mucosa, Adipose (Grasa), Debris (Restos) y Músculo Liso.*
3.  **Metodología de Transfer Learning:** Uso de pesos pre-entrenados en `ImageNet` con **Ajuste Fino (Fine-Tuning)** en características histológicas, metodología validada en revistas como *Nature Medicine*.
4.  **Fiabilidad Clínica:** Alcanza una **Precisión de Validación del 93%**, específicamente ajustado para minimizar falsos positivos comunes de tipo "Estroma-como-Tumor".

### 🌟 Características Principales
* **Diferenciación Multiclase:** Clasifica 8 tipos de tejido distintos para un análisis contextual completo.
* **Diagnóstico Explicable:** Grad-CAM resalta los patrones morfológicos que impulsan la predicción de malignidad.
* **Control de Calidad (QC):** Detección automática de desenfoque para asegurar la viabilidad del diagnóstico.
* **Informes Clínicos Bilingües:** Generación instantánea de reportes en PDF en Inglés y Español.
* **Soporte de Estándares Médicos:** Manejo nativo de archivos de alta resolución `.tif` y `.tiff`.

---
## ⚖️ Disclaimer / Descargo de Responsabilidad
*This tool is intended for research purposes only. Final diagnosis must be validated by a board-certified pathologist. / Esta herramienta está destinada a fines de investigación. El diagnóstico final debe ser validado por un patólogo certificado.*