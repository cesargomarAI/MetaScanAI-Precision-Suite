"""
MetaScan AI - Diagnostic Support Suite
Author: César Gomar Applied AI Engineer
Target: Oncology Departments / Clinical Research
Features: XAI Heatmaps, Quality Control, multi-lang PDF reporting.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. INTERNATIONALIZATION (i18n) ---
TRANSLATIONS = {
    "en": {
        "title": "MetaScan AI | Diagnostic Suite",
        "upload": "Upload H&E Histology Image",
        "qc_check": "Quality Control (QC)",
        "qc_passed": "Image quality validated for clinical inference.",
        "qc_failed": "QC Warning: Image may be blurred or low contrast.",
        "analysis": "Neural Interpretation",
        "heatmap_title": "Explainable AI: Malignancy Heatmap",
        "heatmap_desc": "Red areas indicate patterns suspicious of malignant neoplasia.",
        "result_tumor": "MALIGNANT NEOPLASIA DETECTED",
        "result_normal": "NO EVIDENCE OF MALIGNANCY",
        "generate_report": "Export Clinical Report (PDF)",
        "patient_id": "Patient/Case ID",
        "hospital": "Reference Hospital",
        "doctor": "Requesting Oncologist",
        "disclaimer": "Clinical Decision Support System. Not a standalone diagnostic.",
        "metadata_header": "Clinical Metadata (DICOM Standards)"
    },
    "es": {
        "title": "MetaScan AI | Suite de Diagnóstico",
        "upload": "Cargar Imagen Histológica H&E",
        "qc_check": "Control de Calidad (QC)",
        "qc_passed": "Calidad de imagen validada para inferencia clínica.",
        "qc_failed": "Aviso QC: Imagen borrosa o con bajo contraste.",
        "analysis": "Interpretación Neuronal",
        "heatmap_title": "IA Explicable: Mapa de Calor de Malignidad",
        "heatmap_desc": "Las zonas rojas indican patrones sospechosos de neoplasia maligna.",
        "result_tumor": "NEOPLASIA MALIGNA DETECTADA",
        "result_normal": "SIN EVIDENCIA DE MALIGNIDAD",
        "generate_report": "Exportar Informe Clínico (PDF)",
        "patient_id": "ID del Paciente/Caso",
        "hospital": "Hospital de Referencia",
        "doctor": "Oncólogo Solicitante",
        "disclaimer": "Sistema de Soporte a la Decisión Clínica. No es un diagnóstico autónomo.",
        "metadata_header": "Metadatos Clínicos (Estándar DICOM)"
    }
}

# Mapping our 8 clinical categories
TISSUE_TYPES = {
    0: "TUMOR (Malignant)",
    1: "STROMA (Healthy)",
    2: "COMPLEX STROMA (Healthy)",
    3: "LYMPHOCYTES (Healthy)",
    4: "DEBRIS (Normal/Inert)",
    5: "NORMAL MUCOSA",
    6: "ADIPOSE (Healthy)",
    7: "EMPTY (Background)"
}

# --- 2. PROFESSIONAL LAYOUT & BRANDING ---
st.set_page_config(page_title="MetaScan AI", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stAlert { border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #004a99; }
    </style>
    """, unsafe_allow_html=True)

def perform_qc(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score > 100 

def get_gradcam_heatmap(model, img_array, class_index=0):
    """
    Generate Grad-CAM heatmap.
    Uses explicit input/output connection to avoid Sequential AttributeError.
    """
    # 1. Access the base EfficientNet layer
    # model.layers[0] is your base_model (EfficientNetB0)
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer("top_activation")

    # 2. Re-build the forward pass logic for the Grad-CAM model
    # This ensures Keras follows the path from input to the FINAL output
    img_input = base_model.input
    
    # Path to activations
    conv_layer_output = last_conv_layer.output
    
    # Path to final predictions (following your Sequential layers)
    x = base_model.output
    for layer in model.layers[1:]: # GAP, BN, and Dense(8)
        x = layer(x)
    final_output = x

    # 3. Create the Gradient Model with explicit tensors
    grad_model = tf.keras.models.Model(inputs=[img_input], outputs=[conv_layer_output, final_output])

    # 4. Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# --- 3. SIDEBAR ---
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.warning("Logo not found.")
    
    lang = st.selectbox("🌐 Language", ["English", "Español"])
    lang_code = "en" if lang == "English" else "es"
    t = TRANSLATIONS[lang_code]
    
    st.divider()
    st.subheader(t['metadata_header'])
    p_id = st.text_input(t['patient_id'], "PX-2026-X")
    hosp = st.text_input(t['hospital'], "Clinical Oncology Center")
    doc = st.text_input(t['doctor'], "Dr. Pathologist")
    st.caption(t['disclaimer'])

# --- 4. MAIN ENGINE ---
st.title(t['title'])

@st.cache_resource
def load_engine():
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, models

    # 1. Rebuild the exact architecture we trained in Colab
    # We use weights=None because we will load our own custom weights
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(150, 150, 3))
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(8, activation='softmax') # The 8 classes from Kather dataset
    ])

    # 2. Load the weights (The 'knowledge')
    try:
        model.load_weights('metascan_weights_v3.weights.h5')
        model(np.zeros((1, 150, 150, 3)))
        return model
    except Exception as e:
        st.error(f"Critical error loading weights: {e}")
        st.stop()

model = load_engine()

uploaded_file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg", "tif", "tiff"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(raw_img, caption="Original Histology Slide", use_container_width=True)
        if perform_qc(raw_img):
            st.success(f"✅ {t['qc_passed']}")
        else:
            st.warning(f"⚠️ {t['qc_failed']}")

    with col2:
        st.subheader(t['analysis'])
        
        # Preprocessing at 150x150 (Match training size)
        img_resized = raw_img.resize((150, 150))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array) 
        
        # Multiclass Prediction
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]
        
        # Clinical logic: Only class 0 (TUMOR) is considered Malignant
        is_malignant = (class_idx == 0)
        tissue_name = TISSUE_TYPES[class_idx]
        
        # Results Display
        result_label = t['result_tumor'] if is_malignant else t['result_normal']
        color = "#d9534f" if is_malignant else "#5cb85c"
        
        st.markdown(f"""
            <div style="background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;">
                <h1 style="margin:0;">{result_label}</h1>
                <h3 style="margin:0;">Detected: {tissue_name}</h3>
                <h4 style="margin:0;">Confidence: {confidence*100:.2f}%</h4>
            </div>
        """, unsafe_allow_html=True)

        # XAI: Heatmap for TUMOR class
        st.divider()
        st.write(f"### {t['heatmap_title']}")
        # We always generate heatmap for class 0 (TUMOR) to see where malignancy might be
        heatmap = get_gradcam_heatmap(model, img_array, class_index=0)
        
        heatmap_resized = cv2.resize(heatmap, (raw_img.size[0], raw_img.size[1]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        superimposed_img = cv2.addWeighted(np.array(raw_img), 0.6, heatmap_colored, 0.4, 0)
        st.image(superimposed_img, caption=t['heatmap_desc'], use_container_width=True)

    # --- 5. REPORT GENERATION ---
    st.divider()
    
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        
        # Diccionario local para etiquetas específicas del PDF que faltaban
        pdf_texts = {
            "en": {
                "header": "METASCAN AI - CLINICAL REPORT",
                "section1": "1. PATIENT & CASE INFORMATION",
                "section2": "2. DIAGNOSTIC INTERPRETATION",
                "result": "RESULT",
                "confidence": "Algorithm Confidence",
                "tissue": "Detected Tissue Category",
                "date": "Date"
            },
            "es": {
                "header": "METASCAN AI - INFORME CLÍNICO",
                "section1": "1. INFORMACIÓN DEL PACIENTE Y CASO",
                "section2": "2. INTERPRETACIÓN DIAGNÓSTICA",
                "result": "RESULTADO",
                "confidence": "Confianza del Algoritmo",
                "tissue": "Categoría de Tejido Detectada",
                "date": "Fecha"
            }
        }
        
        # Usamos lang_code que viene del selector de la sidebar
        pt = pdf_texts[lang_code]
        
        # Header & Branding
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(0, 74, 153)
        pdf.cell(0, 15, pt["header"], ln=True, align='C')
        
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"{pt['date']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
        
        # 1. Patient Info
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, pt["section1"], ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 7, f"{t['patient_id']}: {p_id}", ln=True)
        pdf.cell(0, 7, f"{t['hospital']}: {hosp}", ln=True)
        pdf.cell(0, 7, f"{t['doctor']}: {doc}", ln=True)
        
        # 2. Diagnosis
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, pt["section2"], ln=True)
        pdf.set_font("Arial", 'B', 14)
        
        # Traducimos el nombre del tejido si es necesario (o lo dejamos técnico)
        # Aquí usamos las etiquetas que ya tienes en TRANSLATIONS (t['result_tumor'], etc.)
        if is_malignant:
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 10, f"{pt['result']}: {t['result_tumor']}", ln=True)
        else:
            pdf.set_text_color(0, 150, 0)
            pdf.cell(0, 10, f"{pt['result']}: {t['result_normal']} ({tissue_name})", ln=True)
            
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 7, f"{pt['confidence']}: {confidence*100:.2f}%", ln=True)
        pdf.cell(0, 7, f"{pt['tissue']}: {tissue_name}", ln=True)
        
        # 3. Disclaimer (Usamos el del diccionario principal)
        pdf.ln(20)
        pdf.set_font("Arial", 'I', 8)
        pdf.multi_cell(0, 5, t['disclaimer'])
        
        # Fix binario que ya validamos
        pdf_content = pdf.output(dest='S')
        if isinstance(pdf_content, (bytes, bytearray)):
            return bytes(pdf_content)
        return pdf_content.encode('latin-1')

    st.download_button(
        label=f"📥 {t['generate_report']}", 
        data=generate_pdf(), 
        file_name=f"Report_{p_id}.pdf", 
        mime="application/pdf"
    )

else:
    st.info("System ready. Please upload a biopsy slide.")