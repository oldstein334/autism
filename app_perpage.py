import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Load model CNN dengan path yang lebih robust
model_path = os.path.join(os.path.dirname(__file__), "model", "model_deteksi_autisme.h5")

# Cek apakah model ada
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model berhasil dimuat dari: {model_path}")
except Exception as e:
    raise Exception(f"❌ Gagal memuat model: {str(e)}")

# Daftar pertanyaan gejala autisme
pertanyaan = [
    "👀 Menghindari kontak mata",
    "🗣️ Terlambat bicara atau tidak berbicara",
    "🔁 Sering mengulang kata atau gerakan (stimming)",
    "🧏 Tidak merespons saat dipanggil namanya",
    "🤝 Kesulitan berinteraksi atau bermain dengan anak lain"
]

# Fungsi prediksi utama
def predict(image, gejala):
    if image is None:
        return "❗ Silakan upload gambar terlebih dahulu.", "", ""

    try:
        # Validasi gambar
        if not isinstance(image, Image.Image):
            return "❗ Format gambar tidak valid.", "", ""
            
        # Resize dan convert to array
        image = image.resize((224, 224))
        img_array = np.array(image)
        
        # Pastikan gambar memiliki 3 channel (RGB)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return "❗ Gambar harus dalam format RGB.", "", ""

        # Deteksi wajah menggunakan OpenCV
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return (
                "❌ Tidak terdeteksi wajah pada gambar.\n"
                "Pastikan:\n"
                "• Gambar menunjukkan wajah dengan jelas\n"
                "• Wajah menghadap ke depan\n"
                "• Pencahayaan cukup baik\n"
                "• Format gambar JPG/PNG",
                "", ""
            )

        # Proses gambar untuk prediksi
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        return (
            f"❌ Error memproses gambar: {str(e)}\n"
            "Pastikan gambar dalam format yang valid (JPG/PNG).",
            "", ""
        )

    # Prediksi CNN dengan error handling
    try:
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        pred_autism_from_image = confidence < 0.65
    except Exception as e:
        return (
            f"❌ Error saat prediksi model: {str(e)}",
            "", ""
        )

    # Skor dari kuisioner
    skor_gejala = len(gejala)
    persen_gejala = (skor_gejala / len(pertanyaan)) * 100
    pred_autism_from_form = persen_gejala >= 60    # Interpretasi hasil
    if pred_autism_from_image and pred_autism_from_form:
        hasil = "🔴 Tinggi Kemungkinan Autisme"
        penjelasan = (
            "Model mendeteksi kemungkinan autisme dari gambar wajah dan "
            "gejala checklist juga mendukung. Segera konsultasi ke profesional."
        )
    elif pred_autism_from_image or pred_autism_from_form:
        hasil = "🟠 Sedang / Perlu Observasi"
        penjelasan = (
            "Model atau checklist menunjukkan potensi gejala autisme. "
            "Perlu pengamatan lebih lanjut atau konsultasi lanjutan."
        )
    else:
        hasil = "🟢 Tidak Terindikasi Autisme"
        penjelasan = (
            "Model dan kuisioner tidak menunjukkan indikasi autisme. "
            "Namun tetap pantau perkembangan anak secara berkala."
        )

    confidence_str = f"Confidence (Normal): {confidence:.2f} ({'Terdeteksi Autisme' if pred_autism_from_image else 'Normal'})"
    checklist_str = f"Checklist Terpilih: {skor_gejala}/{len(pertanyaan)} → {persen_gejala:.0f}% kemungkinan Autisme"

    hasil_akhir = (
        f"{hasil}\n\n"
        f"📷 Prediksi Gambar Wajah\n{confidence_str}\n\n"
        f"📝 Hasil Kuisioner\n{checklist_str}\n\n"
        f"📌 Kesimpulan\n{penjelasan}\n\n"
        f"🔗 Konsultasi dan Penanganan Lebih Lanjut:\n"
        f"- 👉 [Konsultasi Psikolog Anak - Halodoc](https://www.halodoc.com/cari-dokter/psikolog-anak)\n"
        f"- 📖 [Penanganan Autisme - Halodoc](https://www.halodoc.com/kesehatan/autisme)"
    )

    return hasil_akhir, confidence_str, checklist_str

# Custom CSS untuk styling per-page dengan warna yang mudah dibaca
custom_css = """
/* Background dan tema utama */
.gradio-container {
    background: linear-gradient(135deg, #a8c8ec 0%, #7faadb 50%, #5b8fd1 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
}

/* Page containers */
.page-container {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 20px;
    padding: 30px;
    margin: 20px auto;
    max-width: 900px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    text-align: center;
}

/* Header styling */
.header-section {
    text-align: center;
    padding: 30px 20px;
    background: white;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
}

.logo-title {
    color: #1a202c;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.subtitle {
    color: #2d3748;
    font-size: 1.4em;
    margin-bottom: 15px;
    font-weight: 600;
}

.description {
    color: #4a5568;
    font-size: 1.1em;
    line-height: 1.6;
    max-width: 600px;
    margin: 0 auto;
}

/* Content sections */
.content-section {
    background: white;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
    text-align: left;
}

.section-title {
    color: white;
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 15px;
    border-bottom: 2px solid #4299e1;
    padding-bottom: 8px;
    background: linear-gradient(45deg, #2b6cb0, #1a365d);
    padding: 15px;
    border-radius: 10px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* Illustration area */
.illustration-area {
    text-align: center;
    padding: 40px 20px;
    background: #f7fafc;
    border-radius: 15px;
    margin: 20px 0;
}

.illustration-text {
    color: #4a5568;
    font-size: 4em;
    margin-bottom: 15px;
}

.illustration-caption {
    color: #718096;
    font-size: 1.1em;
    font-weight: 500;
}

/* Warning dan info boxes */
.warning-box {
    background: #fef5e7;
    border: 2px solid #ed8936;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    color: #744210;
    font-weight: 500;
    text-align: left;
}

.info-box {
    background: #ebf8ff;
    border: 2px solid #4299e1;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    color: #2c5282;
    text-align: left;
    font-weight: 500;
}

/* Footer styling */
.footer-section {
    text-align: center;
    padding: 25px;
    background: white;
    border-radius: 15px;
    margin-top: 30px;
    box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
}

.footer-title {
    color: #1a202c;
    font-size: 1.3em;
    font-weight: bold;
    margin-bottom: 15px;
}

.footer-link {
    color: #2b6cb0 !important;
    text-decoration: none;
    font-weight: 700;
    font-size: 1.1em;
    transition: all 0.3s ease;
    border-bottom: 2px solid transparent;
}

.footer-link:hover {
    color: #1a365d !important;
    border-bottom: 2px solid #2b6cb0;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

/* Button styling yang lebih jelas */
.btn, button {
    background: linear-gradient(45deg, #2b6cb0, #1a365d) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 15px 30px !important;
    font-size: 1.1em !important;
    font-weight: bold !important;
    margin: 10px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    box-shadow: 0 4px 15px rgba(43, 108, 176, 0.3) !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2) !important;
}

.btn:hover, button:hover {
    background: linear-gradient(45deg, #1a365d, #2c5282) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(43, 108, 176, 0.4) !important;
    color: white !important;
}

/* Primary button styling */
.btn-primary, button[variant="primary"] {
    background: linear-gradient(45deg, #2b6cb0, #1a365d) !important;
    color: white !important;
    font-weight: bold !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
}

/* Secondary button styling */
.btn-secondary, button[variant="secondary"] {
    background: linear-gradient(45deg, #4a5568, #2d3748) !important;
    color: white !important;
    font-weight: bold !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
}

/* Checkbox styling untuk lebih jelas */
input[type="checkbox"] {
    transform: scale(1.3);
    margin-right: 10px;
}

/* Label untuk checkbox */
.checkbox-group label {
    color: #1a202c !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    cursor: pointer !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease !important;
}

.checkbox-group label:hover {
    background-color: #f7fafc !important;
}

/* Tab styling yang lebih jelas */
.tab-nav button {
    color: #1a202c !important;
    font-weight: 600 !important;
    background-color: #f7fafc !important;
    border: 2px solid #e2e8f0 !important;
}

.tab-nav button.selected {
    background-color: #2b6cb0 !important;
    color: white !important;
    border-color: #2b6cb0 !important;
}

/* Link dalam teks yang bisa diklik */
a {
    color: #2b6cb0 !important;
    font-weight: 600 !important;
    text-decoration: underline !important;
}

a:hover {
    color: #1a365d !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
}

/* Upload area styling - PASTIKAN ICON TERLIHAT JELAS */
.gr-form .gr-image,
.gr-image-upload,
.gr-image,
.gradio-image {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 15px !important;
    overflow: visible !important;
    position: relative !important;
    padding: 0 !important;
}

/* Hapus semua pseudo-elements yang bisa memberikan background */
.gr-image-upload::before,
.gr-image-upload::after,
.gr-image::before,
.gr-image::after {
    display: none !important;
    content: none !important;
    background: none !important;
}

/* Drop zone tanpa background sama sekali */
.gr-image-upload .drop-zone,
.gr-image .drop-zone,
.drop-zone {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: white !important;
}

/* Semua elemen child juga tanpa background */
.gr-image-upload *,
.gr-image * {
    background: transparent !important;
    background-color: transparent !important;
}

/* Icon upload, clipboard, dan web - STYLING ULTIMATE UNTUK VISIBILITY */
.gr-image-upload svg,
.gr-image-upload .icon,
.gr-image-upload button svg,
.gr-image-upload button .icon,
.gr-image-upload button path,
.gr-image-upload button use,
.gr-image-upload button circle,
.gr-image-upload button rect,
.gr-image-upload button polygon,
.gr-image svg,
.gr-file-upload svg,
.upload-area svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    width: 24px !important;
    height: 24px !important;
    filter: drop-shadow(0 0 3px rgba(0,0,0,0.8)) !important;
    z-index: 9999 !important;
    position: relative !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    pointer-events: auto !important;
}

/* Icon hover state */
.gr-image-upload button:hover svg,
.gr-image-upload button:hover .icon,
.gr-image-upload button:hover path,
.gr-image-upload button:hover use,
.gr-image-upload button:hover circle,
.gr-image-upload button:hover rect,
.gr-image-upload button:hover polygon {
    color: #f59e0b !important;
    fill: #f59e0b !important;
    stroke: #f59e0b !important;
    filter: drop-shadow(0 0 5px rgba(245, 158, 11, 0.8)) !important;
    opacity: 1 !important;
}

/* Override untuk semua kemungkinan selector icon */
svg[data-name="folder"],
svg[data-name="clipboard"],
svg[data-name="camera"],
svg[data-name="upload"],
svg[data-name="web"],
svg[data-name="file"],
.gr-image-upload [data-testid*="icon"],
.gr-image-upload [class*="icon"],
.gr-image-upload .upload-icon,
.gr-image-upload .file-icon,
.gr-image-upload .clipboard-icon,
.gr-image-upload .camera-icon {
    color: white !important;
    fill: white !important;
    stroke: white !important;
    z-index: 9999 !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    filter: drop-shadow(0 0 3px rgba(0,0,0,0.8)) !important;
}

/* Styling untuk text dan label di upload area */
.upload-text,
.upload-icon,
.file-upload svg,
.file-upload .icon,
.gr-image-upload .upload-text,
.gr-image-upload .drop-text {
    color: white !important;
    fill: white !important;
    text-shadow: 0 0 3px rgba(0,0,0,0.8) !important;
    filter: drop-shadow(0 0 3px rgba(0,0,0,0.8)) !important;
}

/* Upload button dan icon styling */
[data-testid="upload-button"],
.gr-image-upload [data-testid="upload-button"] {
    color: white !important;
    background: rgba(43, 108, 176, 0.8) !important;
    border: 1px solid white !important;
}

[data-testid="upload-button"]:hover,
.gr-image-upload [data-testid="upload-button"]:hover {
    background: rgba(26, 54, 93, 0.9) !important;
    color: white !important;
}

[data-testid="upload-button"] svg,
.gr-image-upload [data-testid="upload-button"] svg {
    color: white !important;
    fill: white !important;
    filter: drop-shadow(0 0 3px rgba(0,0,0,0.8)) !important;
}

/* Button upload dengan styling khusus untuk icon file, clipboard, web */
.gr-image-upload button[title*="file"],
.gr-image-upload button[aria-label*="file"],
.gr-image-upload button:nth-child(1),
.gr-image-upload button:first-child {
    background: transparent !important;
    border: none !important;
    color: white !important;
    font-weight: normal !important;
    text-shadow: none !important;
    padding: 8px !important;
}

.gr-image-upload button[title*="clipboard"],
.gr-image-upload button[aria-label*="clipboard"],
.gr-image-upload button:nth-child(2) {
    background: transparent !important;
    border: none !important;
    color: white !important;
    font-weight: normal !important;
    text-shadow: none !important;
    padding: 8px !important;
}

.gr-image-upload button[title*="web"],
.gr-image-upload button[aria-label*="web"],
.gr-image-upload button:nth-child(3) {
    background: transparent !important;
    border: none !important;
    color: white !important;
    font-weight: normal !important;
    text-shadow: none !important;
    padding: 8px !important;
}

/* Hover effects sederhana untuk button upload */
.gr-image-upload button[title*="file"]:hover,
.gr-image-upload button:first-child:hover,
.gr-image-upload button[title*="clipboard"]:hover,
.gr-image-upload button:nth-child(2):hover,
.gr-image-upload button[title*="web"]:hover,
.gr-image-upload button:nth-child(3):hover {
    background: rgba(255, 255, 255, 0.1) !important;
    transform: none !important;
    box-shadow: none !important;
    color: #f59e0b !important;
}

/* Pastikan semua SVG dalam button upload terlihat */
.gr-image-upload button svg {
    color: white !important;
    fill: white !important;
    stroke: white !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    z-index: 10000 !important;
    filter: drop-shadow(0 0 2px rgba(0,0,0,0.8)) !important;
}

.gr-image-upload button:hover svg {
    color: #f59e0b !important;
    fill: #f59e0b !important;
    stroke: #f59e0b !important;
    filter: drop-shadow(0 0 3px rgba(245, 158, 11, 0.8)) !important;
}

/* Result text styling */
.result-text {
    color: #1a202c !important;
    font-size: 1em !important;
    line-height: 1.6 !important;
    font-weight: 500 !important;
}

/* Result section headers - make them bold */
.result-text:has-text("Prediksi Gambar Wajah"),
.result-text:has-text("Hasil Kuisioner"),
.result-text:has-text("Kesimpulan") {
    font-weight: 700 !important;
}
"""

# Gradio Interface dengan layout per-page
with gr.Blocks(css=custom_css, title="🧠 Autisense - Deteksi Dini Autisme", theme=gr.themes.Soft()) as iface:
    
    # PAGE 1: HALAMAN UTAMA (HOME)
    with gr.Group(visible=True) as home_page:
        gr.HTML("""
        <div class="page-container">
            <div class="header-section">
                <div class="logo-title">🧠 Autisense</div>
                <div class="subtitle">Deteksi Dini Autisme pada Anak</div>
                <div class="description">
                    Aplikasi AI untuk screening awal potensi autisme melalui analisis wajah dan kuisioner gejala
                </div>
            </div>
            
            <div class="illustration-area">
                <div class="illustration-text">👶👧👦</div>
                <div class="illustration-caption">Bantu deteksi dini autisme pada anak dengan teknologi AI</div>
            </div>
              <div class="info-box">
                <strong style="color: #1a365d; font-size: 1.2em;">ℹ️ Tentang Aplikasi:</strong><br><br>
                <span style="color: #2d3748; font-weight: 500;">
                Autisense menggunakan kecerdasan buatan untuk membantu deteksi dini autisme pada anak melalui:
                <br>• Analisis gambar wajah dengan model CNN
                <br>• Kuisioner gejala-gejala umum autisme
                </span>
            </div>
        </div>
        """)
        
        with gr.Row():
            start_button = gr.Button("🚀 MULAI", variant="primary", size="lg")
            info_button = gr.Button("📚 Tentang Autisme", variant="secondary", size="lg")
    
    # PAGE 2: HALAMAN UPLOAD GAMBAR
    with gr.Group(visible=False) as upload_page:
        gr.HTML("""
        <div class="page-container">
            <div class="header-section">
                <div class="logo-title">📷 Upload Gambar</div>
                <div class="subtitle">Wajah Anak</div>                <div class="description">
                    Upload gambar wajah anak untuk analisis dengan model AI
                </div>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="content-section">')
                gr.Markdown('<div class="section-title">📁 Pilih Gambar</div>')
                image_input = gr.Image(
                    type="pil", 
                    label="Upload gambar wajah anak (format JPG/PNG)",
                    height=400
                )
                gr.HTML('</div>')
                
                gr.HTML("""
                <div class="warning-box">
                    <strong style="color: #744210; font-size: 1.1em;">⚠️ Panduan Upload:</strong><br>
                    <span style="color: #744210; font-weight: 500;">
                    • Pastikan gambar menunjukkan wajah dengan jelas<br>
                    • Wajah menghadap ke depan<br>
                    • Pencahayaan cukup baik<br>
                    • Format JPG atau PNG
                    </span>
                </div>
                """)
        
        with gr.Row():
            back_to_home = gr.Button("⬅️ Kembali", variant="secondary", size="lg")
            to_questionnaire = gr.Button("Lanjutkan ➡️", variant="primary", size="lg")
    
    # PAGE 3: HALAMAN KUISIONER
    with gr.Group(visible=False) as questionnaire_page:
        gr.HTML("""
        <div class="page-container">
            <div class="header-section">
                <div class="logo-title">📝 Kuisioner</div>
                <div class="subtitle">Checklist Gejala Autisme</div>
                <div class="description">
                    Pilih gejala yang sesuai dengan kondisi anak Anda
                </div>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="content-section">')
                gr.Markdown('<div class="section-title">✅ Gejala yang Diamati</div>')
                checkbox_input = gr.CheckboxGroup(
                    choices=pertanyaan,
                    label="Pilih gejala yang sesuai:"
                )
                gr.HTML('</div>')
        
        with gr.Row():
            back_to_upload = gr.Button("⬅️ Kembali", variant="secondary", size="lg")
            analyze_button = gr.Button("🔍 Analisis Sekarang", variant="primary", size="lg")
    
    # PAGE 4: HALAMAN HASIL DETEKSI
    with gr.Group(visible=False) as results_page:
        gr.HTML("""
        <div class="page-container">
            <div class="header-section">
                <div class="logo-title">🧠 Hasil Deteksi</div>
                <div class="subtitle">Analisis Lengkap</div>
                <div class="description">
                    Hasil prediksi berdasarkan gambar dan kuisioner
                </div>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="content-section">')
                gr.Markdown('<div class="section-title">📋 Kesimpulan Lengkap</div>')
                result_output = gr.Textbox(
                    label="Hasil Prediksi",
                    lines=12,
                    max_lines=20,
                    elem_classes="result-text"
                )
                gr.HTML('</div>')
                
                with gr.Tabs():
                    with gr.Tab("📊 Analisis Gambar"):
                        confidence_output = gr.Textbox(
                            label="Confidence Score",
                            lines=4,
                            elem_classes="result-text"
                        )
                    
                    with gr.Tab("📝 Skor Kuisioner"):
                        checklist_output = gr.Textbox(
                            label="Hasil Checklist",
                            lines=4,
                            elem_classes="result-text"
                        )
        
        gr.HTML("""
        <div class="footer-section">
            <div class="footer-title">🔗 Konsultasi Lebih Lanjut</div>
            <p style="color: #2d3748; font-size: 1.1em; line-height: 1.6;">
                <a href="https://www.halodoc.com/cari-dokter/psikolog-anak" target="_blank" class="footer-link">
                    👉 Konsultasi Psikolog Anak - Halodoc
                </a><br><br>
                <a href="https://www.halodoc.com/kesehatan/autisme" target="_blank" class="footer-link">
                    📖 Informasi Penanganan Autisme
                </a>
            </p>
        </div>
        """)
        
        with gr.Row():
            restart_button = gr.Button("🔄 Analisis Baru", variant="primary", size="lg")
    
    # PAGE 5: HALAMAN INFORMASI AUTISME
    with gr.Group(visible=False) as info_page:
        gr.HTML("""
        <div class="page-container">
            <div class="header-section">
                <div class="logo-title">📚 Mengenal Autisme</div>
                <div class="subtitle">Informasi Penting</div>
            </div>
            
            <div class="content-section">
                <div class="section-title">🎯 Apa itu Autisme?</div>
                <div style="color: #2d3748; font-size: 1.1em; line-height: 1.6;">
                    Autisme adalah gangguan perkembangan neurobiologis yang mempengaruhi komunikasi, 
                    interaksi sosial, dan perilaku anak. Deteksi dini sangat penting untuk memberikan 
                    intervensi yang tepat.
                </div>
            </div>
            
            <div class="content-section">
                <div class="section-title">🔍 Gejala Umum Autisme:</div>
                <div style="color: #2d3748; font-size: 1.1em; line-height: 1.8;">
                    • Kesulitan komunikasi verbal dan non-verbal<br>
                    • Menghindari kontak mata<br>
                    • Perilaku repetitif (stimming)<br>
                    • Kesulitan berinteraksi sosial<br>
                    • Keterlambatan perkembangan bicara<br>
                    • Sensitivitas terhadap suara, cahaya, atau tekstur
                </div>
            </div>
            
            <div class="content-section">
                <div class="section-title">💡 Penanganan dan Terapi:</div>
                <div style="color: #2d3748; font-size: 1.1em; line-height: 1.8;">
                    • Terapi perilaku (ABA - Applied Behavior Analysis)<br>
                    • Terapi wicara dan bahasa<br>
                    • Terapi okupasi<br>
                    • Pendidikan khusus<br>
                    • Dukungan keluarga dan lingkungan
                </div>
            </div>
              <div class="warning-box">
                <strong style="color: #744210; font-size: 1.2em;">⚠️ Penting untuk Diingat:</strong><br><br>
                <span style="color: #744210; font-weight: 500;">
                Setiap anak dengan autisme adalah unik. Diagnosis dan penanganan harus dilakukan 
                oleh profesional yang berpengalaman. Aplikasi ini hanya sebagai alat screening awal.
                </span>
            </div>
        </div>
        """)
        
        back_to_main = gr.Button("⬅️ Kembali ke Beranda", variant="secondary", size="lg")
    
    # FUNGSI NAVIGASI ANTAR HALAMAN
    def show_upload_page():
        return (
            gr.update(visible=False),  # home_page
            gr.update(visible=True),   # upload_page  
            gr.update(visible=False),  # questionnaire_page
            gr.update(visible=False),  # results_page
            gr.update(visible=False)   # info_page
        )
    
    def show_questionnaire_page():
        return (
            gr.update(visible=False),  # home_page
            gr.update(visible=False),  # upload_page
            gr.update(visible=True),   # questionnaire_page
            gr.update(visible=False),  # results_page
            gr.update(visible=False)   # info_page
        )
    
    def show_results_page():
        return (
            gr.update(visible=False),  # home_page
            gr.update(visible=False),  # upload_page
            gr.update(visible=False),  # questionnaire_page
            gr.update(visible=True),   # results_page
            gr.update(visible=False)   # info_page
        )
    
    def show_home_page():
        return (
            gr.update(visible=True),   # home_page
            gr.update(visible=False),  # upload_page
            gr.update(visible=False),  # questionnaire_page
            gr.update(visible=False),  # results_page
            gr.update(visible=False)   # info_page
        )
    
    def show_info_page():
        return (
            gr.update(visible=False),  # home_page
            gr.update(visible=False),  # upload_page
            gr.update(visible=False),  # questionnaire_page
            gr.update(visible=False),  # results_page
            gr.update(visible=True)    # info_page
        )
    
    def analyze_and_show_results(image, gejala):
        hasil, confidence, checklist = predict(image, gejala)
        
        return (
            gr.update(visible=False),  # home_page
            gr.update(visible=False),  # upload_page
            gr.update(visible=False),  # questionnaire_page
            gr.update(visible=True),   # results_page
            gr.update(visible=False),  # info_page
            hasil,                     # result_output
            confidence,                # confidence_output
            checklist                  # checklist_output
        )
    
    # EVENT HANDLERS
    start_button.click(
        show_upload_page,
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page]
    )
    
    to_questionnaire.click(
        show_questionnaire_page,
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page]
    )
    
    analyze_button.click(
        analyze_and_show_results,
        inputs=[image_input, checkbox_input],
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page, 
                result_output, confidence_output, checklist_output]
    )
    
    # Tombol navigasi
    back_to_home.click(
        show_home_page,
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page]
    )
    
    back_to_upload.click(
        show_upload_page,
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page]
    )
    
    restart_button.click(
        show_home_page,
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page]
    )
    
    info_button.click(
        show_info_page,
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page]
    )
    
    back_to_main.click(
        show_home_page,
        outputs=[home_page, upload_page, questionnaire_page, results_page, info_page]
    )

if __name__ == "__main__":
    print("🚀 Memulai Autisense - Aplikasi deteksi autisme...")
    print(f"📍 Model dimuat dari: {model_path}")
    print("🌐 Aplikasi akan berjalan di localhost")
    print("🎨 Menggunakan tampilan per-page sesuai desain")
    
    iface.launch(
        server_name="127.0.0.1",
        share=False,
        debug=True
    )
