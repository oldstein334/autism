import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model CNN
model = tf.keras.models.load_model("model_deteksi_autisme.h5")

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
        # Resize dan convert to array
        image = image.resize((224, 224))
        img_array = np.array(image)

        # Deteksi wajah menggunakan OpenCV
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return (
                "❌ Gambar tidak terdeteksi sebagai wajah anak.\n"
                "Harap unggah gambar wajah yang jelas dari depan (bukan hewan, objek, atau latar belakang).",
                "", ""
            )

        # Proses lanjut jika wajah terdeteksi
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        return (
            "❌ Gagal memproses gambar.\nPastikan gambar dalam format JPG/PNG dan menunjukkan wajah anak.",
            "", ""
        )

    # Prediksi CNN
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    pred_autism_from_image = confidence < 0.65

    # Skor dari kuisioner
    skor_gejala = len(gejala)
    persen_gejala = (skor_gejala / len(pertanyaan)) * 100
    pred_autism_from_form = persen_gejala >= 60

    # Interpretasi hasil
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
        f"📷 **Prediksi Gambar Wajah**\n{confidence_str}\n\n"
        f"📝 **Hasil Kuisioner**\n{checklist_str}\n\n"
        f"📌 **Kesimpulan**\n{penjelasan}\n\n"
        f"🔗 **Konsultasi dan Penanganan Lebih Lanjut:**\n"
        f"- 👉 [Konsultasi Psikolog Anak - Halodoc](https://www.halodoc.com/cari-dokter/psikolog-anak)\n"
        f"- 📖 [Penanganan Autisme - Halodoc](https://www.halodoc.com/kesehatan/autisme?srsltid=AfmBOopQhhu23YT0wFLKaFAwc_1wtL-IOX8RLzqjazB_mKI-Dtodal9N)"
    )

    return hasil_akhir, confidence_str, checklist_str



# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="📷 Upload Gambar Wajah Anak"),
        gr.CheckboxGroup(choices=pertanyaan, label="✅ Checklist Gejala Autisme")
    ],
    outputs=[
        gr.Textbox(label="🧠 Kesimpulan Lengkap"),
        gr.Textbox(label="📊 Confidence Gambar"),
        gr.Textbox(label="📋 Persentase Kuisioner")
    ],
    title="💡 Deteksi Dini Autisme pada Anak",
    description="""
Aplikasi ini membantu mendeteksi **kemungkinan autisme pada anak** melalui:

1. 📷 **Analisis wajah dengan model CNN**
2. 📝 **Kuisioner gejala-gejala umum autisme**

⚠️ **Penting:** Hasil ini bersifat *screening awal* dan **bukan diagnosis resmi**.

👉 Jika hasil menunjukkan potensi autisme, disarankan segera konsultasi ke profesional.  
📞 Konsultasi online: [Halodoc - Psikolog Anak](https://www.halodoc.com/cari-dokter/psikolog-anak)  
📚 Informasi penanganan: [Halodoc - Penanganan Autisme](https://www.halodoc.com/kesehatan/autisme?srsltid=AfmBOopQhhu23YT0wFLKaFAwc_1wtL-IOX8RLzqjazB_mKI-Dtodal9N)
""",
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()