# Aplikasi Deteksi Autisme

Aplikasi web untuk deteksi autisme menggunakan teknologi machine learning dan Gradio interface.

## Fitur

- **Home**: Halaman utama dengan informasi tentang aplikasi
- **Upload Gambar**: Upload gambar untuk analisis deteksi autisme
- **Kuisioner**: Kuesioner interaktif untuk assessment tambahan
- **Hasil**: Tampilan hasil analisis dan rekomendasi
- **Info**: Informasi lebih lanjut tentang autisme

## Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/oldstein334/autism.git
cd autism
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Download Model File** (Diperlukan):
   - Model file (`model_deteksi_autisme.h5`) tidak disertakan dalam repository karena ukurannya yang besar (>100MB)
   - Silakan download model file dari [link yang akan disediakan] 
   - Letakkan file model di folder `model/model_deteksi_autisme.h5`

4. Jalankan aplikasi:
```bash
python app_perpage.py
```

## Struktur Project

```
autism/
├── app_perpage.py          # File utama aplikasi
├── model/
│   └── model_deteksi_autisme.h5  # Model ML (download terpisah)
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **Gradio**: Framework untuk interface web
- **TensorFlow/Keras**: Machine learning model
- **Pillow**: Image processing

## Cara Penggunaan

1. Buka aplikasi di browser (biasanya http://localhost:7860)
2. Pilih tab yang diinginkan:
   - **Upload**: Upload gambar untuk analisis
   - **Kuisioner**: Isi kuesioner untuk assessment
   - **Hasil**: Lihat hasil analisis
   - **Info**: Baca informasi tentang autisme

## Catatan

- Pastikan model file telah didownload dan ditempatkan di lokasi yang benar
- Aplikasi memerlukan koneksi internet untuk beberapa fitur
- Hasil analisis bersifat informatif dan tidak menggantikan diagnosis medis profesional

## Kontribusi

Silakan buat issue atau pull request untuk perbaikan dan pengembangan aplikasi.

## Lisensi

[Sesuaikan dengan lisensi yang diinginkan]
