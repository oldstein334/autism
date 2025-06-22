# Model Folder

File model AI untuk deteksi autisme harus ditempatkan di folder ini.

## File yang Diperlukan:
- `model_deteksi_autisme.h5` - Model CNN untuk prediksi autisme

## Download Model:
Karena ukuran file model > 100MB (terlalu besar untuk GitHub), silakan download dari:
[LINK DOWNLOAD AKAN DISEDIAKAN]

## Cara Penempatan:
1. Download file `model_deteksi_autisme.h5`
2. Letakkan di folder `model/`
3. Pastikan path lengkap: `model/model_deteksi_autisme.h5`

## Verifikasi:
Setelah menempatkan file model, jalankan aplikasi dengan:
```bash
python app_perpage.py
```

Jika model berhasil dimuat, Anda akan melihat pesan:
"✅ Model berhasil dimuat dari: [path]"
