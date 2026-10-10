# Praktikum MLflow: CNN Fashion-MNIST

Implementasi tugas pada slide 88 `slide_presentation.pdf`. Pipeline mencakup baseline CNN, 20 trial Optuna dengan nested runs, tiga seed, dua ablation, DVC, dan prediksi melalui alias model registry.

Hasil eksperimen berada pada [laporan](reports/laporan.md). Periksa status di awal laporan sebelum mengumpulkannya. Laporan yang belum selesai tidak berisi metrik hasil buatan.

## Menyiapkan proyek di Windows

Gunakan Python 64-bit versi 3.12 sampai 3.14, Git, dan koneksi internet untuk setup. Di lingkungan ini Python yang tersedia adalah 3.14.2; `.venv` memakai interpreter tersebut. Setup memasang paket ke environment proyek, bukan ke Python global.

```powershell
cd 'C:\Akbar\MLops Tugas'
.\scripts\setup.ps1
```

Jika PowerShell memblokir script lokal, jalankan script yang sudah kamu periksa dengan:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Setup memasang `requirements.txt`, mengunduh Fashion-MNIST, menyimpan split tetap, menjalankan `dvc add data`, dan membuat remote DVC lokal di `storage/dvc`. Data mentah dan split tidak masuk Git. DVC memverifikasi dataset terhadap checksum sumber saat torchvision mengunduhnya.

Setelah persiapan data, simpan kode dan pointer dataset dalam commit sebelum training:

```powershell
git status
git add .
git commit -m "Implement Fashion-MNIST experiment tracking practicum"
```

Gunakan identitas Git milikmu jika Git meminta nama/email. Tidak ada perintah push dalam setup atau pipeline.

## Memeriksa dan menjalankan pipeline

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Pada terminal pertama, jalankan server. Metadata dan model registry memakai SQLite; file artifact disimpan melalui server ke folder lokal.

```powershell
.\scripts\start_mlflow.ps1
```

Pada terminal kedua, jalankan pemeriksaan singkat. Ini memakai 128 train dan 64 validation selama satu epoch, pada experiment terpisah; run ini bukan bagian dari 20 trial atau hasil praktikum.

```powershell
.\.venv\Scripts\python.exe -m fashion_mlops smoke
```

Lalu jalankan seluruh eksperimen:

```powershell
.\scripts\run_practicum.ps1
```

Alternatif satu terminal: `scripts/run_practicum.ps1 -StartServer` menyalakan MLflow di background dengan log di `outputs/`, lalu menjalankan pipeline. Server tetap hidup agar UI bisa digunakan setelah training selesai. Jangan memakai opsi ini jika server sudah berjalan pada port 5000.

Pipeline berurutan: baseline → tune → seeds → ablation → evaluate → register → predict → report. Total minimum 30 training run: 1 baseline, 20 tuning, 3 full model, dan 6 ablation. Parent runs serta smoke tidak dihitung.

Durasi bergantung pada CPU dan beban laptop. Gunakan waktu baseline sebagai perkiraan awal; jangan mengubah data atau epoch di tengah study untuk mempercepat run yang belum selesai.

## Menjalankan tahap secara terpisah

Semua perintah memakai prefix `.\.venv\Scripts\python.exe -m fashion_mlops`:

| Perintah | Perilaku |
|---|---|
| `doctor` | Memeriksa interpreter dan paket yang dibutuhkan; exit code 1 jika ada yang kurang |
| `prepare-data` | Mengunduh, memilih subset, menyimpan indeks, menghitung normalisasi train, dan memversikan data |
| `smoke` | Menguji satu epoch kecil pada experiment terpisah |
| `baseline` | Melatih baseline; mengembalikan run tersimpan jika sudah selesai |
| `tune` | Menjalankan sampai 20 trial COMPLETE dan run MLflow FINISHED |
| `seeds` | Melatih konfigurasi terbaik dari awal untuk seed 42, 123, dan 2026 |
| `ablation` | Menghapus augmentasi atau dropout, masing-masing dengan tiga seed yang sama |
| `evaluate` | Mengunci kandidat, membuka test set, dan mengevaluasi tiga full model |
| `register` | Mendaftarkan champion dan membandingkan output registry dengan checkpoint |
| `predict` | Memuat `models:/fashion-mnist-cnn@champion` dan memprediksi sepuluh contoh validation |
| `report` | Menulis laporan dan CSV dari hasil nyata; menandai laporan parsial |
| `status` | Menampilkan jumlah run selesai serta status evaluasi/registry |
| `all` | Menjalankan semua tahap setelah data siap dan server hidup |

Konfigurasi berada di `configs/default.yaml`. Argumen `--config` menerima file YAML alternatif. Tentukan konfigurasi sebelum training pertama; hash konfigurasi dan data harus tetap sama saat melanjutkan pipeline. Satu proses pipeline diperbolehkan pada satu waktu.

## Desain eksperimen

Fashion-MNIST dipilih sebagai dataset citra kecil yang dapat dilatih di CPU. Subset terdiri dari 12.000 train dan 3.000 validation yang seimbang per kelas, dipilih dari 60.000 training resmi. Sebanyak 10.000 test resmi dikunci untuk tahap akhir. Indeks split disimpan, terpisah dari seed training. Normalisasi hanya dihitung dari subset train.

CNN memakai dua convolution, 16 dan 32 channel, masing-masing diikuti ReLU dan pooling. Classifier memakai 64 unit, dropout, dan 10 logit kelas. Baseline: Adam, learning rate 0.001, batch size 64, dropout 0.3, lima epoch. Augmentasi berupa random crop 28×28 dengan padding 2 serta horizontal flip. Model full selalu memiliki augmentasi dan dropout nonzero agar kedua ablation memiliki komponen yang benar-benar dapat dihapus.

Metrik seleksi adalah validation macro F1, bukan accuracy atau test F1. Checkpoint memakai epoch dengan validation F1 tertinggi. Optuna mencari learning rate 0.0001–0.003 pada skala log, batch size 64/128, dan dropout 0.2/0.35/0.5. Setiap trial mendapat lima epoch, seed training 42, dan split yang sama. Trial berjalan satu per satu tanpa pruning.

Konfigurasi terbaik dilatih ulang menggunakan tiga seed. Ablation hanya mengubah augmentasi atau dropout; komponen lain dan seed tetap. Selisih ablation dihitung berpasangan: F1 full minus F1 ablation pada seed yang sama. Mean ± std menggunakan sample standard deviation (ddof=1).

Seleksi champion dari tiga full model memakai validation F1, dengan run ID sebagai pemecah seri. Pilihan disimpan sebelum test dibuka. Ketiga full model dievaluasi pada satu tahap final. Test tidak digunakan untuk tuning, memilih seed, atau mengulang keputusan. Hasil evaluasi yang sudah tersimpan dipakai ulang saat pipeline dilanjutkan; training baru ditolak setelah seleksi final dikunci.

## Melanjutkan setelah interupsi

Jalankan kembali `all` dengan konfigurasi dan data yang sama. Baseline dan seed yang selesai dilewati. SQLite Optuna menyimpan trial; hanya trial COMPLETE yang dihitung menuju target 20. Trial RUNNING yang tertinggal ditandai gagal pada proses lanjutan. Run gagal tetap tersimpan sebagai bukti, bukan dihapus.

Usulan Optuna setelah resume dapat berbeda dari pencarian tanpa interupsi karena SQLite tidak menyimpan state RNG sampler. Parameter setiap trial tersimpan sehingga masing-masing training tetap bisa diulang. Setelah tahap test dikunci, jangan menyesuaikan model berdasarkan hasil test.

## Mengambil screenshot MLflow

Buka `http://127.0.0.1:5000`, pilih experiment `fashion-mnist-cnn`, dan simpan screenshot asli di `reports/screenshots/`.

1. `tuning_compare.png`: buka child runs pada parent `optuna-sweep`, pilih beberapa trial termasuk yang terbaik, lalu buka **Compare**. Tampilkan learning rate, batch size, dropout, dan `best_val_f1_macro`.
2. `seeds_compare.png`: filter tag `purpose = full`, pilih tiga run, dan tampilkan seed serta validation/test F1.
3. `ablation_compare.png`: pilih full dan kedua ablation dengan seed 42 yang sama; tampilkan `augmentation`, `dropout`, dan validation F1. Rata-rata tiga seed ada dalam laporan.
4. `registry_champion.png`: buka model `fashion-mnist-cnn` di model registry; tampilkan version, alias `champion`, dan source run.

Screenshot tidak dibuat dari grafik buatan yang menyerupai UI. Jalankan kembali `report` setelah menambahkan screenshot agar daftar bukti ikut diperbarui. Nama dan posisi menu dapat berbeda menurut versi MLflow yang terpasang.

## Pengumpulan dan reproduksi

Kumpulkan repositori kode, konfigurasi, dependensi, pointer DVC, laporan, CSV hasil, dan screenshot. `outputs`, data mentah, cache, SQLite, dan artifact MLflow diabaikan oleh Git. Screenshot dan figure laporan juga diabaikan secara default karena merupakan artifact; tambahkan bukti yang sudah final secara eksplisit:

```powershell
git add -f reports/screenshots/*.png reports/figures/*
git add reports/laporan.md reports/run_results.csv reports/status.json
git commit -m "Add practicum results and MLflow evidence"
```

Setiap run menyimpan versi paket aktual sebagai artifact `context/requirements-lock.txt`. Untuk mengulang environment persis, ambil file itu dari run sumber dan gunakan interpreter dengan versi yang sama. Commit kode, hash DVC, hash indeks split, seed, serta diff yang belum di-commit juga dicatat pada run.

Remote DVC lokal berfungsi sebagai salinan pada laptop ini; remote itu tidak dapat diakses otomatis oleh dosen dari mesin lain. Untuk reproduksi data tanpa remote bersama, jalankan `prepare-data` dari konfigurasi yang sama dan cocokkan hash dengan pointer/manifest sumber. Jika pengumpulan memerlukan data melalui `dvc pull` pada mesin lain, gunakan remote bersama yang kamu sediakan sebelum pengumpulan.

## Referensi

- Tugas dan prinsip eksperimen: slide 19–23, 28, 31, 44, 47–48, dan 88 pada PDF materi.
- [MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [MLflow untuk PyTorch dan model registry](https://mlflow.org/docs/latest/ml/deep-learning/pytorch)
- [Optuna: SQLite dan resume](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html)
- [DVC: data versioning](https://doc.dvc.org/start)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
