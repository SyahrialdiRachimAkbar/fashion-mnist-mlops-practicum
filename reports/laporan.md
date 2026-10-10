# Praktikum model experimentation dan tracking

Status eksperimen: belum selesai; angka hasil belum boleh disimpulkan. Screenshot dikumpulkan terpisah.

## Pertanyaan dan hipotesis

Apakah tuning hyperparameter memperbaiki macro F1 CNN Fashion-MNIST dibanding baseline? Apakah augmentasi ringan dan dropout membantu generalisasi pada split yang tetap?

Hipotesis diuji melalui baseline, tuning, pengulangan tiga seed, dan penghapusan komponen satu per satu. Hipotesis yang tidak didukung hasil tetap dilaporkan.

## Metode

Rancangan memakai subset stratified Fashion-MNIST: 12,000 train dan 3,000 validation dari training resmi, dengan split seed 42. Test set resmi hanya dibuka setelah seleksi dikunci. Normalisasi dihitung pada train saja.

CNN terdiri dari convolution 16 dan 32 channel, max pooling, serta classifier 64 unit. Optimizer Adam; setiap training mendapat 5 epoch di CPU. Checkpoint dipilih dari validation macro F1 tertinggi; jika seri, epoch lebih awal dipertahankan. Accuracy dan loss menjadi metrik tambahan.

Optuna memakai TPE, sampler seed 42, learning rate log-uniform 0.0001 sampai 0.003, batch size [64, 128], dan dropout [0.2, 0.35, 0.5]. Targetnya 20 trial selesai tanpa pruning, dijalankan berurutan sebagai nested runs MLflow.

Konfigurasi terbaik dilatih ulang dengan seed [42, 123, 2026]. Kedua ablation memakai seed yang sama tanpa tuning ulang. Std yang dilaporkan adalah sample standard deviation (ddof=1).

## Bukti pelaksanaan

| Kelompok | Run selesai |
|---|---:|
| baseline | 0 |
| tuning | 0 |
| full | 0 |
| ablation_no_augmentation | 0 |
| ablation_no_dropout | 0 |

Hash dataset DVC: `belum tersedia`.

SHA-256 indeks split: `belum tersedia`.

## Hasil

Belum ada training run selesai. Tabel metrik dan kesimpulan akan dibuat setelah eksperimen dijalankan; laporan ini tidak memuat angka hasil simulasi.

## Reproduksibilitas dan batasan

Setiap run menyimpan commit Git, diff kode, hash DVC, hash split, konfigurasi, seed, versi environment, metrik per epoch, dan model. Source code, pointer DVC, serta laporan disimpan di Git; database MLflow, artifact, dan data mentah berada di penyimpanan lokal.

Subset dan anggaran epoch membatasi kesimpulan pada rancangan praktikum ini. Rata-rata tiga seed menggambarkan variasi inisialisasi dan shuffling pada satu split, bukan variasi seluruh kemungkinan dataset. Studi Optuna yang dilanjutkan menyimpan trial sebelumnya, tetapi urutan usulan baru dapat berbeda dari pencarian tanpa interupsi karena state sampler tidak disimpan di SQLite.

## Screenshot pengumpulan

Simpan screenshot asli MLflow UI dalam reports/screenshots: tuning_compare.png, seeds_compare.png, ablation_compare.png, dan registry_champion.png. Screenshot harus menunjukkan run yang benar-benar selesai. Panduan pemilihannya ada pada README.

Screenshot yang tersedia: belum ada.

## Pekerjaan yang belum selesai

Selesaikan kelompok eksperimen yang belum mencapai target, evaluasi final, registrasi dan prediksi champion, lalu ambil screenshot MLflow UI. Jangan mengumpulkan laporan ini sebagai laporan hasil final.
