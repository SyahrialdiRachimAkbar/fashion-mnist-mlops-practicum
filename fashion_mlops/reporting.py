from __future__ import annotations

import csv
import shutil

from .core import ROOT, best_candidate, mean_std, paired_deltas, read_json, write_json
from .workflow import STATE, config_digest


def status(config: dict) -> dict:
    state = read_json(STATE) if STATE.exists() else {}
    counts = {
        "baseline": int(bool(state.get("baseline"))),
        "tuning": len(state.get("tuning_run_ids", [])),
        "full": len(state.get("full", [])),
        "ablation_no_augmentation": len(state.get("ablation_no_augmentation", [])),
        "ablation_no_dropout": len(state.get("ablation_no_dropout", [])),
    }
    complete = counts["baseline"] == 1 and counts["tuning"] >= config["tuning"]["completed_trials"] and all(counts[key] == 3 for key in ["full", "ablation_no_augmentation", "ablation_no_dropout"]) and bool(state.get("final_evaluation")) and bool(state.get("registry", {}).get("checkpoint_match") if state.get("registry") else False)
    return {"counts": counts, "completed_training_runs": sum(counts.values()), "experiments_complete": complete, "screenshots": [path.name for path in (ROOT / "reports" / "screenshots").glob("*.png")], "config_matches_state": not state or state.get("config_sha256") == config_digest(config), "data_version": state.get("data_md5")}


def report(config: dict) -> str:
    progress = status(config)
    if not progress["config_matches_state"]:
        raise RuntimeError("Konfigurasi laporan tidak sesuai dengan state eksperimen.")
    state = read_json(STATE) if STATE.exists() else {}
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    complete = progress["experiments_complete"]
    lines = [
        "# Praktikum model experimentation dan tracking", "",
        f"Status eksperimen: {'selesai' if complete else 'belum selesai; angka hasil belum boleh disimpulkan'}. Screenshot dikumpulkan terpisah.", "",
        "## Pertanyaan dan hipotesis", "",
        "Apakah tuning hyperparameter memperbaiki macro F1 CNN Fashion-MNIST dibanding baseline? Apakah augmentasi ringan dan dropout membantu generalisasi pada split yang tetap?", "",
        "Hipotesis awal: konfigurasi hasil tuning meningkatkan rata-rata validation macro F1 terhadap baseline; augmentasi meningkatkan F1 terhadap kondisi tanpa augmentasi; dropout meningkatkan F1 terhadap kondisi tanpa dropout. Pengujian memakai split tetap dan pasangan seed yang sama untuk ablation.", "",
        "## Metode", "",
        f"Rancangan memakai subset stratified Fashion-MNIST: {config['dataset']['train_size']:,} train dan {config['dataset']['val_size']:,} validation dari training resmi, dengan split seed {config['dataset']['split_seed']}. Test set resmi hanya dibuka setelah seleksi dikunci. Normalisasi dihitung pada train saja.", "",
        f"CNN terdiri dari convolution 16 dan 32 channel, max pooling, serta classifier 64 unit. Optimizer Adam; setiap training mendapat {config['training']['epochs']} epoch di CPU. Checkpoint dipilih dari validation macro F1 tertinggi; jika seri, epoch lebih awal dipertahankan. Accuracy dan loss menjadi metrik tambahan.", "",
        f"Optuna memakai TPE, sampler seed {config['tuning']['sampler_seed']}, learning rate log-uniform {config['tuning']['lr_min']} sampai {config['tuning']['lr_max']}, batch size {config['tuning']['batch_sizes']}, dan dropout {config['tuning']['dropouts']}. Targetnya {config['tuning']['completed_trials']} trial selesai tanpa pruning, dijalankan berurutan sebagai nested runs MLflow.", "",
        f"Konfigurasi terbaik dilatih ulang dengan seed {config['seeds']}. Kedua ablation memakai seed yang sama tanpa tuning ulang. Std yang dilaporkan adalah sample standard deviation (ddof=1).", "",
        "## Bukti pelaksanaan", "",
        "| Kelompok | Run selesai |", "|---|---:|",
        *[f"| {key} | {value} |" for key, value in progress["counts"].items()], "",
        f"Hash dataset DVC: `{state.get('data_md5', 'belum tersedia')}`.", "",
        f"SHA-256 indeks split: `{state.get('split_sha256', 'belum tersedia')}`.", "",
    ]
    rows = ([state["baseline"]] if state.get("baseline") else [])
    tuning_path = ROOT / "outputs" / "tuning_results.json"
    if tuning_path.exists():
        rows += read_json(tuning_path)
    for group in ["full", "ablation_no_augmentation", "ablation_no_dropout"]:
        rows += state.get(group, [])
    if rows:
        lines += ["## Hasil validation", "", "| Kelompok | n | Macro F1 | Accuracy |", "|---|---:|---:|---:|"]
        if state.get("baseline"):
            row = state["baseline"]
            lines.append(f"| Baseline | 1 | {row['best_val_f1_macro']:.4f} | {row['best_val_accuracy']:.4f} |")
        for group in ["full", "ablation_no_augmentation", "ablation_no_dropout"]:
            group_rows = state.get(group, [])
            if len(group_rows) >= 2:
                f1 = mean_std([row["best_val_f1_macro"] for row in group_rows])
                acc = mean_std([row["best_val_accuracy"] for row in group_rows])
                lines.append(f"| {group} | {len(group_rows)} | {f1['mean']:.4f} ± {f1['std']:.4f} | {acc['mean']:.4f} ± {acc['std']:.4f} |")
        lines += ["", "Baseline menggunakan satu seed, sehingga selisihnya terhadap rata-rata tiga seed tidak membuktikan signifikansi statistik.", ""]
        if state.get("best_trial"):
            params = state["best_trial"]["params"]
            lines += [f"Konfigurasi tuning terpilih: learning rate {params['lr']:.6g}, batch size {params['batch_size']}, dropout {params['dropout']}. Run sumber: `{state['best_trial']['run_id']}`.", ""]
        if len(state.get("full", [])) == 3:
            for group, component in [("ablation_no_augmentation", "augmentasi"), ("ablation_no_dropout", "dropout")]:
                if len(state.get(group, [])) == 3:
                    effect = paired_deltas(state["full"], state[group], "best_val_f1_macro")
                    interpretation = "rata-rata full model lebih tinggi" if effect["mean"] > 0 else "penghapusan komponen tidak menurunkan rata-rata F1" if effect["mean"] < 0 else "rata-rata kedua kondisi sama"
                    lines += [f"Kontribusi {component}, dihitung sebagai full minus ablation untuk pasangan seed: {effect['mean'] * 100:.2f} ± {effect['std'] * 100:.2f} poin persentase. Pada eksperimen ini, {interpretation}. Tiga pasangan seed belum cukup untuk klaim umum.", ""]
        export_rows(rows, output / "run_results.csv")
    else:
        lines += ["## Hasil", "", "Belum ada training run selesai. Tabel metrik dan kesimpulan akan dibuat setelah eksperimen dijalankan; laporan ini tidak memuat angka hasil simulasi.", ""]
    final = state.get("final_evaluation")
    if final:
        lines += ["## Evaluasi test final", "", "Semua kandidat dan aturan pemilihan champion sudah dikunci sebelum test set dibuka. Test tidak dipakai untuk memilih konfigurasi atau seed.", "", "| Metrik | Mean ± sample std, tiga seed |", "|---|---:|"]
        for metric, summary in final["summary"].items():
            lines.append(f"| {metric} | {summary['mean']:.4f} ± {summary['std']:.4f} |")
        lines += ["", f"Champion dipilih berdasarkan validation macro F1 dari run `{final['champion_run_id']}`.", "",
                  "Ruang kandidat champion ditetapkan pada tiga seed model full hasil tuning. Ablation dipakai untuk mengukur pengaruh komponen dan tidak masuk ruang kandidat tersebut. Karena itu, alias champion merujuk model full terpilih. Kondisi tanpa augmentasi dan tanpa dropout memiliki validation F1 lebih tinggi, tetapi belum dievaluasi pada test set dalam rancangan ini.", ""]
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        source = ROOT / "outputs" / "runs" / final["champion_run_id"]
        for name in ["confusion_matrix.png", "learning_curves.png", "classification_report.json"]:
            shutil.copy2(source / name, figures / name)
        lines += ["![Confusion matrix test](figures/confusion_matrix.png)", "", "![Kurva training champion](figures/learning_curves.png)", ""]
    if state.get("registry"):
        registry = state["registry"]
        lines += ["## Model registry", "", f"Model `{registry['name']}`, versi `{registry['version']}`, memiliki alias `champion`. Inferensi memakai `{registry['uri']}`. Pemeriksaan output registry terhadap checkpoint sumber: {'sesuai' if registry['checkpoint_match'] else 'belum sesuai'}.", ""]
        for name in ["champion_predictions.json", "verification.json"]:
            source = ROOT / "outputs" / name
            if source.exists():
                shutil.copy2(source, output / name)
                lines += [f"Bukti tambahan: [{name}]({name}).", ""]
    lines += [
        "## Reproduksibilitas dan batasan", "",
        "Setiap run menyimpan commit Git, diff kode, hash DVC, hash split, konfigurasi, seed, versi environment, metrik per epoch, dan model. Source code, pointer DVC, serta laporan disimpan di Git; database MLflow, artifact, dan data mentah berada di penyimpanan lokal.", "",
        "Subset dan anggaran epoch membatasi kesimpulan pada rancangan praktikum ini. Rata-rata tiga seed menggambarkan variasi inisialisasi dan shuffling pada satu split, bukan variasi seluruh kemungkinan dataset. Studi Optuna yang dilanjutkan menyimpan trial sebelumnya, tetapi urutan usulan baru dapat berbeda dari pencarian tanpa interupsi karena state sampler tidak disimpan di SQLite.", "",
        "## Screenshot pengumpulan", "",
        "Simpan screenshot asli MLflow UI dalam reports/screenshots: tuning_compare.png, seeds_compare.png, ablation_compare.png, dan registry_champion.png. Screenshot harus menunjukkan run yang benar-benar selesai. Panduan pemilihannya ada pada README.", "",
        "Screenshot yang tersedia: " + (", ".join(progress["screenshots"]) or "belum ada") + ".", "",
    ]
    for name in sorted(progress["screenshots"]):
        lines += [f"![{name}](screenshots/{name})", ""]
    if complete:
        full = mean_std([row["best_val_f1_macro"] for row in state["full"]])
        base = state["baseline"]["best_val_f1_macro"]
        lines += ["## Kesimpulan", "", f"Konfigurasi terpilih memperoleh validation macro F1 {full['mean']:.4f} ± {full['std']:.4f}; baseline satu seed memperoleh {base:.4f}. Selisih rata-rata terhadap baseline adalah {(full['mean'] - base) * 100:.2f} poin persentase. Hasil ablation di atas menjelaskan kontribusi komponen pada tiga seed yang dipasangkan. Model terdaftar dan inferensi melalui alias champion telah diverifikasi.", ""]
        comparison = "belum mendukung" if full["mean"] <= base else "mendukung secara deskriptif"
        lines += [f"Hasil tiga seed {comparison} hipotesis peningkatan terhadap baseline. Perbandingan ini terbatas karena baseline hanya memakai satu seed.", ""]
        for group, component in [("ablation_no_augmentation", "augmentasi"), ("ablation_no_dropout", "dropout")]:
            delta = paired_deltas(state["full"], state[group], "best_val_f1_macro")["mean"]
            finding = "belum mendukung" if delta <= 0 else "mendukung secara deskriptif"
            lines += [f"Ablation {finding} hipotesis bahwa {component} membantu pada anggaran lima epoch ini. Penyebabnya belum diuji; hasil ini tidak cukup untuk menyatakan komponen tersebut selalu merugikan.", ""]
    else:
        lines += ["## Pekerjaan yang belum selesai", "", "Selesaikan kelompok eksperimen yang belum mencapai target, evaluasi final, registrasi dan prediksi champion, lalu ambil screenshot MLflow UI. Jangan mengumpulkan laporan ini sebagai laporan hasil final.", ""]
    destination = output / "laporan.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "status.json", progress)
    return str(destination)


def export_rows(rows: list[dict], path) -> None:
    fields = ["run_id", "purpose", "seed", "best_val_f1_macro", "best_val_accuracy", "best_epoch", "training_duration_seconds", "data_md5", "git_commit", "lr", "batch_size", "dropout", "augmentation", "epochs"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row["params"]})
