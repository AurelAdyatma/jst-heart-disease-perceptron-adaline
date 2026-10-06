# coding: utf-8
"""
TB1_PrediksiPenyakitJantung.py
Eksperimen Perceptron & ADALINE untuk prediksi penyakit jantung.
Jalankan: python TB1_PrediksiPenyakitJantung.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  TUGAS BESAR 1 - JST IT6053")
print("  Prediksi Penyakit Jantung (UCI Heart Disease Cleveland)")
print("  Perceptron & ADALINE")
print("=" * 65)

# ============================================================
# 1. MEMUAT DATA
# ============================================================
print("\n[1] MEMUAT DATASET")
df = pd.read_csv("dataset.csv")
print(f"  Dimensi   : {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"  Fitur (X) : {list(df.columns[:-1])}")
print(f"  Target (y): 'target' -> 0=tidak sakit, 1=sakit")
vc = df["target"].value_counts()
print(f"\n  Distribusi kelas:")
print(f"    Kelas 0 (tidak sakit) : {vc[0]} ({vc[0]/len(df)*100:.1f}%)")
print(f"    Kelas 1 (sakit)       : {vc[1]} ({vc[1]/len(df)*100:.1f}%)")
print(f"\n  Missing value: {df.isnull().sum().sum()}")
print(f"  Duplikat    : {df.duplicated().sum()}")

# Label -1/+1  (+1 = terindikasi penyakit jantung = POSITIF)
X = df.drop("target", axis=1).values.astype(float)
y_raw = df["target"].values
y = np.where(y_raw == 1, 1, -1)
FEATURE_NAMES = list(df.columns[:-1])

# ============================================================
# 2. SPLIT DATA (80:20, stratified)
# ============================================================
print("\n[2] PEMBAGIAN DATA (80:20, stratified, seed=42)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
print(f"  Data latih : {X_train.shape[0]} ({X_train.shape[0]/len(y)*100:.1f}%)")
print(f"  Data uji   : {X_test.shape[0]}  ({X_test.shape[0]/len(y)*100:.1f}%)")
print(f"  Latih +1/-1: {(y_train==1).sum()}/{(y_train==-1).sum()}")
print(f"  Uji   +1/-1: {(y_test==1).sum()}/{(y_test==-1).sum()}")

# Standardisasi - fit HANYA dari data latih
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)
X_train_raw = X_train.copy()
X_test_raw  = X_test.copy()
print(f"\n  Mean latih (3 fitur pertama) : {np.round(scaler.mean_[:3], 4)}")
print(f"  Std latih  (3 fitur pertama) : {np.round(scaler.scale_[:3], 4)}")

# ============================================================
# 3. IMPLEMENTASI PERCEPTRON (NumPy mandiri)
# ============================================================
class Perceptron:
    """
    Perceptron (Rosenblatt, 1958).
    z = w'x + b
    y_hat = +1 jika z >= 0, -1 jika z < 0
    Delta = eta * (y - y_hat)
    w <- w + Delta * x
    b <- b + Delta
    """
    def __init__(self, eta=0.01, n_epochs=50, random_state=42):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        self.errors_ = []

        for _ in range(self.n_epochs):
            n_errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.w_) + self.b_
                y_hat = 1 if z >= 0 else -1
                delta = self.eta * (yi - y_hat)
                if delta != 0:
                    self.w_ += delta * xi
                    self.b_ += delta
                    n_errors += 1
            self.errors_.append(n_errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


# ============================================================
# 4. IMPLEMENTASI ADALINE (NumPy mandiri, Batch GD)
# ============================================================
class Adaline:
    """
    ADALINE - Batch Gradient Descent (Widrow & Hoff, 1960).
    e = y - z  (error dari output LINEAR, bukan threshold)
    w <- w + eta * X'.e / n
    b <- b + eta * mean(e)
    J = (1/2n) sum((y - z)^2)
    Threshold hanya saat PREDIKSI.
    """
    def __init__(self, eta=0.01, n_epochs=50, random_state=42):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        self.losses_ = []
        self.diverged_ = False
        self.diverge_epoch_ = None
        n = X.shape[0]

        for epoch in range(self.n_epochs):
            z = self.net_input(X)
            e = y - z
            self.w_ += self.eta * X.T.dot(e) / n
            self.b_ += self.eta * e.mean()
            loss = 0.5 * np.mean(e ** 2)
            self.losses_.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                self.diverged_ = True
                self.diverge_epoch_ = epoch + 1
                print(f"      [ADALINE eta={self.eta}] Divergen pada epoch {epoch+1}!")
                break
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


# ============================================================
# 5. FUNGSI EVALUASI
# ============================================================
def evaluate(model, Xtr, ytr, Xte, yte):
    yp_tr = model.predict(Xtr)
    yp_te = model.predict(Xte)
    return {
        "acc_train": accuracy_score(ytr, yp_tr),
        "acc_test":  accuracy_score(yte, yp_te),
        "precision": precision_score(yte, yp_te, pos_label=1, zero_division=0),
        "recall":    recall_score(yte, yp_te, pos_label=1, zero_division=0),
        "f1":        f1_score(yte, yp_te, pos_label=1, zero_division=0),
        "cm":        confusion_matrix(yte, yp_te, labels=[-1, 1]),
    }


# ============================================================
# 6. PERHITUNGAN MANUAL (2 langkah / iterasi)
# ============================================================
print("\n" + "=" * 65)
print("[3] PERHITUNGAN MANUAL")

eta_man = 0.01

# --- PERCEPTRON ---
print("\n  --- Perceptron: 2 langkah pembaruan (eta=0.01) ---")
wm = np.zeros(X_train_std.shape[1])
bm = 0.0
print(f"  Inisialisasi: w=0 (13-dim), b=0")

for step, (xi, yi) in enumerate(zip(X_train_std[:2], y_train[:2]), 1):
    z    = np.dot(wm, xi) + bm
    yhat = 1 if z >= 0 else -1
    delta = eta_man * (yi - yhat)
    print(f"\n  Langkah {step}: x[{step-1}], y={yi}")
    print(f"    z = w'x + b = {z:.4f}")
    print(f"    y_hat = {yhat}, Delta = {eta_man}*({yi}-{yhat}) = {delta:.4f}")
    if delta != 0:
        wm += delta * xi
        bm += delta
        print(f"    Pembaruan: w <- w + Delta*x, b <- b + Delta")
    else:
        print(f"    Prediksi benar, tidak ada pembaruan")
    print(f"    w_baru[0:3] = {np.round(wm[:3], 4)}, b_baru = {bm:.4f}")

# --- ADALINE ---
print(f"\n  --- ADALINE: 2 iterasi batch (eta=0.01, 5 sampel) ---")
X5 = X_train_std[:5]
y5 = y_train[:5]
wa = np.zeros(X5.shape[1])
ba = 0.0
n5 = 5
print(f"  Inisialisasi: w=0 (13-dim), b=0")

for it in range(1, 3):
    z5   = X5.dot(wa) + ba
    e5   = y5 - z5
    J5   = 0.5 * np.mean(e5**2)
    gw5  = X5.T.dot(e5) / n5
    gb5  = e5.mean()
    wa  += eta_man * gw5
    ba  += eta_man * gb5
    J5_after = 0.5 * np.mean((y5 - (X5.dot(wa) + ba))**2)
    print(f"\n  Iterasi Batch {it}:")
    print(f"    J sebelum update = {J5:.6f}")
    print(f"    e = {np.round(e5, 4)}")
    print(f"    dJ/dw[0:3]= {np.round(gw5[:3], 6)}, dJ/db = {gb5:.6f}")
    print(f"    w[0:3] baru= {np.round(wa[:3], 6)}, b baru = {ba:.6f}")
    print(f"    J setelah update = {J5_after:.6f}")


# ============================================================
# 7. EKSPERIMEN (8 konfigurasi)
# ============================================================
print("\n" + "=" * 65)
print("[4] EKSPERIMEN (8 pelatihan)")

configs = [
    ("Perceptron", 0.001, True),
    ("Perceptron", 0.01,  True),
    ("Perceptron", 0.1,   True),
    ("Perceptron", 0.01,  False),
    ("ADALINE",    0.001, True),
    ("ADALINE",    0.01,  True),
    ("ADALINE",    0.1,   True),
    ("ADALINE",    0.01,  False),
]
N_EPOCHS = 50
results = []

for i, (mname, eta, use_std) in enumerate(configs):
    Xtr = X_train_std if use_std else X_train_raw
    Xte = X_test_std  if use_std else X_test_raw

    if mname == "Perceptron":
        model = Perceptron(eta=eta, n_epochs=N_EPOCHS, random_state=RANDOM_SEED)
    else:
        model = Adaline(eta=eta, n_epochs=N_EPOCHS, random_state=RANDOM_SEED)

    model.fit(Xtr, y_train)
    m = evaluate(model, Xtr, y_train, Xte, y_test)

    if mname == "Perceptron":
        curve    = model.errors_
        final_e  = float(curve[-1])
        diverged = False
    else:
        curve    = model.losses_
        diverged = model.diverged_
        final_e  = float('nan') if diverged else float(curve[-1])

    res = dict(
        no=i+1, model=mname, eta=eta, epochs=N_EPOCHS,
        std="Ya" if use_std else "Tidak",
        acc_train=m["acc_train"], acc_test=m["acc_test"],
        precision=m["precision"], recall=m["recall"], f1=m["f1"],
        cm=m["cm"], curve=curve, final_e=final_e,
        w_final=model.w_.copy(), b_final=model.b_,
        diverged=diverged,
    )
    results.append(res)

    dv = " [DIVERGEN]" if diverged else ""
    std_s = "Ya" if use_std else "Tidak"
    print(f"\n  [{i+1}] {mname} | eta={eta} | std={std_s}{dv}")
    print(f"      AccLatih={m['acc_train']:.4f} | AccUji={m['acc_test']:.4f}")
    print(f"      P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    if not np.isnan(final_e):
        print(f"      Loss/Error akhir: {final_e:.4f}")
    else:
        print(f"      Loss/Error akhir: NaN (divergen)")
    print(f"      w[0:3]={np.round(res['w_final'][:3],4)}, b={res['b_final']:.4f}")


# ============================================================
# 8. TABEL RINGKASAN
# ============================================================
print("\n" + "=" * 65)
print("[5] TABEL RINGKASAN KONFIGURASI")
hdr = f"{'No':>3} {'Model':>10} {'eta':>6} {'Std':>5} {'AccTrain':>10} {'AccTest':>9} {'Loss/Err':>10} {'P':>6} {'R':>6} {'F1':>6}"
print(f"\n{hdr}")
print("-" * 75)
for r in results:
    es = f"{r['final_e']:.4f}" if not np.isnan(r['final_e']) else "   NaN"
    print(f"{r['no']:>3} {r['model']:>10} {r['eta']:>6} {r['std']:>5} "
          f"{r['acc_train']:>10.4f} {r['acc_test']:>9.4f} {es:>10} "
          f"{r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f}")


# ============================================================
# 9. ANALISIS EPOCH 10, 30, 50
# ============================================================
print("\n" + "=" * 65)
print("[6] KONDISI PADA EPOCH 10, 30, 50")
print(f"\n  {'Model':>10} {'eta':>6} {'Std':>5} | {'Epoch10':>10} {'Epoch30':>10} {'Epoch50':>10}")
print("  " + "-" * 58)
for r in results:
    c = r["curve"]
    def gv(idx): return f"{c[idx]:.4f}" if len(c) > idx else "N/A"
    print(f"  {r['model']:>10} {r['eta']:>6} {r['std']:>5} | {gv(9):>10} {gv(29):>10} {gv(49):>10}")


# ============================================================
# 10. BOBOT DAN BIAS AKHIR
# ============================================================
print("\n" + "=" * 65)
print("[7] BOBOT DAN BIAS AKHIR")
for r in results:
    print(f"\n  [{r['no']}] {r['model']} eta={r['eta']} std={r['std']}")
    print(f"      b_akhir = {r['b_final']:.6f}")
    for fn, wv in zip(FEATURE_NAMES, r["w_final"]):
        print(f"      w[{fn:>8}] = {wv:>10.6f}")


# ============================================================
# 11. VISUALISASI
# ============================================================
print("\n" + "=" * 65)
print("[8] MEMBUAT VISUALISASI...")

plt.rcParams['font.family'] = 'DejaVu Sans'

# 11.1 Kurva Perceptron
perc_r = [r for r in results if r["model"] == "Perceptron"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Perceptron - Jumlah Misklasifikasi per Epoch", fontsize=13, fontweight='bold')
for ax, r in zip(axes, perc_r):
    ax.plot(range(1, len(r["curve"])+1), r["curve"], 'b-o', ms=3, lw=1.5)
    for ep in [10, 30, 50]:
        if ep <= len(r["curve"]):
            ax.axvline(ep, color='red', ls='--', alpha=0.5, lw=1)
            ax.text(ep, r["curve"][ep-1], f"E{ep}\n{r['curve'][ep-1]}",
                    ha='center', va='bottom', fontsize=7, color='red')
    sl = "std" if r["std"]=="Ya" else "no-std"
    ax.set_title(f"eta={r['eta']}, {sl}\nAcc uji={r['acc_test']:.3f}", fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("# Misklasifikasi"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_perceptron_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Tersimpan: plot_perceptron_curves.png")

# 11.2 Kurva ADALINE
ada_r = [r for r in results if r["model"] == "ADALINE"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("ADALINE - Loss J per Epoch", fontsize=13, fontweight='bold')
for ax, r in zip(axes, ada_r):
    col = 'red' if r["diverged"] else 'darkorange'
    ax.plot(range(1, len(r["curve"])+1), r["curve"], color=col, lw=1.5)
    if r["diverged"]:
        ax.set_title(f"eta={r['eta']} - DIVERGEN\n(epoch {len(r['curve'])})", fontsize=9, color='red')
    else:
        for ep in [10, 30, 50]:
            if ep <= len(r["curve"]):
                ax.axvline(ep, color='blue', ls='--', alpha=0.5, lw=1)
        sl = "std" if r["std"]=="Ya" else "no-std"
        ax.set_title(f"eta={r['eta']}, {sl}\nAcc uji={r['acc_test']:.3f}", fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss J"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_adaline_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Tersimpan: plot_adaline_curves.png")

# 11.3 Confusion Matrix
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Confusion Matrix pada Data Uji  (-1=tidak sakit, +1=sakit)", fontsize=12, fontweight='bold')
for ax, r in zip(axes.flatten(), results):
    cm = r["cm"]
    ax.imshow(cm, cmap='Blues')
    for (row, col_), val in np.ndenumerate(cm):
        ax.text(col_, row, str(val), ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if val > cm.max()/2 else 'black')
    sl = "std" if r["std"]=="Ya" else "no-std"
    ax.set_title(f"[{r['no']}] {r['model']} eta={r['eta']} {sl}\nAcc={r['acc_test']:.3f} F1={r['f1']:.3f}", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred-1", "Pred+1"], fontsize=8)
    ax.set_yticklabels(["Act-1", "Act+1"], fontsize=8)
plt.tight_layout()
plt.savefig("plot_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Tersimpan: plot_confusion_matrix.png")

# 11.4 Perbandingan Akurasi
fig, ax = plt.subplots(figsize=(13, 5))
lbls = [f"[{r['no']}] {r['model']}\neta={r['eta']} {'std' if r['std']=='Ya' else 'no-std'}" for r in results]
x = np.arange(len(results)); w = 0.35
b1 = ax.bar(x-w/2, [r['acc_train'] for r in results], w, label='Akurasi Latih', color='steelblue')
b2 = ax.bar(x+w/2, [r['acc_test']  for r in results], w, label='Akurasi Uji',   color='coral')
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f"{b.get_height():.3f}", ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=7.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("Akurasi")
ax.set_title("Perbandingan Akurasi Latih vs Uji - Seluruh Konfigurasi", fontsize=12, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("plot_accuracy_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Tersimpan: plot_accuracy_comparison.png")

# 11.5 Decision Boundary (PCA 2 komponen)
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_tr_pca = pca.fit_transform(X_train_std)
X_te_pca = pca.transform(X_test_std)
bp = Perceptron(eta=0.01, n_epochs=50, random_state=RANDOM_SEED)
bp.fit(X_tr_pca, y_train)
xx, yy = np.meshgrid(
    np.linspace(X_tr_pca[:,0].min()-1, X_tr_pca[:,0].max()+1, 200),
    np.linspace(X_tr_pca[:,1].min()-1, X_tr_pca[:,1].max()+1, 200)
)
Z = bp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (Xi, yi, title) in zip(axes, [
    (X_tr_pca, y_train, "Data Latih"),
    (X_te_pca, y_test,  "Data Uji"),
]):
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu')
    sc = ax.scatter(Xi[:,0], Xi[:,1], c=yi, cmap='RdBu',
                    edgecolors='k', linewidths=0.4, s=40)
    ax.set_xlabel("PCA Komponen 1"); ax.set_ylabel("PCA Komponen 2")
    ax.set_title(f"Decision Boundary - Perceptron (eta=0.01, std)\n{title}", fontsize=9)
    ax.legend(*sc.legend_elements(), title="Kelas"); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot_decision_boundary.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Tersimpan: plot_decision_boundary.png")

print("\n" + "=" * 65)
print("  SELESAI! Semua file output tersimpan.")
print("=" * 65)
