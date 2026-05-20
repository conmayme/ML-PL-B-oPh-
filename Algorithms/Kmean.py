import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import display, Image
import warnings
warnings.filterwarnings('ignore')

X_train = pd.read_csv("X_train_scaled.csv")
X_test  = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv")
y_test  = pd.read_csv("y_test.csv")

TARGET  = "NObeyesdad"
K_RANGE = range(2, 12)
RANDOM  = 42

LABEL_MAP = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight I",
    3: "Overweight II",
    4: "Obesity I",
    5: "Obesity II",
    6: "Obesity III",
}

X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)[TARGET].values

inertias, silhouettes, dbis = [], [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM, n_init=10)
    labels = km.fit_predict(X_all)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_all, labels))
    dbis.append(davies_bouldin_score(X_all, labels))

best_k_sil = list(K_RANGE)[np.argmax(silhouettes)]
best_k_dbi = list(K_RANGE)[np.argmin(dbis)]

K_BEST = best_k_sil

print(f"\n{'K':>3} | {'Inertia':>10} | {'Silhouette':>10} | {'DBI':>8}")
print("-" * 42)
for k, ine, sil, dbi in zip(K_RANGE, inertias, silhouettes, dbis):
    tag = ""
    if k == best_k_sil:
        tag += " ◄ Best Silhouette"
    if k == best_k_dbi:
        tag += " ◄ Best DBI"
    print(f"{k:3d} | {ine:10.1f} | {sil:10.4f} | {dbi:8.4f}{tag}")

print(f"\nBest K by Silhouette : {best_k_sil}")
print(f"Best K by DBI        : {best_k_dbi}")
print(f"Chosen K (= Best Silhouette) : {K_BEST}")

km_final = KMeans(n_clusters=K_BEST, random_state=RANDOM, n_init=10)
cluster_labels = km_final.fit_predict(X_all)

sil_final = silhouette_score(X_all, cluster_labels)
ari_final = adjusted_rand_score(y_all, cluster_labels)
dbi_final = davies_bouldin_score(X_all, cluster_labels)

print(f"\n[K={K_BEST}] Silhouette={sil_final:.4f} | ARI={ari_final:.4f} | "
      f"DBI={dbi_final:.4f} | Inertia={km_final.inertia_:.1f}")

df_result = X_all.copy()
df_result["cluster"]    = cluster_labels
df_result["true_label"] = y_all

print("\nDominant True Label per Cluster:")
for c in range(K_BEST):
    sub = df_result[df_result["cluster"] == c]["true_label"]
    dom = int(sub.mode()[0])
    print(f"  Cluster {c} (n={len(sub):4d}) → [{dom}] {LABEL_MAP.get(dom, dom)}")

df_result.to_csv("kmeans_result_original.csv", index=False)
print("\nSaved → kmeans_result_original.csv")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)

km_dbi = KMeans(n_clusters=best_k_dbi, random_state=RANDOM, n_init=10)
labels_dbi = km_dbi.fit_predict(X_all)

k_list = list(K_RANGE)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("K-Means Clustering — Obesity Dataset", fontsize=15, fontweight="bold")

sc1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_all,
                          cmap="tab10", alpha=0.6, s=15)
axes[0, 0].set_title("Nhãn Gốc (NObeyesdad)")
axes[0, 0].set_xlabel("PC1")
axes[0, 0].set_ylabel("PC2")
plt.colorbar(sc1, ax=axes[0, 0])

# 2. K-means K_BEST (= best Silhouette)
sc2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                          cmap="tab10", alpha=0.6, s=15)
axes[0, 1].set_title(f"K-Means Best Silhouette (K={K_BEST})")
axes[0, 1].set_xlabel("PC1")
axes[0, 1].set_ylabel("PC2")
plt.colorbar(sc2, ax=axes[0, 1])

# 3. K-means best DBI
sc3 = axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbi,
                          cmap="tab10", alpha=0.6, s=15)
axes[0, 2].set_title(f"K-Means Best DBI (K={best_k_dbi})")
axes[0, 2].set_xlabel("PC1")
axes[0, 2].set_ylabel("PC2")
plt.colorbar(sc3, ax=axes[0, 2])

# 4. Elbow (Inertia)
axes[1, 0].plot(k_list, inertias, "bo-", linewidth=2, markersize=7)
axes[1, 0].axvline(x=K_BEST, color="green", linestyle="--", alpha=0.7,
                   label=f"Best Silhouette K={K_BEST}")
axes[1, 0].set_xlabel("Số cụm (K)")
axes[1, 0].set_ylabel("Inertia")
axes[1, 0].set_title("Elbow Method (Inertia)")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 5. Silhouette Score
axes[1, 1].plot(k_list, silhouettes, "go-", linewidth=2, markersize=7)
axes[1, 1].axvline(x=best_k_sil, color="green", linestyle="--", alpha=0.7,
                   label=f"Best K={best_k_sil}")
axes[1, 1].set_xlabel("Số cụm (K)")
axes[1, 1].set_ylabel("Silhouette Score")
axes[1, 1].set_title("Silhouette Score theo số cụm")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# 6. Davies-Bouldin Index
axes[1, 2].plot(k_list, dbis, "ro-", linewidth=2, markersize=7)
axes[1, 2].axvline(x=best_k_dbi, color="red", linestyle="--", alpha=0.7,
                   label=f"Best K={best_k_dbi}")
axes[1, 2].set_xlabel("Số cụm (K)")
axes[1, 2].set_ylabel("Davies-Bouldin Index")
axes[1, 2].set_title("Davies-Bouldin Index theo số cụm")
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("kmeans_visualization.png", dpi=150, bbox_inches="tight")
plt.show()

