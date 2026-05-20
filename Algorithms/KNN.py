import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score,accuracy_score,precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

class_names = [
    'Insufficient\nWeight',
    'Normal\nWeight',
    'Overweight\nLevel I',
    'Overweight\nLevel II',
    'Obesity\nType I',
    'Obesity\nType II',
    'Obesity\nType III',
]

custom_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#e377c2', '#8c564b', '#7f7f7f',
]

TARGET = "NObeyesdad"

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label',      fontsize=11)
    ax.set_title(f'Confusion Matrix — {title}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'cm_{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, name)
    return acc, f1

results = {}

# Xử lý với dữ liệu gốc
X_train_origin = pd.read_csv("/content/X_train_scaled.csv")
y_train_origin = pd.read_csv("/content/y_train.csv")
X_test_origin = pd.read_csv("/content/X_test_scaled.csv")
y_test_origin = pd.read_csv("/content/y_test.csv")

model = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance')
model.fit(X_train_origin, y_train_origin.values.ravel())
y_pred_raw = model.predict(X_test_origin)

acc, f1 = evaluate("Dữ liệu gốc (Scaled)", y_test_origin, y_pred_raw)
results["Gốc"] = {"Accuracy": acc, "F1-score": f1}

cv_scores = cross_val_score(
    model,
    X_train_origin,
    y_train_origin.values.ravel(),
    cv=5,
    scoring='f1_weighted'
)

train_score = f1_score(y_train_origin.values.ravel(),
                       model.predict(X_train_origin), average='weighted')
test_score  = f1_score(y_test_origin.values.ravel(),
                       y_pred, average='weighted')

print(f"F1 Train          : {train_score:.4f}")
print(f"F1 Test           : {test_score:.4f}")
print(f"F1 CV (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Chênh lệch Train-Test: {abs(train_score - test_score):.4f}")

# xử lý với dư liệu đã được PCA
X_train_pca = pd.read_csv("/content/X_train_pca.csv")
X_test_pca = pd.read_csv("/content/X_test_pca.csv")

y_train_pca = X_train_pca.pop('NObeyesdad')
y_test_pca = X_test_pca.pop('NObeyesdad')

model_pca = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance')
model_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = model_pca.predict(X_test_pca)

acc, f1 = evaluate("PCA", y_test_pca, y_pred_pca)
results["PCA"] = {"Accuracy": acc, "F1-score": f1}

cv_scores = cross_val_score(
    model,
    X_train_origin,
    y_train_origin.values.ravel(),
    cv=5,
    scoring='f1_weighted'
)

train_score = f1_score(y_train_origin.values.ravel(),
                       model.predict(X_train_origin), average='weighted')
test_score  = f1_score(y_test_origin.values.ravel(),
                       y_pred, average='weighted')

print(f"F1 Train          : {train_score:.4f}")
print(f"F1 Test           : {test_score:.4f}")
print(f"F1 CV (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Chênh lệch Train-Test: {abs(train_score - test_score):.4f}")

# xử lý với dư liệu đã được LDA
X_train_lda = pd.read_csv("/content/X_train_lda_3.csv")
X_test_lda = pd.read_csv("/content/X_test_lda_3.csv")

y_train_lda = X_train_lda.pop('NObeyesdad')
y_test_lda = X_test_lda.pop('NObeyesdad')

model_lda = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance')
model_lda.fit(X_train_lda, y_train_lda)
y_pred_lda = model_lda.predict(X_test_lda)

acc, f1 = evaluate("LDA", y_test_lda, y_pred_lda)
results["LDA"] = {"Accuracy": acc, "F1-score": f1}

X_train_lda = pd.read_csv("/content/X_train_lda_5.csv")
X_test_lda  = pd.read_csv("/content/X_test_lda_5.csv")

y_train_lda = X_train_lda.pop('NObeyesdad')
y_test_lda  = X_test_lda.pop('NObeyesdad')

model_lda = SVC(kernel="linear", C=1, class_weight="balanced", random_state=42)
model_lda.fit(X_train_lda, y_train_lda)
y_pred_lda = model_lda.predict(X_test_lda)

acc, f1 = evaluate("LDA", y_test_lda, y_pred_lda)
results["LDA"] = {"Accuracy": acc, "F1-score": f1}

cv_scores = cross_val_score(
    model_lda,
    X_train_lda,
    y_train_lda,
    cv=5,
    scoring='f1_weighted'
)

train_score = f1_score(y_train_lda,
                       model_lda.predict(X_train_lda), average='weighted')
test_score  = f1_score(y_test_lda,
                       y_pred_lda, average='weighted')

print(f"F1 Train          : {train_score:.4f}")
print(f"F1 Test           : {test_score:.4f}")
print(f"F1 CV (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Chênh lệch Train-Test: {abs(train_score - test_score):.4f}")