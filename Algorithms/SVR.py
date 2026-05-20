import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

y_train = pd.read_csv("/content/y_train.csv")
y_test = pd.read_csv("/content/y_test.csv")
X_train_orig = pd.read_csv("/content/X_train_scaled.csv")
X_test_orig  = pd.read_csv("/content/X_test_scaled.csv")
X_train_pca = pd.read_csv("/content/X_train_pca.csv")
X_test_pca = pd.read_csv("/content/X_test_pca.csv")
X_train_lda = pd.read_csv("/content/X_train_lda_5.csv")
X_test_lda = pd.read_csv("/content/X_test_lda_5.csv")


for df in [X_train_pca, X_test_pca, X_train_lda, X_test_lda]:
    if 'NObeyesdad' in df.columns:
        df.drop('NObeyesdad', axis=1, inplace=True)

def svm_regression_on_decision_values(X_train, X_test, y_train, y_test, svm_model, dataset_name):

    y_train_1d = y_train.values.ravel()
    y_test_1d = y_test.values.ravel()

    df_train = svm_model.decision_function(X_train)
    df_test = svm_model.decision_function(X_test)

    print("\nShape decision_function:", df_train.shape)

    if df_train.ndim == 2:

        print("Áp dụng PCA cho decision_function...")

        pca = PCA(n_components=1)

        y_train_score = pca.fit_transform(df_train).ravel()
        y_test_score = pca.transform(df_test).ravel()

    else:

        y_train_score = df_train
        y_test_score = df_test

    svr = SVR(kernel='rbf', C=10, epsilon=0.1)

    svr.fit(X_train, y_train_score)

    y_pred = svr.predict(X_test)

    mse = mean_squared_error(y_test_score, y_pred)
    r2 = r2_score(y_test_score, y_pred)

    print(f"\n[Hồi quy SVM] - {dataset_name}")
    print(f"MSE : {mse:.4f}")
    print(f"R²  : {r2:.4f}")

    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        x=y_test_score,
        y=y_pred,
        hue=y_test_1d,
        palette='tab10',
        alpha=0.7,
        edgecolor='k'
    )

    min_val = min(y_test_score.min(), y_pred.min())
    max_val = max(y_test_score.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle='--',
        color='red',
        label='Ideal (y = x)'
    )

    plt.xlim(min_val - 1, max_val + 1)
    plt.ylim(min_val - 1, max_val + 1)

    plt.xlabel("Decision Function thực tế (PCA Reduced)")
    plt.ylabel("Dự đoán SVR")

    plt.title(f"Hồi quy SVM – {dataset_name}")

    plt.legend(title="Class")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return mse, r2

# xử lý với dữ liệu gốc
svm_orig = SVC(kernel="rbf", C=1, class_weight="balanced", random_state=42)
svm_orig.fit(X_train_orig, y_train.values.ravel())

mse_goc, r2_goc = svm_regression_on_decision_values(
    X_train_orig, X_test_orig, y_train, y_test, svm_orig, "Dữ liệu gốc"
)

# xử lý với dữ liệu PCA
svm_pca = SVC(kernel="rbf", C=1, class_weight="balanced", random_state=42)
svm_pca.fit(X_train_pca, y_train.values.ravel())
mse_pca, r2_pca = svm_regression_on_decision_values(
    X_train_pca, X_test_pca, y_train, y_test, svm_pca, "PCA"
)

# xử lý với dữ liệu LDA
svm_lda = SVC(kernel="rbf", C=1, class_weight="balanced", random_state=42)
svm_lda.fit(X_train_lda, y_train.values.ravel())

mse_lda, r2_lda = svm_regression_on_decision_values(
    X_train_lda, X_test_lda, y_train, y_test, svm_lda, "LDA"
)