import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
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

def softmax_as_regression(X_train, X_test, y_train, y_test, lr_model, dataset_name):

    y_train_1d = y_train.values.ravel()
    y_test_1d  = y_test.values.ravel()

    proba   = lr_model.predict_proba(X_test)
    classes = lr_model.classes_
    y_pred  = proba @ classes

    mse = mean_squared_error(y_test_1d, y_pred)
    r2  = r2_score(y_test_1d, y_pred)

    print(f"\n[Softmax → Hồi quy] - {dataset_name}")
    print(f"MSE : {mse:.4f}")
    print(f"R²  : {r2:.4f}")

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=y_test_1d, y=y_pred,
        hue=y_test_1d, palette='tab10',
        alpha=0.7, edgecolor='k'
    )
    min_val = min(y_test_1d.min(), y_pred.min())
    max_val = max(y_test_1d.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle='--', color='red', label='Ideal (y = x)')
    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    plt.xlabel("Nhãn thực tế (mã số)")
    plt.ylabel("Giá trị dự đoán liên tục")
    plt.title(f"Softmax → Hồi quy – {dataset_name}")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mse, r2

# thực hiện trên dữ liệu gốc
lr_orig = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                              max_iter=1000, random_state=42)
lr_orig.fit(X_train_orig, y_train.values.ravel())

mse_goc, r2_goc = softmax_as_regression(
    X_train_orig, X_test_orig, y_train, y_test, lr_orig, "Dữ liệu gốc"
)

# thực hiện trên dữ liệu PCA
lr_pca = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                             max_iter=1000, random_state=42)
lr_pca.fit(X_train_pca, y_train.values.ravel())

mse_pca, r2_pca = softmax_as_regression(
    X_train_pca, X_test_pca, y_train, y_test, lr_pca, "PCA"
)

# thực hiện trên dữ liệu LDA
lr_lda = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                             max_iter=1000, random_state=42)
lr_lda.fit(X_train_lda, y_train.values.ravel())

mse_lda, r2_lda = softmax_as_regression(
    X_train_lda, X_test_lda, y_train, y_test, lr_lda, "LDA"
)