from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1) Veri: sklearn'in yerleşik "digits" (8x8) veri seti
digits = load_digits()
X, y = digits.data, digits.target

# 2) Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Model: L2 cezalıklı lojistik regresyon
clf = LogisticRegression(max_iter=2000, n_jobs=None)  # CPU’yu kullanır
clf.fit(X_train, y_train)

# 4) Değerlendirme
y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

# 5) Karışıklık matrisi görselleştir
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(values_format="d")
plt.title("Digits - Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

# 6) Modeli kaydet
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
model_path = models_dir / "digits_logreg.joblib"
joblib.dump(clf, model_path)
print(f"\nModel kaydedildi: {model_path.resolve()}")
