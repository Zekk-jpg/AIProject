from pathlib import Path
import joblib
from sklearn.datasets import load_digits

# Kaydedilmiş modeli yükle
model_path = Path("models/digits_logreg.joblib")
clf = joblib.load(model_path)

# Örnek veri: test setinden bir görüntü alalım
digits = load_digits()
sample_index = 0
X_sample = digits.data[sample_index].reshape(1, -1)
y_true = digits.target[sample_index]

y_pred = clf.predict(X_sample)[0]
proba = clf.predict_proba(X_sample)[0]

print(f"Gerçek etiket: {y_true}")
print(f"Tahmin: {y_pred}")
print("Sınıf olasılıkları (ilk 10):", [round(p, 3) for p in proba[:10]])
