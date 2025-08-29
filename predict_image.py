import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import joblib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, footprint_rectangle
from scipy.ndimage import center_of_mass

def preprocess_image_to_digits_space(img_path: Path, show_steps: bool = False) -> np.ndarray:
    # 1) Yükle ve griye çevir
    img = Image.open(img_path).convert("L")

    # 2) Kontrast ve gerekli ise tersleme
    img_eq = ImageOps.autocontrast(img, cutoff=2)
    arr = np.asarray(img_eq, dtype=np.float32)

    # 3) Otsu ile binarize (arka plan/siyah-beyaz ayrımı)
    t = threshold_otsu(arr)
    # arka planı siyah, rakamı beyaz yap
    bin_img = (arr > t).astype(np.uint8)  # 0/1

    # Ortalama çok karanlıksa tersle
    if bin_img.mean() < 0.5:
        bin_img = 1 - bin_img

    # 4) Rakamı biraz kalınlaştır (ince çizimler için çok işe yarar)
    from skimage.morphology import dilation, footprint_rectangle
    bin_img = dilation(bin_img, footprint_rectangle((3, 3)))  # DOĞRU: 3x3 kare çekirdek

    # 5) Bounding box ile kırp
    coords = np.argwhere(bin_img > 0)
    if coords.size == 0:
        # boş resim koruması
        square_img = Image.fromarray((bin_img * 255).astype(np.uint8))
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        crop = bin_img[y0:y1, x0:x1]

        # 6) Kare tuvale pad’le (orantı koru)
        h, w = crop.shape
        s = max(h, w)
        square_arr = np.zeros((s, s), dtype=np.uint8)
        y_off = (s - h) // 2
        x_off = (s - w) // 2
        square_arr[y_off:y_off+h, x_off:x_off+w] = crop

        # 7) Kütle merkeziyle hafif hizalama (ortalamaya getir)
        cy, cx = center_of_mass(square_arr)
        ty = int(round(s/2 - cy))
        tx = int(round(s/2 - cx))
        square_arr = np.roll(square_arr, shift=(ty, tx), axis=(0, 1))

        square_img = Image.fromarray((square_arr * 255).astype(np.uint8))

    # 8) 8×8’e indir
    img_8 = square_img.resize((8, 8), Image.Resampling.LANCZOS)

    # 9) 0..255 -> 0..16 ölçeği
    arr8 = np.asarray(img_8, dtype=np.float32)
    arr8 = np.clip(arr8 / 255.0 * 16.0, 0, 16)

    if show_steps:
        fig, axes = plt.subplots(1, 4, figsize=(8, 2))
        axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(img_eq, cmap="gray"); axes[1].set_title("Auto-contrast"); axes[1].axis("off")
        axes[2].imshow(square_img, cmap="gray"); axes[2].set_title("Centered+Thick"); axes[2].axis("off")
        axes[3].imshow(arr8, cmap="gray"); axes[3].set_title("8×8 (0..16)"); axes[3].axis("off")
        plt.tight_layout(); plt.show()

    return arr8.reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="Çizdiğin rakamı (PNG/JPG) tahmin et")
    parser.add_argument("--image", "-i", type=str, required=True, help="Görüntü yolu (PNG/JPG)")
    parser.add_argument("--model", "-m", type=str, default="models/digits_logreg.joblib", help="Model yolu")
    parser.add_argument("--show", action="store_true", help="Önişleme adımlarını göster")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}. Önce main.py çalıştır.")

    clf = joblib.load(model_path)

    x = preprocess_image_to_digits_space(Path(args.image), show_steps=args.show)
    pred = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0]

    print(f"Tahmin: {pred}")
    print("Olasılıklar (0..9):", [round(float(p), 3) for p in proba])

if __name__ == "__main__":
    main()
