# mnist_canvas.py — çok basamaklı sayı tanıma (MNIST CNN)
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from skimage import measure, morphology
# Yeni skimage API'lerine uyumlu kare çekirdek seçici
try:
    # skimage >= 0.25
    from skimage.morphology import footprint_rectangle as _rect_footprint
    def _square_kernel(n: int):
        return _rect_footprint((n, n))
except Exception:
    try:
        # skimage 0.25 civarı
        from skimage.morphology import footprint_square as _sq
        def _square_kernel(n: int):
            return _sq(n)
    except Exception:
        # daha eski sürümler
        from skimage.morphology import square as _sq_legacy
        def _square_kernel(n: int):
            return _sq_legacy(n)

from skimage.morphology import binary_closing  # closing için

from mnist_cnn import Net  # eğitim kodu main-guard altında olduğu için import güvenli

# ----------------- Global ayarlar -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ----------------- Yardımcı: Segmentasyon -----------------
def segment_and_prepare(img_pil, min_size=30, dilate_size=1, close_size=2, pad_ratio=0.18):
    """
    img_pil: 'L' modunda (0 = siyah zemin, 255 = beyaz çizim) PIL.Image
    Dönüş: [(tensor, (x0,y0,x1,y1))]  -> soldan sağa sıralanmış rakam parçaları
    """
    arr = np.array(img_pil, dtype=np.uint8)

    # 1) Binarize: anti-alias gri pikselleri eşiğin altında bırak
    bin_img = (arr > 10).astype(np.uint8)

    # 2) Küçük gürültüleri at
    bin_img = morphology.remove_small_objects(bin_img.astype(bool), min_size=min_size).astype(np.uint8)

    # 3) Halkaları kapat (9/0/6 gibi) + 4) İnce çizgiyi biraz kalınlaştır
    if close_size and close_size >= 1:
        bin_img = binary_closing(bin_img.astype(bool), _square_kernel(close_size)).astype(np.uint8)
    if dilate_size and dilate_size >= 1:
        bin_img = morphology.binary_dilation(bin_img.astype(bool), _square_kernel(dilate_size)).astype(np.uint8)

    # 5) Bağlı bileşenler
    labels = measure.label(bin_img, connectivity=2)
    props = measure.regionprops(labels)

    boxes = []
    for p in props:
        y0, x0, y1, x1 = p.bbox
        h, w = (y1 - y0), (x1 - x0)
        area = h * w
        if h < 10 or w < 6 or area < 80:
            continue
        boxes.append((x0, y0, x1, y1))

    if not boxes:
        return []

    # 6) Soldan sağa sırala
    boxes.sort(key=lambda b: b[0])

    # 7) Her rakamı pad'leyip 28×28'e hazırla
    out = []
    for (x0, y0, x1, y1) in boxes:
        crop = img_pil.crop((x0, y0, x1, y1))  # siyah zemin, beyaz çizim

        w, h = crop.size
        m = max(w, h)
        pad = int(pad_ratio * m)  # MNIST benzeri çerçeve
        s = m + 2 * pad
        square_img = Image.new("L", (s, s), color=0)
        offset = ((s - w) // 2, (s - h) // 2)
        square_img.paste(crop, offset)

        digit28 = square_img.resize((28, 28), Image.Resampling.LANCZOS)
        t = transform(digit28).unsqueeze(0).to(device)
        out.append((t, (x0, y0, x1, y1)))

    return out

# ----------------- Uygulama -----------------
class DigitApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MNIST Digit Recognizer")

        # Modeli yükle
        model_path = Path("models/mnist_cnn.pt")
        if not model_path.exists():
            raise FileNotFoundError("models/mnist_cnn.pt bulunamadı. Önce 'python mnist_cnn.py' ile modeli eğit/kaydet.")
        self.model = Net().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # UI
        self.canvas = tk.Canvas(self.master, width=280, height=280, bg="white")
        self.canvas.pack()

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Tahmin Et", command=self.predict).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.button_frame, text="Temizle",   command=self.clear  ).grid(row=0, column=1, padx=5, pady=5)

        self.label = tk.Label(self.master, text="Bir sayı yazın (ör. 2025) ve 'Tahmin Et' tıklayın.", font=("Arial", 12))
        self.label.pack(pady=8)

        # Çizim buffer (siyah zemin, beyaz çizim)
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # BBox id'leri
        self.box_ids = []

        # Çizim bağlama
        self.brush_r = 8
        self.canvas.bind("<B1-Motion>", self.paint)

    # ---- Çizim ----
    def paint(self, event):
        r = self.brush_r
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                                fill="black", width=20, outline="black")
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

    # ---- Temizle ----
    def clear(self):
        for bid in self.box_ids:
            self.canvas.delete(bid)
        self.box_ids.clear()

        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Temizlendi. Yeni bir sayı çizin.")

    # ---- BBox çiz ----
    def _draw_box(self, box, color="#4CAF50"):
        x0, y0, x1, y1 = box
        bid = self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)
        self.box_ids.append(bid)

    # ---- Tahmin ----
    def predict(self):
        # Eski kutuları sil
        for bid in self.box_ids:
            self.canvas.delete(bid)
        self.box_ids.clear()

        parts = segment_and_prepare(self.image)
        if not parts:
            self.label.config(text="Rakam bulunamadı. Daha kalın ve aralıklı yazın.")
            return

        digits, confs = [], []
        with torch.no_grad():
            for tensor, box in parts:
                out = self.model(tensor)
                pred = out.argmax(dim=1, keepdim=True).item()
                prob = float(torch.exp(out)[0, pred].cpu().numpy())
                digits.append(str(pred))
                confs.append(prob)
                self._draw_box(box)

        number = "".join(digits)
        self.label.config(text=f"Tahmin: {number}  |  Ortalama Güven: {np.mean(confs):.2f}")

# ----------------- Çalıştır -----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
