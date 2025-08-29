import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from mnist_cnn import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yükle
model = Net().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pt", map_location=device))
model.eval()

# Preprocess (28x28 MNIST formatına uygun)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_image(img_path, show=False):
    img = Image.open(img_path).convert("L")

    # 1) Arka planı beyaz → rakamı siyah çevirelim (MNIST formatı tam tersi)
    img = ImageOps.invert(img)

    # 2) Otsu threshold ile ikili hale getir (çok ince çizgiler kaybolmasın)
    arr = np.array(img)
    threshold = arr.mean()  # basit eşik
    arr = (arr > threshold).astype(np.uint8) * 255

    # 3) Pillow image'e geri dön
    img = Image.fromarray(arr)

    # 4) Kare tuval + merkezleme
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    max_side = max(img.size)
    square = Image.new("L", (max_side, max_side), color=0)
    offset = ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2)
    square.paste(img, offset)

    # 5) 28×28'e ölçekle
    img = square.resize((28, 28), Image.Resampling.LANCZOS)

    if show:
        plt.imshow(img, cmap="gray")
        plt.title("Preprocessed Input")
        plt.show()

    # 6) Tensor’a dönüştür
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        probs = torch.exp(output).cpu().numpy()[0]
    return pred.item(), probs

# Test
digit_path = Path("digit.png")
label, probs = predict_image(digit_path, show=True)
print("Tahmin:", label)
print("Olasılıklar:", np.round(probs, 3))
