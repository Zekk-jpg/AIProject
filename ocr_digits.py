import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from mnist_cnn import Net
from skimage import measure, morphology

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def segment_and_prepare_from_path(path: Path):
    img = Image.open(path).convert("L")
    # varsayım: beyaz zemin, siyah rakam → MNIST ile uyum için invert:
    arr = 255 - np.array(img, dtype=np.uint8)

    bin_img = (arr > 0).astype(np.uint8)
    bin_img = morphology.remove_small_objects(bin_img.astype(bool), min_size=30).astype(np.uint8)
    bin_img = morphology.binary_dilation(bin_img, morphology.square(3)).astype(np.uint8)
    labels = measure.label(bin_img, connectivity=2)
    props = measure.regionprops(labels)

    boxes = []
    for p in props:
        y0, x0, y1, x1 = p.bbox
        h, w = (y1 - y0), (x1 - x0)
        if h < 10 or w < 6 or h*w < 80:
            continue
        boxes.append((x0, y0, x1, y1))
    if not boxes:
        return []

    boxes.sort(key=lambda b: b[0])

    out = []
    pil_inverted = Image.fromarray(arr)
    for (x0, y0, x1, y1) in boxes:
        crop = pil_inverted.crop((x0, y0, x1, y1))
        m = max(crop.size)
        square = Image.new("L", (m, m), color=0)
        square.paste(crop, ((m - crop.size[0]) // 2, (m - crop.size[1]) // 2))
        digit28 = square.resize((28, 28), Image.Resampling.LANCZOS)
        out.append(transform(digit28).unsqueeze(0).to(device))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", "-i", required=True)
    args = ap.parse_args()

    model = Net().to(device)
    model.load_state_dict(torch.load("models/mnist_cnn.pt", map_location=device))
    model.eval()

    parts = segment_and_prepare_from_path(Path(args.image))
    if not parts:
        print("Hiç rakam bulunamadı.")
        return

    digits = []
    confs = []
    with torch.no_grad():
        for t in parts:
            out = model(t)
            pred = out.argmax(dim=1, keepdim=True).item()
            prob = torch.exp(out).cpu().numpy()[0][pred]
            digits.append(str(pred))
            confs.append(float(prob))

    print("Okunan sayı:", "".join(digits))
    print("Rakam güvenleri:", np.round(confs, 3))

if __name__ == "__main__":
    main()
