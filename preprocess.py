"""
Brain MRI Image Contrast Enhancement & Digital Preprocessing

A simple pipeline that makes raw brain MRI slices easier to read:
    1. Intensity normalization   -> brings all images to the same 0-255 range
    2. Denoising (median filter) -> reduces noise while keeping edges
    3. CLAHE                     -> improves local contrast so structure borders are clearer

Usage:
    python preprocess.py --input data --output results
    python preprocess.py --input data --output results --clip 2.0 --tile 8 --median 3
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_gray(path):
    """Read an image as 8-bit grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def normalize(img):
    """Stretch intensities to the full 0-255 range."""
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def denoise(img, kernel=3):
    """Median filter: removes speckle noise and keeps edges fairly sharp."""
    return cv2.medianBlur(img, kernel)


def enhance_contrast(img, clip=2.0, tile=8):
    """CLAHE: contrast enhancement applied on small tiles, with a clip limit
    so that noise is not over-amplified."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(img)


def pipeline(img, clip=2.0, tile=8, median=3):
    """Run the full preprocessing pipeline and return every step."""
    norm = normalize(img)
    den = denoise(norm, median)
    enh = enhance_contrast(den, clip, tile)
    return {"original": img, "normalized": norm, "denoised": den, "enhanced": enh}


# ---------- Simple quality measures ----------

def contrast_std(img):
    """Standard deviation of intensities (higher = more contrast)."""
    return float(np.std(img))


def entropy(img):
    """Shannon entropy of the histogram (higher = more information in the image)."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_strength(img):
    """Mean gradient magnitude (higher = sharper structure borders)."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def measure(img):
    return {
        "contrast_std": contrast_std(img),
        "entropy": entropy(img),
        "edge_strength": edge_strength(img),
    }


# ---------- Saving results ----------

def save_comparison(steps, out_path, title):
    """Save a side-by-side figure: images on top, histograms below."""
    names = ["original", "denoised", "enhanced"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, name in enumerate(names):
        axes[0, i].imshow(steps[name], cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(name.capitalize())
        axes[0, i].axis("off")
        axes[1, i].hist(steps[name].ravel(), bins=256, range=(0, 255), color="gray")
        axes[1, i].set_xlabel("Intensity")
        axes[1, i].set_ylabel("Pixels")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Brain MRI contrast enhancement & preprocessing")
    ap.add_argument("--input", default="data", help="folder with MRI slice images")
    ap.add_argument("--output", default="results", help="folder to save results")
    ap.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--tile", type=int, default=8, help="CLAHE tile grid size")
    ap.add_argument("--median", type=int, default=3, help="median filter kernel (odd number)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    (out_dir / "enhanced").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in '{in_dir}'. See data/README.md.")

    rows = ["file,metric_set,contrast_std,entropy,edge_strength"]
    for path in files:
        steps = pipeline(load_gray(path), args.clip, args.tile, args.median)
        cv2.imwrite(str(out_dir / "enhanced" / f"{path.stem}_enhanced.png"), steps["enhanced"])
        save_comparison(steps, out_dir / "figures" / f"{path.stem}_comparison.png", path.name)
        for label in ("original", "enhanced"):
            m = measure(steps[label])
            rows.append(f"{path.name},{label},{m['contrast_std']:.2f},{m['entropy']:.3f},{m['edge_strength']:.2f}")
        print(f"Processed {path.name}")

    (out_dir / "metrics.csv").write_text("\n".join(rows) + "\n")
    print(f"Done. {len(files)} image(s) processed. Results saved in '{out_dir}'.")


if __name__ == "__main__":
    main()
