from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters, feature

RAW = Path("data/raw")
IMGOUT = Path("outputs/images")
PLOTOUT = Path("outputs/plots")
TILEOUT = Path("outputs/tiles")
METRICS = Path("outputs/image_metrics.csv")


def setup_dirs():
    IMGOUT.mkdir(parents=True, exist_ok=True)
    PLOTOUT.mkdir(parents=True, exist_ok=True)
    TILEOUT.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)


def load_img(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=float)


def norm_percent(img, lo=1, hi=99):
    low = np.percentile(img, lo)
    high = np.percentile(img, hi)

    if high == low:
        return np.zeros_like(img)

    clipped = np.clip(img, low, high)
    return (clipped - low) / (high - low)


def stretch(img, lo=5, hi=95):
    low = np.percentile(img, lo)
    high = np.percentile(img, hi)

    if high == low:
        return np.zeros_like(img)

    clipped = np.clip(img, low, high)
    return (clipped - low) / (high - low)


def find_edges(img):
    normed = norm_percent(img)
    return feature.canny(normed, sigma=2)


def residual(img):
    normed = norm_percent(img)
    smooth = filters.gaussian(normed, sigma=2)
    return normed - smooth


def sharpness(img):
    normed = norm_percent(img)
    lap = filters.laplace(normed)
    return float(np.var(lap))


def edge_frac(edge_img):
    return float(np.sum(edge_img) / edge_img.size)


def save_img(img, outpath, title=None, vmin=None, vmax=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_hist(img, outpath):
    plt.figure(figsize=(8, 5))
    plt.hist(img.flatten(), bins=100)
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.title("Pixel intensity histogram")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_compare(original, normed, stretched, edge_img, resid, outpath):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    res_lim = max(abs(float(np.min(resid))), abs(float(np.max(resid))))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(normed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Normalized")
    axes[1].axis("off")

    axes[2].imshow(stretched, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Stretched")
    axes[2].axis("off")

    axes[3].imshow(edge_img, cmap="gray")
    axes[3].set_title("Edges")
    axes[3].axis("off")

    axes[4].imshow(resid, cmap="gray", vmin=-res_lim, vmax=res_lim)
    axes[4].set_title("Residual")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_tiles(img, path, tile_size=300, max_tiles=12):
    height, width = img.shape
    normed = norm_percent(img)
    count = 0

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            row_end = min(row + tile_size, height)
            col_end = min(col + tile_size, width)
            tile = normed[row:row_end, col:col_end]
            tile_path = TILEOUT / f"{path.stem}_tile_{count:03d}.png"
            save_img(tile, tile_path, vmin=0, vmax=1)
            count += 1

            if count >= max_tiles:
                return count

    return count


def summarize(path, img, edge_img, resid):
    p1 = float(np.percentile(img, 1))
    p5 = float(np.percentile(img, 5))
    p95 = float(np.percentile(img, 95))
    p99 = float(np.percentile(img, 99))

    return {
        "file": path.name,
        "height_px": int(img.shape[0]),
        "width_px": int(img.shape[1]),
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "mean": float(np.mean(img)),
        "median": float(np.median(img)),
        "std": float(np.std(img)),
        "p1": p1,
        "p99": p99,
        "contrast_range": p99 - p1,
        "bright_frac": float(np.mean(img >= p95)),
        "dark_frac": float(np.mean(img <= p5)),
        "sharpness": sharpness(img),
        "edge_frac": edge_frac(edge_img),
        "resid_std": float(np.std(resid)),
    }


def write_txt(path, stats):
    outfile = PLOTOUT / f"{path.stem}_summary.txt"

    with open(outfile, "w") as file:
        file.write(f"Image file: {path.name}\n\n")
        file.write("Pixel and image quality stats\n")
        file.write("-----------------------------\n")

        for key, value in stats.items():
            file.write(f"{key}: {value}\n")


def write_csv(rows):
    if not rows:
        return

    fields = list(rows[0].keys())

    with open(METRICS, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one(path):
    img = load_img(path)
    normed = norm_percent(img)
    stretched = stretch(img)
    edge_img = find_edges(img)
    resid = residual(img)
    res_lim = max(abs(float(np.min(resid))), abs(float(np.max(resid))))

    stats = summarize(path, img, edge_img, resid)

    save_img(normed, IMGOUT / f"{path.stem}_norm.png", "Normalized lunar image", vmin=0, vmax=1)
    save_img(stretched, IMGOUT / f"{path.stem}_stretch.png", "Contrast stretched lunar image", vmin=0, vmax=1)
    save_img(edge_img, IMGOUT / f"{path.stem}_edges.png", "Lunar surface edges")
    save_img(resid, IMGOUT / f"{path.stem}_residual.png", "Residual image", vmin=-res_lim, vmax=res_lim)
    save_hist(img, PLOTOUT / f"{path.stem}_hist.png")
    save_compare(img, normed, stretched, edge_img, resid, PLOTOUT / f"{path.stem}_compare.png")

    stats["tiles_saved"] = save_tiles(img, path)
    write_txt(path, stats)

    print(f"\nProcessed {path.name}")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


def main():
    setup_dirs()

    imgs = []
    imgs.extend(RAW.glob("*.png"))
    imgs.extend(RAW.glob("*.jpg"))
    imgs.extend(RAW.glob("*.jpeg"))
    imgs.extend(RAW.glob("*.tif"))
    imgs.extend(RAW.glob("*.tiff"))

    if not imgs:
        print("No image files found in data/raw.")
        print("Add a lunar image file to data/raw, then run the script again.")
        return

    rows = []

    for path in imgs:
        rows.append(run_one(path))

    write_csv(rows)
    print(f"\nSaved metrics table to {METRICS}")


if __name__ == "__main__":
    main()
