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


def setupdirs():
    IMGOUT.mkdir(parents=True, exist_ok=True)
    PLOTOUT.mkdir(parents=True, exist_ok=True)
    TILEOUT.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)


def loadimg(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=float)


def normpercent(img, lo=1, hi=99):
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


def findedges(img):
    normed = normpercent(img)
    return feature.canny(normed, sigma=2)


def residual(img):
    normed = normpercent(img)
    smooth = filters.gaussian(normed, sigma=2)
    return normed - smooth


def sharpness(img):
    normed = normpercent(img)
    lap = filters.laplace(normed)
    return float(np.var(lap))


def edgefrac(edgeimg):
    return float(np.sum(edgeimg) / edgeimg.size)


def saveimg(img, outpath, title=None, vmin=None, vmax=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def savehist(img, outpath):
    plt.figure(figsize=(8, 5))
    plt.hist(img.flatten(), bins=100)
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.title("Pixel intensity histogram")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def savecompare(original, normed, stretched, edgeimg, resid, outpath):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    reslim = max(abs(float(np.min(resid))), abs(float(np.max(resid))))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(normed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Normalized")
    axes[1].axis("off")

    axes[2].imshow(stretched, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Stretched")
    axes[2].axis("off")

    axes[3].imshow(edgeimg, cmap="gray")
    axes[3].set_title("Edges")
    axes[3].axis("off")

    axes[4].imshow(resid, cmap="gray", vmin=-reslim, vmax=reslim)
    axes[4].set_title("Residual")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def savetiles(img, path, tilesize=300, maxtiles=12):
    height, width = img.shape
    normed = normpercent(img)
    count = 0

    for row in range(0, height, tilesize):
        for col in range(0, width, tilesize):
            rowend = min(row + tilesize, height)
            colend = min(col + tilesize, width)
            tile = normed[row:rowend, col:colend]
            tilepath = TILEOUT / f"{path.stem}_tile_{count:03d}.png"
            saveimg(tile, tilepath, vmin=0, vmax=1)
            count += 1

            if count >= maxtiles:
                return count

    return count


def summarize(path, img, edgeimg, resid):
    p1 = float(np.percentile(img, 1))
    p5 = float(np.percentile(img, 5))
    p95 = float(np.percentile(img, 95))
    p99 = float(np.percentile(img, 99))

    return {
        "file": path.name,
        "heightPx": int(img.shape[0]),
        "widthPx": int(img.shape[1]),
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "mean": float(np.mean(img)),
        "median": float(np.median(img)),
        "std": float(np.std(img)),
        "p1": p1,
        "p99": p99,
        "contrastRange": p99 - p1,
        "brightFrac": float(np.mean(img >= p95)),
        "darkFrac": float(np.mean(img <= p5)),
        "sharpness": sharpness(img),
        "edgeFrac": edgefrac(edgeimg),
        "residStd": float(np.std(resid)),
    }


def writetxt(path, stats):
    outfile = PLOTOUT / f"{path.stem}_summary.txt"

    with open(outfile, "w") as file:
        file.write(f"Image file: {path.name}\n\n")
        file.write("Pixel and image quality stats\n")
        file.write("-----------------------------\n")

        for key, value in stats.items():
            file.write(f"{key}: {value}\n")


def writecsv(rows):
    if not rows:
        return

    fields = list(rows[0].keys())

    with open(METRICS, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def runone(path):
    img = loadimg(path)
    normed = normpercent(img)
    stretched = stretch(img)
    edgeimg = findedges(img)
    resid = residual(img)
    reslim = max(abs(float(np.min(resid))), abs(float(np.max(resid))))

    stats = summarize(path, img, edgeimg, resid)

    saveimg(normed, IMGOUT / f"{path.stem}_norm.png", "Normalized lunar image", vmin=0, vmax=1)
    saveimg(stretched, IMGOUT / f"{path.stem}_stretch.png", "Contrast stretched lunar image", vmin=0, vmax=1)
    saveimg(edgeimg, IMGOUT / f"{path.stem}_edges.png", "Lunar surface edges")
    saveimg(resid, IMGOUT / f"{path.stem}_residual.png", "Residual image", vmin=-reslim, vmax=reslim)
    savehist(img, PLOTOUT / f"{path.stem}_hist.png")
    savecompare(img, normed, stretched, edgeimg, resid, PLOTOUT / f"{path.stem}_compare.png")

    stats["tilesSaved"] = savetiles(img, path)
    writetxt(path, stats)

    print(f"\nProcessed {path.name}")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


def main():
    setupdirs()

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
        rows.append(runone(path))

    writecsv(rows)
    print(f"\nSaved metrics table to {METRICS}")


if __name__ == "__main__":
    main()
