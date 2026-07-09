from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters, feature


RAW_DIR = Path("data/raw")
OUT_IMG = Path("outputs/images")
OUT_PLOTS = Path("outputs/plots")
OUT_TILES = Path("outputs/tiles")
METRICS_FILE = Path("outputs/image_metrics.csv")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_TILES.mkdir(parents=True, exist_ok=True)


def load_image(path):
    image = Image.open(path).convert("L")
    return np.array(image, dtype=float)


def normalize_percentile(image, low_percentile=1, high_percentile=99):
    low = np.percentile(image, low_percentile)
    high = np.percentile(image, high_percentile)

    if high == low:
        return np.zeros_like(image)

    clipped = np.clip(image, low, high)
    return (clipped - low) / (high - low)


def contrast_stretch(image, low_percentile=5, high_percentile=95):
    low = np.percentile(image, low_percentile)
    high = np.percentile(image, high_percentile)

    if high == low:
        return np.zeros_like(image)

    stretched = np.clip(image, low, high)
    return (stretched - low) / (high - low)


def detect_edges(image):
    normalized = normalize_percentile(image)
    return feature.canny(normalized, sigma=2)


def estimate_noise(image):
    normalized = normalize_percentile(image)
    blurred = filters.gaussian(normalized, sigma=2)
    return normalized - blurred


def sharpness_score(image):
    normalized = normalize_percentile(image)
    laplace_image = filters.laplace(normalized)
    return float(np.var(laplace_image))


def edge_density(edges):
    return float(np.sum(edges) / edges.size)


def save_image(image, output_path, title=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_histogram(image, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(image.flatten(), bins=100)
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.title("Pixel Intensity Histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_comparison(original, normalized, stretched, edges, noise, output_path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(normalized, cmap="gray")
    axes[1].set_title("Normalized")
    axes[1].axis("off")

    axes[2].imshow(stretched, cmap="gray")
    axes[2].set_title("Contrast Stretched")
    axes[2].axis("off")

    axes[3].imshow(edges, cmap="gray")
    axes[3].set_title("Detected Edges")
    axes[3].axis("off")

    axes[4].imshow(noise, cmap="gray")
    axes[4].set_title("Noise Residual")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_tiles(image, path, tile_size=300):
    height, width = image.shape
    normalized = normalize_percentile(image)

    count = 0

    for row in range(0, height - tile_size + 1, tile_size):
        for col in range(0, width - tile_size + 1, tile_size):
            tile = normalized[row:row + tile_size, col:col + tile_size]
            tile_path = OUT_TILES / f"{path.stem}_tile_{count:03d}.png"
            save_image(tile, tile_path)
            count += 1

            if count >= 12:
                return count

    return count


def image_summary(path, image, edges, noise):
    return {
        "file": path.name,
        "height_px": int(image.shape[0]),
        "width_px": int(image.shape[1]),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "median": float(np.median(image)),
        "standard_deviation": float(np.std(image)),
        "p1": float(np.percentile(image, 1)),
        "p99": float(np.percentile(image, 99)),
        "sharpness_score": sharpness_score(image),
        "edge_density": edge_density(edges),
        "noise_residual_std": float(np.std(noise)),
    }


def write_summary(path, stats):
    output_file = OUT_PLOTS / f"{path.stem}_summary.txt"

    with open(output_file, "w") as file:
        file.write(f"Image file: {path.name}\n\n")
        file.write("Pixel and image-quality statistics\n")
        file.write("----------------------------------\n")

        for key, value in stats.items():
            file.write(f"{key}: {value}\n")


def write_metrics_csv(all_stats):
    if not all_stats:
        return

    fieldnames = list(all_stats[0].keys())

    with open(METRICS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)


def process_file(path):
    image = load_image(path)

    normalized = normalize_percentile(image)
    stretched = contrast_stretch(image)
    edges = detect_edges(image)
    noise = estimate_noise(image)
    stats = image_summary(path, image, edges, noise)

    save_image(normalized, OUT_IMG / f"{path.stem}_normalized.png", "Normalized Lunar Image")
    save_image(stretched, OUT_IMG / f"{path.stem}_contrast_stretched.png", "Contrast-Stretched Lunar Image")
    save_image(edges, OUT_IMG / f"{path.stem}_edges.png", "Detected Lunar Surface Edges")
    save_image(noise, OUT_IMG / f"{path.stem}_noise_residual.png", "Noise Residual Estimate")

    save_histogram(image, OUT_PLOTS / f"{path.stem}_histogram.png")
    save_comparison(
        image,
        normalized,
        stretched,
        edges,
        noise,
        OUT_PLOTS / f"{path.stem}_comparison.png"
    )

    tile_count = save_tiles(image, path)
    stats["tiles_saved"] = tile_count

    write_summary(path, stats)

    print(f"\nProcessed {path.name}")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


def main():
    image_files = []
    image_files.extend(RAW_DIR.glob("*.png"))
    image_files.extend(RAW_DIR.glob("*.jpg"))
    image_files.extend(RAW_DIR.glob("*.jpeg"))
    image_files.extend(RAW_DIR.glob("*.tif"))
    image_files.extend(RAW_DIR.glob("*.tiff"))

    if not image_files:
        print("No image files found in data/raw.")
        print("Add a lunar image file to data/raw, then run the script again.")
        return

    all_stats = []

    for path in image_files:
        stats = process_file(path)
        all_stats.append(stats)

    write_metrics_csv(all_stats)
    print(f"\nSaved metrics table to {METRICS_FILE}")


if __name__ == "__main__":
    main()