# Lunar Image Processing and Quality Analysis

This project processes public lunar imagery with Python to inspect image quality, contrast behavior, brightness distribution, and surface-feature visibility. The workflow treats each image as numerical pixel data, applies basic image-processing methods, and generates diagnostic outputs for comparison and documentation.

## Data Source

The test image used in this project is NASA's Moon Mosaic, a public lunar photomosaic made from Lunar Reconnaissance Orbiter imagery.

## Project Motivation

Lunar images need more than visual inspection. Brightness distribution, contrast, edge visibility, and noise affect whether a lunar image is useful for surface interpretation and documentation. This project builds a small image-analysis workflow that connects astronomical imaging with numerical analysis in Python.

## Methods

The workflow:

1. Loads lunar image files as grayscale numerical arrays
2. Computes pixel statistics
3. Generates pixel-intensity histograms
4. Applies percentile-based normalization
5. Applies contrast stretching
6. Detects surface edges using a Canny edge detector
7. Estimates residual noise by subtracting a blurred image from the normalized image
8. Saves processed images and diagnostic plots

## Tools

- Python
- NumPy
- Matplotlib
- Pillow
- scikit-image

## Outputs

The project generates:

- Normalized lunar images
- Contrast-stretched lunar images
- Edge-detection images
- Noise residual images
- Pixel-intensity histograms
- Side-by-side comparison plots
- Text summaries of image statistics

## Results

The workflow successfully processed the lunar mosaic and generated diagnostic image products. The processed image set includes normalized imagery, contrast-stretched imagery, an edge map, a noise-residual estimate, image tiles, a histogram, a side-by-side comparison plot, and a CSV metrics table.

For the test image, the script computed:

- Image size: 1024 px × 1024 px
- Mean pixel value: 97.94
- Median pixel value: 110.00
- Standard deviation: 67.17
- 1st percentile pixel value: 0.00
- 99th percentile pixel value: 215.00
- Sharpness score: 0.0114
- Edge density: 0.0486
- Noise residual standard deviation: 0.0424
- Tiles saved: 9

The contrast-stretched output improves visibility in brighter lunar regions. The edge-detection output highlights strong surface boundaries, albedo transitions, crater-like structures, and visible mosaic seams. The noise-residual output shows small-scale variation after smoothing.

## Example Diagnostic Output

The main comparison plot shows the original lunar image, normalized image, contrast-stretched image, detected edges, and noise-residual estimate side by side. This makes it easier to evaluate how each processing step changes surface visibility and image quality.

## Limitations

The test image is a public lunar mosaic rather than a raw calibrated science frame. Some detected edges come from mosaic boundaries and compression artifacts, not only lunar surface features. This project is meant as an image-processing workflow demonstration, not a calibrated planetary-science measurement pipeline.

## Project Structure

```text
lunar-image-processing/
├── data/
│   └── raw/
├── outputs/
│   ├── images/
│   └── plots/
├── src/
│   └── process_image.py
├── README.md
├── requirements.txt
└── .gitignore