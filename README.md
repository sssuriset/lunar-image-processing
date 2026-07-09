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