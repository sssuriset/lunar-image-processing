# Lunar Image Processing and Quality Analysis

This is a small Python project for working with lunar surface images. I made it to test basic image-processing steps that help inspect contrast, brightness spread, edge visibility, and residual noise in lunar imagery.

The script reads images from `data/raw/`, processes them, and saves the outputs into the `outputs/` folder.

## What it does

For each image, the script creates:

- a normalized grayscale image
- a contrast-stretched image
- an edge map
- a residual/noise image
- smaller image tiles
- a comparison plot
- a CSV row with image-quality metrics

This is not a crater-detection or terrain-classification project. It is a preprocessing and quality-check workflow for comparing lunar image products.

## Installation

Clone the repo:

    git clone https://github.com/sssuriset/lunar-image-processing.git
    cd lunar-image-processing

Install the dependencies:

    python3 -m pip install -r requirements.txt

Main packages used:

    numpy
    matplotlib
    pillow
    scikit-image

## How to run it

Put one or more lunar images in:

    data/raw/

Then run:

    python3 src/process_image.py

Supported image types:

    .png
    .jpg
    .jpeg
    .tif
    .tiff

The script saves processed images, plots, tiles, and the metrics CSV in `outputs/`.

## Processing steps

### Normalization

Each image is converted to grayscale and rescaled using percentile limits. Percentile scaling is used instead of the absolute minimum and maximum because a few extreme pixels can flatten the useful brightness range.

### Contrast stretching

The script stretches the useful brightness range of the image. This makes crater rims, shadow boundaries, surface texture, and bright terrain easier to compare visually.

### Edge detection

An edge map is created to show sharp boundaries in the image. This gives a rough check of how much visible structure is present after processing.

### Residual image

The script subtracts a blurred version of the normalized image from the normalized image itself. The result leaves smaller-scale variation behind. I use this as a residual/noise check, not as a calibrated noise model.

The residual plot uses symmetric limits so positive and negative residuals are displayed evenly.

### Tiling

The image is split into smaller sections. Edge tiles are kept when the image dimensions do not divide evenly by the tile size, so the script does not quietly drop the right or bottom edge of an image.

## Metrics

The CSV file records:

    mean
    median
    std
    sharpness
    edgeFrac
    residStd
    contrastRange
    brightFrac
    darkFrac

Metric meanings:

- `mean`: average normalized brightness
- `median`: median normalized brightness
- `std`: spread of pixel values
- `sharpness`: rough sharpness estimate based on image variation
- `edgeFrac`: fraction of pixels marked as edges
- `residStd`: spread of the residual image
- `contrastRange`: difference between high and low percentile brightness values
- `brightFrac`: fraction of very bright pixels
- `darkFrac`: fraction of very dark pixels

These metrics are useful for comparing images in the same batch. They should not be treated as calibrated lunar surface measurements.

## Limitations

This project uses classical image-processing methods only. It does not georeference images, match features to lunar catalogs, classify terrain, or estimate physical reflectance.

Some detected edges may come from image seams, compression artifacts, or brightness transitions rather than actual lunar surface features.

## Possible upgrades

Reasonable next steps:

- add FITS image support
- add command-line options for tile size and contrast limits
- add batch summary plots
- test crater or ridge candidate detection
- compare processed outputs against labeled lunar features
