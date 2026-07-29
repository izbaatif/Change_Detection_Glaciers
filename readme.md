# Glacier Surface Change Detection Using SAR Time-Series and DSM Validation

This repository contains a Python-based processing framework for detecting glacier surface changes using multi-temporal Synthetic Aperture Radar (SAR) imagery and validating detected changes using optical imagery and Digital Surface Models (DSM).

The framework was developed for glacier monitoring applications, particularly for detecting structural changes in glacier ablation zones using Sentinel-1 dual-polarization SAR time-series data.

The pipeline integrates:

- Sentinel-1 VV/VH SAR time-series analysis
- Statistical anomaly detection
- Machine learning based anomaly detection
- Optical image similarity validation
- DSM-based elevation change validation
- Geospatial visualization


---

# Features

## SAR Preprocessing

The pipeline supports:

- GeoTIFF SAR data handling
- Glacier boundary based cropping
- VV and VH polarization processing
- Temporal stack generation
- Spatial filtering
- VV/VH ratio computation


## SAR Anomaly Detection Methods

Implemented anomaly detection approaches:

### Statistical Methods

- Spatial thresholding
- Rolling temporal thresholding
- Z-score based detection
- Wilks' Lambda change detection


### Machine Learning Method

- Isolation Forest based anomaly detection


## Optical Validation

Detected SAR anomalies can be validated using optical imagery through:

- FSIM (Feature Similarity Index)
- SSIM (Structural Similarity Index)
- ZNCC (Zero Normalized Cross Correlation)
- LESH (Local Edge Structure Histogram)
- Mutual Information
- HOG similarity
- SAD (Sum of Absolute Differences)


Statistical validation includes:

- Mann-Whitney U test
- Cohen's d effect size
- Bootstrap sampling comparison against random glacier locations


## DSM Validation

The framework supports:

- DSM clipping using glacier boundaries
- DSM coregistration
- Elevation difference computation

\[
DSM_{difference}=DSM_{after}-DSM_{before}
\]

- SAR anomaly projection onto DSM grids
- Visualization of anomaly locations over elevation change maps



### Input Data Organization

project/

│
├── burts/
│
│   ├── 2025_04_05/
│   │      ├── VV.tif
│   │      └── VH.tif
│   │
│   ├── 2025_04_17/
│   │      ├── VV.tif
│   │      └── VH.tif
│
│
├── optical_data/
│
│   ├── before_image.tif
│   └── after_image.tif
│
│
├── DSM/
│
│   ├── before_dsm.tif
│   └── after_dsm.tif
│
│
├── glacier_boundary/
│
│   └── glacier.shp
│
└── outputs/
