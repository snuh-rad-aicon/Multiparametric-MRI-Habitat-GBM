# Multiparametric-MRI-Habitat-GBM

**Researcher**: Min Seo Choi, Department of Radiology, Seoul National University Hospital

This research focuses on analyzing tumor habitats within IDH-wildtype glioblastoma (GBM) using Gaussian Mixture Model (GMM) based clustering. We utilize multi-parameteric physiologic MRI (ADC, rCBV, Ktrans) to quantify intratumoral heterogeneity and identify habitat patterns differences between molecular GBM (mol-GBM) and histological GBM (hist-GBM)

## Research Overview

This study presents the following methodology to understand intratumoral heterogeneity in glioblastoma:

- **Multi-parameteric MRI Analysis**: Characterization of tumor microenvironment using ADC (diffusion), rCBV (perfusion), and Ktrans (permeability) imaging from DWI, DSC-MRI, and DCE-MRI, respectively
- **Habitat-based Clustering**: Classification of 8 molecular habitat regions using GMM
- **Tumor Heterogeneity Quantification**: Analysis of tumor biological characteristics through distribution and properties of each habitat
- **Prognostic Prediction Modeling**: Clinical outcome prediction through histogram feature-based logistic regression models

This study aims to enhance
- the understanding of intratumoral heterogeneity in glioblastoma (GBM) by radiologically describe tumor habitats in a high-dimensional manner using multiparametric physiologic MRI.
- the understading of recently distinguished but relatively underexplored mol-GBM, thereby contributing to refined radiological insights and advancing stratified therapeutic approaches.

### Model Download

Download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1I9gCqnBPHSGjp3nKmY6w_sVHz21MPSeK?usp=sharing) and place them in the appropriate directories:

1. Download the `habitat` folder and place it in `.models/habitat/`
2. Download the `histogram_feature` folder and place it in `.models/histogram_feature/`

The final directory structure should be:
```
.models/
├── habitat/
│   └── gmm/
│       ├── gmm_adc_n2.pkl
│       ├── gmm_cbv_n2.pkl
│       └── gmm_ktrans_n2.pkl
└── histogram_feature/
    └── [histogram feature model files]
```

## Image Preprocessing

The following preprocessing steps are required before using this tool:

### 1. Image Resampling
Resample CE-T1WI to isometric-voxel size to enable evaluation of changes between consecutive scans.

### 2. Skull Stripping
Perform skull stripping on all sequences using the [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) algorithm.

### 3. Image Registration
Register all images except CE-T1WI to the isometric voxel CE-T1WI image using rigid transformation with [ANTsPy](https://github.com/ANTsX/ANTsPy).

### 4. Tumor Segmentation
Generate tumor segmentation masks using [HD-GLIO](https://github.com/CCI-Bonn/HD-GLIO) (a variant of the nnU-Net framework) with T1, CET1, T2, and FLAIR data.

## Input Files

The following NIfTI files are required:
- `adc.nii.gz`: ADC (Apparent Diffusion Coefficient) map (units: μm²/ms)
- `cbv.nii.gz`: CBV (Cerebral Blood Volume) map  
- `ktrans.nii.gz`: KTRANS (Volume Transfer Constant) map (units: min⁻¹)
- `WT_tumor_mask.nii.gz`: Tumor region mask (binary mask)

**Note**: ADC should be normalized to μm²/ms units, and KTRANS to min⁻¹ units.

## Usage

### 1. Command Line Usage

```bash
python analysis/habitat_clustering.py \
    --adc path/to/adc.nii.gz \
    --cbv path/to/cbv.nii.gz \
    --ktrans path/to/ktrans.nii.gz \
    --mask path/to/WT_tumor_mask.nii.gz \
    --model-dir models/habitat/gmm \
    --output path/to/cluster_map_1-8.nii.gz
```

#### Parameter Description
- `--adc`: ADC NIfTI file path (required)
- `--cbv`: CBV NIfTI file path (required)
- `--ktrans`: KTRANS NIfTI file path (required)
- `--mask`: Tumor mask NIfTI file path (required)
- `--model-dir`: Directory containing GMM models (default: `.models/habitat/gmm`)
- `--output`: Output clustering mask file path (default: `cluster_map_1-8.nii.gz` in the same directory as ADC file)

## Clustering Algorithm

### 1. Feature-wise Binary Classification
Each MRI parameter (ADC, CBV, KTRANS) is binary classified into high/low values using GMM:
- **High value cluster**: 1
- **Low value cluster**: 0

### 2. Combined Cluster Generation
Eight habitat clusters are created by weighted combination of the three feature binary labels:

```
Final Cluster = ADC×4 + CBV×2 + KTRANS×1 + 1
```

### 3. Cluster Interpretation
| Cluster | ADC | CBV | KTRANS | Characteristics |
|---------|-----|-----|--------|-----------------|
| 1 | Low | Low | Low | Low perfusion, low diffusion, low permeability |
| 2 | Low | Low | High | Low perfusion, low diffusion, high permeability |
| 3 | Low | High | Low | High perfusion, low diffusion, low permeability |
| 4 | Low | High | High | High perfusion, low diffusion, high permeability |
| 5 | High | Low | Low | Low perfusion, high diffusion, low permeability |
| 6 | High | Low | High | Low perfusion, high diffusion, high permeability |
| 7 | High | High | Low | High perfusion, high diffusion, low permeability |
| 8 | High | High | High | High perfusion, high diffusion, high permeability |

## Output Results

### Console Output Example
```
Loading NIfTI files...
Tumor mask contains 15432 voxels
Applying GMM clustering...
Processing adc...
adc: High cluster = 8234 voxels, Low cluster = 7198 voxels
Processing cbv...
cbv: High cluster = 6789 voxels, Low cluster = 8643 voxels
Processing ktrans...
ktrans: High cluster = 9876 voxels, Low cluster = 5556 voxels
Creating combined clustering map...

Cluster distribution:
Cluster 1: 1543 voxels (10.0%)
Cluster 2: 2134 voxels (13.8%)
Cluster 3: 1876 voxels (12.2%)
Cluster 4: 2345 voxels (15.2%)
Cluster 5: 1987 voxels (12.9%)
Cluster 6: 2234 voxels (14.5%)
Cluster 7: 1789 voxels (11.6%)
Cluster 8: 1524 voxels (9.9%)

Saving clustering map to results/cluster_map_1-8.nii.gz
Clustering completed successfully!
```

### Output Files
- `cluster_map_1-8.nii.gz`: Clustering mask file
  - Background region: 0
  - Clusters 1-8: respective cluster number values
  - Data type: uint8
  - Same affine and header information as original ADC file

## Analysis Tools

### Cluster Volume Analysis

Calculate the volume (mm³) of each habitat cluster from the generated clustering mask.

#### Usage
```bash
python analysis/cluster_volume.py \
    --input path/to/cluster_map_1-8.nii.gz \
    --output results/cluster_volumes.csv
```

#### Output Example
```
Cluster Volume Analysis:
============================================================
Voxel dimensions: 1.00 × 1.00 × 1.00 mm
Voxel volume: 1.000 mm³

Cluster 1:   1234 voxels =   1234.0 mm³
Cluster 2:   2345 voxels =   2345.0 mm³
Cluster 3:   1876 voxels =   1876.0 mm³
...
============================================================
Total tumor volume:  15432 voxels =  15432.0 mm³

Detailed Results:
--------------------------------------------------------------------------------
C1: Low-Low-Low     |   1234 voxels |   1234.0 mm³ |   8.0%
C2: Low-Low-High    |   2345 voxels |   2345.0 mm³ |  15.2%
C3: Low-High-Low    |   1876 voxels |   1876.0 mm³ |  12.2%
```

### Histogram Feature Analysis

Extract histogram features from ADC, CBV, and KTRANS and perform predictions using pre-trained linear regression models.

#### Usage
```bash
python analysis/histogram_features.py \
    --adc path/to/adc.nii.gz \
    --cbv path/to/cbv.nii.gz \
    --ktrans path/to/ktrans.nii.gz \
    --mask path/to/WT_tumor_mask.nii.gz \
    --model-dir models/histogram_feature \
    --output results/histogram_results.csv
```

#### Extracted Features
From each imaging modality (ADC, CBV, KTRANS):
- **Mean**: Average value
- **Median**: Median value  
- **P10**: 10th percentile
- **P90**: 90th percentile

A total of 12 histogram features are extracted and used as input for the linear regression model.

## License

This project was developed for research purposes.

## Contact

**Email**: seo01070107@gmail.com

Please register an issue if you have any questions about the code.
