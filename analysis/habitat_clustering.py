import os
import numpy as np
import nibabel as nib
import joblib
from pathlib import Path
import argparse


def generate_clustering_mask(adc_path, cbv_path, ktrans_path, mask_path, 
                            model_dir=".models/habitat/gmm", 
                            output_path=None):
    """
    Generate clustering mask using pre-trained GMM models.
    
    Parameters:
    -----------
    adc_path : str
        Path to ADC NIfTI file
    cbv_path : str
        Path to CBV NIfTI file  
    ktrans_path : str
        Path to KTRANS NIfTI file
    mask_path : str
        Path to WT tumor mask NIfTI file
    model_dir : str
        Directory containing GMM model files
    output_path : str
        Path to save clustering mask (optional)
        
    Returns:
    --------
    cluster_map : numpy.ndarray
        3D clustering mask with values 1-8
    """
    
    # Define features and model file format
    features = ['adc', 'cbv', 'ktrans']
    model_fmt = 'gmm_{feat}_n{n}.pkl'
    
    # Build model paths
    model_paths = {
        feat: os.path.join(model_dir, model_fmt.format(feat=feat, n=2))
        for feat in features
    }
    
    # Check if all model files exist
    for feat, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GMM model file not found: {model_path}")
    
    # Load NIfTI images
    print("Loading NIfTI files...")
    img_adc = nib.load(adc_path)
    img_cbv = nib.load(cbv_path)
    img_ktrans = nib.load(ktrans_path)
    img_mask = nib.load(mask_path)
    
    # Get data arrays
    adc = img_adc.get_fdata()
    cbv = img_cbv.get_fdata()
    ktrans = img_ktrans.get_fdata()
    mask = img_mask.get_fdata().astype(bool)
    
    print(f"Tumor mask contains {np.sum(mask)} voxels")
    
    # Extract tumor region values for each feature
    vals = {
        'adc': adc[mask],
        'cbv': cbv[mask], 
        'ktrans': ktrans[mask]
    }
    
    # Apply GMM clustering for each feature
    print("Applying GMM clustering...")
    bin_labels = {}
    
    for feat in features:
        print(f"Processing {feat}...")
        
        # Load GMM model
        gmm = joblib.load(model_paths[feat])
        
        # Prepare data for prediction
        x = vals[feat].reshape(-1, 1)
        
        # Predict clusters
        predictions = gmm.predict(x)
        cluster_means = gmm.means_.flatten()
        
        # Create binary labels (high values = 1, low values = 0)
        if cluster_means[0] > cluster_means[1]:
            bin_labels[feat] = np.where(predictions == 0, 1, 0)
        else:
            bin_labels[feat] = np.where(predictions == 1, 1, 0)
        
        print(f"{feat}: High cluster = {np.sum(bin_labels[feat])} voxels, "
              f"Low cluster = {len(bin_labels[feat]) - np.sum(bin_labels[feat])} voxels")
    
    # Combine binary labels into single clustering map (1-8)
    print("Creating combined clustering map...")
    combined = (
        bin_labels['adc'] * 4 +
        bin_labels['cbv'] * 2 +
        bin_labels['ktrans'] * 1
    ) + 1  # Convert 0-7 to 1-8
    
    # Create full-size clustering map
    cluster_map = np.zeros_like(adc, dtype=np.uint8)
    cluster_map[mask] = combined
    
    # Print cluster distribution
    unique, counts = np.unique(combined, return_counts=True)
    print("\nCluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} voxels ({count/len(combined)*100:.1f}%)")
    
    # Save clustering map if output path is provided
    if output_path:
        print(f"Saving clustering map to {output_path}")
        out_img = nib.Nifti1Image(cluster_map, img_adc.affine, img_adc.header)
        nib.save(out_img, output_path)
    
    return cluster_map


def main():
    parser = argparse.ArgumentParser(description="Generate clustering mask using GMM models")
    parser.add_argument("--adc", required=True, help="Path to ADC NIfTI file")
    parser.add_argument("--cbv", required=True, help="Path to CBV NIfTI file")
    parser.add_argument("--ktrans", required=True, help="Path to KTRANS NIfTI file")
    parser.add_argument("--mask", required=True, help="Path to WT tumor mask NIfTI file")
    parser.add_argument("--model-dir", default=".models/habitat/gmm", 
                       help="Directory containing GMM model files")
    parser.add_argument("--output", help="Output path for clustering mask")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        base_dir = os.path.dirname(args.adc)
        args.output = os.path.join(base_dir, "cluster_map_1-8.nii.gz")
    
    try:
        cluster_map = generate_clustering_mask(
            adc_path=args.adc,
            cbv_path=args.cbv, 
            ktrans_path=args.ktrans,
            mask_path=args.mask,
            model_dir=args.model_dir,
            output_path=args.output,
        )
        
        print(f"\nClustering completed successfully!")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
