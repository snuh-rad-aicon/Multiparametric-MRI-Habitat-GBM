"""
Cluster volume analysis for habitat clustering results.
Calculate the volume (mm³) of each habitat cluster.
"""

import numpy as np
import nibabel as nib
import pandas as pd
import argparse
import os

def get_voxel_volume(nifti_img):
    """
    Calculate voxel volume from NIfTI image affine matrix.
    
    Parameters:
    -----------
    nifti_img : nibabel.Nifti1Image
        NIfTI image object
        
    Returns:
    --------
    voxel_volume : float
        Volume of single voxel in mm³
    """
    
    # Get voxel dimensions from affine matrix
    affine = nifti_img.affine
    voxel_dims = np.abs(np.diag(affine)[:3])  # x, y, z dimensions
    voxel_volume = np.prod(voxel_dims)  # mm³
    
    return voxel_volume


def calculate_cluster_volumes(cluster_path, output_path=None, verbose=True):
    """
    Calculate volume of each cluster in the habitat clustering mask.
    
    Parameters:
    -----------
    cluster_path : str
        Path to cluster map NIfTI file
    output_path : str, optional
        Path to save results CSV file
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with cluster volumes
    """
    
    # Load cluster mask
    if verbose:
        print(f"Loading cluster mask: {cluster_path}")
    
    cluster_img = nib.load(cluster_path)
    cluster_data = cluster_img.get_fdata().astype(int)
    
    # Get voxel volume
    voxel_volume = get_voxel_volume(cluster_img)
    voxel_dims = np.abs(np.diag(cluster_img.affine)[:3])
    
    if verbose:
        print(f"Voxel dimensions: {voxel_dims[0]:.2f} × {voxel_dims[1]:.2f} × {voxel_dims[2]:.2f} mm")
        print(f"Voxel volume: {voxel_volume:.3f} mm³")
    
    # Define cluster labels
    cluster_labels = [
        'C1: Low-Low-Low',      # ADC=Low, CBV=Low, KTRANS=Low
        'C2: Low-Low-High',     # ADC=Low, CBV=Low, KTRANS=High  
        'C3: Low-High-Low',     # ADC=Low, CBV=High, KTRANS=Low
        'C4: Low-High-High',    # ADC=Low, CBV=High, KTRANS=High
        'C5: High-Low-Low',     # ADC=High, CBV=Low, KTRANS=Low
        'C6: High-Low-High',    # ADC=High, CBV=Low, KTRANS=High
        'C7: High-High-Low',    # ADC=High, CBV=High, KTRANS=Low
        'C8: High-High-High'    # ADC=High, CBV=High, KTRANS=High
    ]
    
    # Calculate volumes for each cluster
    results = []
    total_tumor_voxels = 0
    total_tumor_volume = 0
    
    if verbose:
        print("\nCluster Volume Analysis:")
        print("=" * 60)
    
    for cluster_id in range(1, 9):
        # Count voxels for this cluster
        cluster_mask = cluster_data == cluster_id
        voxel_count = np.sum(cluster_mask)
        volume_mm3 = voxel_count * voxel_volume
        
        # Calculate percentage of total tumor
        total_tumor_voxels += voxel_count
        total_tumor_volume += volume_mm3
        
        results.append({
            'cluster_id': cluster_id,
            'cluster_label': cluster_labels[cluster_id-1],
            'voxel_count': voxel_count,
            'volume_mm3': volume_mm3
        })
        
        if verbose:
            print(f"Cluster {cluster_id}: {voxel_count:6d} voxels = {volume_mm3:8.1f} mm³")
    
    # Calculate percentages
    for result in results:
        if total_tumor_volume > 0:
            result['percentage'] = (result['volume_mm3'] / total_tumor_volume) * 100
        else:
            result['percentage'] = 0.0
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("=" * 60)
        print(f"Total tumor volume: {total_tumor_voxels:6d} voxels = {total_tumor_volume:8.1f} mm³")
        print("\nDetailed Results:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            if row['voxel_count'] > 0:
                print(f"{row['cluster_label']:20s} | "
                      f"{row['voxel_count']:6d} voxels | "
                      f"{row['volume_mm3']:8.1f} mm³ | "
                      f"{row['percentage']:5.1f}%")
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return results_df





def main():
    parser = argparse.ArgumentParser(description="Analyze habitat cluster volumes")
    parser.add_argument("--input", required=True, help="Path to cluster mask file")
    parser.add_argument("--output", help="Output CSV file path")
    
    args = parser.parse_args()
    
    import os
    input_path = args.input
    
    try:
        if os.path.isfile(input_path):
            print("Analyzing cluster volumes...")
            results_df = calculate_cluster_volumes(
                cluster_path=input_path,
                output_path=args.output,
                verbose=True
            )
        else:
            print(f"Error: {input_path} is not a valid file")
            return 1
            
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 