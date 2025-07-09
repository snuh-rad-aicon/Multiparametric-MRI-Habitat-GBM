"""
Histogram feature extraction and linear regression for habitat clustering.
Extract percentile, mean, median values from ADC, CBV, KTRANS and predict using linear models.
"""

import numpy as np
import nibabel as nib
import pandas as pd
import joblib
import os
import argparse
import warnings

# Suppress scikit-learn version compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def extract_histogram_features(adc_path, cbv_path, ktrans_path, mask_path):
    """
    Extract histogram features from ADC, CBV, KTRANS within tumor mask.
    
    Parameters:
    -----------
    adc_path : str
        Path to ADC NIfTI file
    cbv_path : str
        Path to CBV NIfTI file
    ktrans_path : str
        Path to KTRANS NIfTI file
    mask_path : str
        Path to tumor mask NIfTI file
        
    Returns:
    --------
    features_dict : dict
        Dictionary with extracted features
    """
    
    print("Loading images...")
    
    # Load images
    adc_img = nib.load(adc_path)
    cbv_img = nib.load(cbv_path)
    ktrans_img = nib.load(ktrans_path)
    mask_img = nib.load(mask_path)
    
    # Get data arrays
    adc_data = adc_img.get_fdata()
    cbv_data = cbv_img.get_fdata()
    ktrans_data = ktrans_img.get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)
    
    print(f"Tumor mask contains {np.sum(mask_data)} voxels")
    
    # Extract tumor region values
    adc_tumor = adc_data[mask_data]
    cbv_tumor = cbv_data[mask_data]
    ktrans_tumor = ktrans_data[mask_data]
    
    features = {}
    
    # Calculate features for each modality
    for name, values in [('adc', adc_tumor), ('cbv', cbv_tumor), ('ktrans', ktrans_tumor)]:
        features[f'{name}_mean'] = np.mean(values)
        features[f'{name}_median'] = np.median(values)
        features[f'{name}_p10'] = np.percentile(values, 10)
        features[f'{name}_p90'] = np.percentile(values, 90)
        
        print(f"{name.upper()}: mean={features[f'{name}_mean']:.3f}, "
              f"median={features[f'{name}_median']:.3f}, "
              f"p10={features[f'{name}_p10']:.3f}, "
              f"p90={features[f'{name}_p90']:.3f}")
    
    return features


def predict_with_models(features_dict, model_dir="models/histogram_feature"):
    """
    Predict using linear regression models in histogram_feature directory.
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary with extracted features
    model_dir : str
        Directory containing linear regression models
        
    Returns:
    --------
    predictions : dict
        Dictionary with model predictions
    """
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return {}
    
    # Find all model files in subdirectories
    model_files = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.joblib'):
                relative_path = os.path.relpath(os.path.join(root, file), model_dir)
                model_files.append(relative_path)
    
    if not model_files:
        print(f"No .joblib model files found in {model_dir}")
        return {}
    
    print(f"Found {len(model_files)} model files")
    
    predictions = {}
    
    print("\nMaking predictions:")
    print("-" * 40)
    
    for model_file in sorted(model_files):
        model_path = os.path.join(model_dir, model_file)
        # Create a cleaner model name for display
        model_name = model_file.replace('.joblib', '').replace('/', '_').replace('\\', '_')
        
        try:
            # Load model and scaler
            model_scaler = joblib.load(model_path)
            model = model_scaler['model']
            scaler = model_scaler['scaler']
            
            # Determine feature set based on model type
            if 'multivariable_regression' in model_file:
                feature_names = []
                if 'adc' in model_name or 'all_modalities' in model_name:
                    feature_names.append('adc_p10')
                if 'cbv' in model_name or 'all_modalities' in model_name:
                    feature_names.append('cbv_mean')
                    feature_names.append('cbv_p90')
                if 'ktrans' in model_name or 'all_modalities' in model_name:
                    feature_names.append('ktrans_mean')
                    feature_names.append('ktrans_p90')
            else:
                # Univariable regression uses single specific feature
                feature_name = model_name.split('WT_')[1] if 'WT_' in model_name else model_name
                feature_names = [feature_name]
            
            # Prepare feature vector
            feature_vector = [features_dict.get(name, 0) for name in feature_names]
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaler
            feature_array_scaled = scaler.transform(feature_array)
            
            # Make prediction and get probability
            prediction = model.predict(feature_array_scaled)[0]
            prediction_proba = model.predict_proba(feature_array_scaled)[0]
            
            # Interpret prediction: 1 = Histological GBM, 0 = Molecular GBM
            gbm_type = "Histological GBM" if prediction == 1 else "Molecular GBM"
            
            # Get probability for the predicted class
            prob_molecular = prediction_proba[0]  # Probability of class 0 (Molecular GBM)
            prob_histological = prediction_proba[1]  # Probability of class 1 (Histological GBM)
            
            # Store results
            predictions[f"{model_name}_prediction"] = gbm_type
            predictions[f"{model_name}_prob_molecular"] = prob_molecular
            predictions[f"{model_name}_prob_histological"] = prob_histological
            
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
            continue
    
    return predictions


def analyze_histogram_features(adc_path, cbv_path, ktrans_path, mask_path, 
                              model_dir="/media/data/mschoi/project/MolGBM_github/models/histogram_feature",
                              output_path=None):
    """
    Complete analysis: extract features and make predictions.
    
    Parameters:
    -----------
    adc_path : str
        Path to ADC NIfTI file
    cbv_path : str
        Path to CBV NIfTI file
    ktrans_path : str
        Path to KTRANS NIfTI file
    mask_path : str
        Path to tumor mask NIfTI file
    model_dir : str
        Directory containing linear regression models
    output_path : str, optional
        Path to save results CSV
        
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with features and predictions
    """
    
    # Extract histogram features
    features = extract_histogram_features(adc_path, cbv_path, ktrans_path, mask_path)
    
    # Make predictions
    predictions = predict_with_models(features, model_dir)
    
    # Combine results
    results = {**features, **predictions}
    
    # Create DataFrame
    results_df = pd.DataFrame([results])
    
    print("\nResults Summary:")
    print("=" * 50)
    
    # Display features
    print("Histogram Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.4f}")
    
    # Display predictions
    if predictions:
        print("\nModel Predictions:")
        # Group predictions by model
        model_names = set()
        for key in predictions.keys():
            if '_prediction' in key:
                model_names.add(key.replace('_prediction', ''))

        for model in sorted(model_names):
            gbm_type = predictions.get(f"{model}_prediction", "N/A")
            prob_molecular = predictions.get(f"{model}_prob_molecular", 0)
            prob_histological = predictions.get(f"{model}_prob_histological", 0)
            raw_pred = predictions.get(f"{model}_raw_prediction", "N/A")
            
            print(f"  {model}:")
            print(f"    Prediction: {gbm_type}")
            print(f"    Probabilities: Molecular={prob_molecular:.4f}, Histological={prob_histological:.4f}")
            print()
    
    # Save results
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Extract histogram features and predict using linear models")
    parser.add_argument("--adc", required=True, help="Path to ADC NIfTI file")
    parser.add_argument("--cbv", required=True, help="Path to CBV NIfTI file")
    parser.add_argument("--ktrans", required=True, help="Path to KTRANS NIfTI file")
    parser.add_argument("--mask", required=True, help="Path to tumor mask NIfTI file")
    parser.add_argument("--model-dir", 
                       default="models/histogram_feature",
                       help="Directory containing linear regression models")
    parser.add_argument("--output", help="Output CSV file path")
    
    args = parser.parse_args()
    
    try:
        results_df = analyze_histogram_features(
            adc_path=args.adc,
            cbv_path=args.cbv,
            ktrans_path=args.ktrans,
            mask_path=args.mask,
            model_dir=args.model_dir,
            output_path=args.output
        )
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 