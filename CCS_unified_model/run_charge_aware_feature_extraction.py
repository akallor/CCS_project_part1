#!/usr/bin/env python3
"""
Run ESM-2 feature extraction with engineered features for both training and test data.
This script automates the process of extracting ESM-2 features and engineered residue features
for both training and testing datasets.
"""

import os
from charge_aware_esm_feature_extraction import ESMFeatureExtractor

def extract_features():
    """Extract ESM-2 features and engineered features for both training and test data."""
    
    print("=" * 80)
    print("ESM-2 FEATURE EXTRACTION WITH ENGINEERED FEATURES")
    print("=" * 80)
    
    # Configuration
    config = {
        'model_type': "esm2_t6_8M_UR50D",  # Smallest ESM-2 model
        'aggregation_strategy': "global_mean",
        'batch_size': 5000,  # Reduced batch size to save memory
    }
    
    # Data paths
    data_paths = {
        'training': {
            'input': '/hpc/shared/uu_immunopeptidomics/ccs_data/train_1_new_charge1_lab.tsv',
            'output': '/hpc/shared/uu_immunopeptidomics/ccs_data/esm_features_train_chg1.pt'
        },
        'testing': {
            'input': '/hpc/shared/uu_immunopeptidomics/ccs_data/test_1_new_charge1_lab.tsv',
            'output': '/hpc/shared/uu_immunopeptidomics/ccs_data/esm_features_test_chg1.pt'
        }
    }
    
    print(f"Model: {config['model_type']}")
    print(f"Aggregation Strategy: {config['aggregation_strategy']}")
    print(f"Batch Size: {config['batch_size']}")
    print("=" * 80)
    
    # Initialize feature extractor
    feature_extractor = ESMFeatureExtractor(
        model_type=config['model_type'],
        aggregation_strategy=config['aggregation_strategy'],
        batch_size=config['batch_size']
    )
    
    # Extract features for training data
    print("\n" + "=" * 60)
    print("EXTRACTING TRAINING FEATURES")
    print("=" * 60)
    
    if os.path.exists(data_paths['training']['input']):
        feature_extractor.extract_features_from_file(
            input_file_path=data_paths['training']['input'],
            output_file_path=data_paths['training']['output'],
            sequence_column=1,  # Column containing peptide sequences
            delimiter='\t',
            skip_header=True
        )
        print(f"✅ Training features saved to: {data_paths['training']['output']}")
    else:
        print(f"❌ Training data file not found: {data_paths['training']['input']}")
        return False
    
    # Extract features for test data
    print("\n" + "=" * 60)
    print("EXTRACTING TEST FEATURES")
    print("=" * 60)
    
    if os.path.exists(data_paths['testing']['input']):
        feature_extractor.extract_features_from_file(
            input_file_path=data_paths['testing']['input'],
            output_file_path=data_paths['testing']['output'],
            sequence_column=1,  # Column containing peptide sequences
            delimiter='\t',
            skip_header=True
        )
        print(f"✅ Test features saved to: {data_paths['testing']['output']}")
    else:
        print(f"❌ Test data file not found: {data_paths['testing']['input']}")
        return False
    
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETED!")
    print("=" * 80)
    print("Next steps:")
    print("1. Run CCS prediction with hybrid ESM-2 + RNN model:")
    print("   python rnn_charge_aware_ccs_predictor.py")
    print("2. The system will automatically use extracted features")
    print("=" * 80)
    
    return True

def main():
    """Main function."""
    try:
        success = extract_features()
        if success:
            print("\n🎉 Feature extraction completed successfully!")
        else:
            print("\n💥 Feature extraction failed. Please check your data files.")
    except Exception as e:
        print(f"\n❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
