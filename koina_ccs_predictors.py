import pandas as pd
import numpy as np
import argparse
import sys
import re
from pathlib import Path

#Run different CCS predictors through the koina server to compare against our model later

try:
    from koinapy import Koina
except ImportError:
    print("Error: koinapy not found. Please install it with: pip install koinapy")
    sys.exit(1)

def clean_peptide_sequence(sequence):
    """
    Clean peptide sequence by removing all non-alphabetic characters.
    
    Parameters:
    sequence : str - Raw peptide sequence that may contain PTM indicators
    
    Returns:
    str - Cleaned peptide sequence containing only alphabetic characters
    """
    if pd.isna(sequence):
        return sequence
    
    # Convert to string if not already
    sequence = str(sequence)
    
    # Remove all non-alphabetic characters (keeping only A-Z, a-z)
    sequence = re.sub(r'[^A-Za-z]', '', sequence)
    
    # Convert to uppercase for consistency
    sequence = sequence.upper()
    
    return sequence

def validate_input_file(file_path):
    """
    Validate that the input file exists and has required columns.
    
    Parameters:
    file_path : str - Path to the input TSV file
    
    Returns:
    pandas.DataFrame - Loaded data if valid
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Load the TSV file
    try:
        data = pd.read_csv(file_path, sep='\t')
        data = data.rename(columns = {'Peptide':'peptide_sequences',
        'Charge':'precursor_charges'})
    except Exception as e:
        raise ValueError(f"Error reading TSV file: {e}")
    
    # Check for required columns
    required_columns = ['peptide_sequences', 'precursor_charges']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty data
    if data.empty:
        raise ValueError("Input file is empty")
    
    # Check for missing values in required columns
    for col in required_columns:
        if data[col].isna().any():
            print(f"Warning: Found missing values in column '{col}'. These rows will be removed.")
            data = data.dropna(subset=[col])
    
    if data.empty:
        raise ValueError("No valid data remaining after removing missing values")
    
    # Clean peptide sequences
    print("Cleaning peptide sequences...")
    data['peptide_sequences_original'] = data['peptide_sequences'].copy()
    data['peptide_sequences'] = data['peptide_sequences'].apply(clean_peptide_sequence)
    
    # Show some examples of cleaned sequences
    print("\nExamples of peptide sequence cleaning:")
    sample_data = data[['peptide_sequences_original', 'peptide_sequences']].head(10)
    for idx, row in sample_data.iterrows():
        if row['peptide_sequences_original'] != row['peptide_sequences']:
            print(f"Original: {row['peptide_sequences_original']} -> Cleaned: {row['peptide_sequences']}")
    
    # Remove rows with empty peptide sequences after cleaning
    empty_sequences = data['peptide_sequences'].str.len() == 0
    if empty_sequences.any():
        print(f"Warning: Found {empty_sequences.sum()} empty peptide sequences after cleaning. Removing them.")
        data = data[~empty_sequences]
    
    if data.empty:
        raise ValueError("No valid data remaining after cleaning sequences")
    
    print(f"Final data shape after cleaning: {data.shape}")
    
    return data

def run_koina_prediction(model_name, input_data):
    """
    Run Koina model prediction on the input data.
    
    Parameters:
    model_name : str - Name of the Koina model to use
    input_data : pandas.DataFrame - Input data with peptide_sequences and precursor_charges
    
    Returns:
    pandas.DataFrame - Model predictions
    """
    try:
        # Initialize the model
        print(f"Initializing {model_name} model...")
        model = Koina(model_name, "koina.wilhelmlab.org:443")
        
        # Check model inputs
        print(f"Model inputs: {model.model_inputs}")
        
        # Prepare inputs for prediction
        inputs = pd.DataFrame()
        inputs['peptide_sequences'] = input_data['peptide_sequences'].values
        inputs['precursor_charges'] = input_data['precursor_charges'].values
        
        # Run prediction
        print(f"Running predictions for {len(inputs)} sequences...")
        predictions = model.predict(inputs)
        
        return predictions
        
    except Exception as e:
        raise RuntimeError(f"Error running Koina prediction: {e}")

def save_predictions(predictions, output_file, input_data):
    """
    Save predictions to TSV file along with input data.
    
    Parameters:
    predictions : pandas.DataFrame - Model predictions
    output_file : str - Output file path
    input_data : pandas.DataFrame - Original input data
    """
    try:
        # Combine input data with predictions
        result_df = input_data.copy()
        
        # Add prediction columns
        for col in predictions.columns:
            result_df[f'predicted_{col}'] = predictions[col].values
        
        # Reorganize columns to show original and cleaned sequences
        if 'peptide_sequences_original' in result_df.columns:
            # Reorder columns to show original, cleaned, then predictions
            cols = ['peptide_sequences_original', 'peptide_sequences', 'precursor_charges']
            pred_cols = [col for col in result_df.columns if col.startswith('predicted_')]
            other_cols = [col for col in result_df.columns if col not in cols + pred_cols]
            result_df = result_df[cols + other_cols + pred_cols]
        
        # Save to TSV
        result_df.to_csv(output_file, sep='\t', index=False)
        print(f"Predictions saved to: {output_file}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total sequences processed: {len(result_df)}")
        print(f"Prediction columns added: {list(predictions.columns)}")
        
        # Show first few results
        print(f"\nFirst 5 predictions:")
        print(result_df.head())
        
    except Exception as e:
        raise RuntimeError(f"Error saving predictions: {e}")

def main():
    """
    Main function to run the Koina CCS prediction pipeline.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run Koina model predictions on peptide sequences from TSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python koina_ccs_predictor.py input_data.tsv IM2Deep
  python koina_ccs_predictor.py peptides.tsv IM2Deep --output my_predictions.tsv
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input TSV file containing peptide_sequences and precursor_charges columns'
    )
    
    parser.add_argument(
        'model_name',
        help='Name of the Koina model to use (e.g., IM2Deep)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output file path (default: <model_name>_ccs_predictions.tsv)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Validate and load input file
        print(f"Loading input file: {args.input_file}")
        input_data = validate_input_file(args.input_file)
        print(f"Loaded {len(input_data)} sequences from input file")
        
        if args.verbose:
            print(f"Input data columns: {list(input_data.columns)}")
            print(f"First few sequences:")
            print(input_data.head())
        
        # Determine output file name
        if args.output:
            output_file = args.output
        else:
            output_file = f"{args.model_name}_ccs_predictions.tsv"
        
        # Run predictions
        predictions = run_koina_prediction(args.model_name, input_data)
        
        # Save results
        save_predictions(predictions, output_file, input_data)
        
        print(f"\nPrediction pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
