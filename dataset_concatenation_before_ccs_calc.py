#Concatenate and process all the files created above

import pandas as pd
import re
import os
from pathlib import Path

def concatenate_mhc_files(file_paths, required_columns, output_file):
    """
    Concatenate multiple TSV files containing MHC data and save to a combined file.

    Parameters:
    -----------
    file_paths : list
        List of file paths to concatenate
    required_columns : list
        List of column names to extract from each file
    output_file : str
        Path for the output combined file

    Returns:
    --------
    pd.DataFrame
        Combined dataframe with all data
    """

    combined_data = []
    file_info = []

    for file_path in file_paths:
        try:
            # Extract dataset ID using regex
            dataset_match = re.search(r'(PXD\d+)', file_path, re.IGNORECASE)
            dataset_id = dataset_match.group(1).upper() if dataset_match else f"Unknown_{len(file_info)+1}"

            print(f"Processing {dataset_id}: {file_path}")

            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue

            # Read the TSV file
            df = pd.read_csv(file_path, sep="\t")

            # Check if all required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {dataset_id}: {missing_cols}")
                # Use only available columns
                available_cols = [col for col in required_columns if col in df.columns]
            else:
                available_cols = required_columns

            # Select only required columns
            df_subset = df[available_cols].copy()

            # Add source dataset column
            df_subset['Dataset'] = dataset_id

            # Add file info for summary
            file_info.append({
                'Dataset': dataset_id,
                'File': os.path.basename(file_path),
                'Rows': len(df_subset),
                'Columns': len(available_cols)
            })

            combined_data.append(df_subset)
            print(f"  - Loaded {len(df_subset):,} rows with {len(available_cols)} columns")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    if not combined_data:
        raise ValueError("No files were successfully processed!")

    # Concatenate all dataframes
    print("\nCombining all datasets...")
    combined_df = pd.concat(combined_data, ignore_index=True, sort=False)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to TSV file
    combined_df.to_csv(output_file, sep="\t", index=False)

    # Print summary
    print(f"\n{'='*50}")
    print("CONCATENATION SUMMARY")
    print(f"{'='*50}")

    for info in file_info:
        print(f"{info['Dataset']:>10}: {info['Rows']:>8,} rows")

    print(f"{'='*20}")
    print(f"{'TOTAL':>10}: {len(combined_df):>8,} rows")
    print(f"\nCombined data saved to: {output_file}")
    print(f"Final dataset shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")

    return combined_df

# Usage example with your specific file paths
def main():
    # Define your file paths
    file_paths = [
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD038782/pxd038782_complete.tsv",
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD035344/pxd035344_complete.tsv",
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD022194/pxd022194_complete.tsv",
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD038273/pxd038273_complete.tsv"
    ]

    # Define required columns
    req_cols = ['Peptide', 'Mass', 'm/z', 'Length', '1/k0 Range']

    # Define output file path
    output_file = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/combined_mhc1_data.tsv"

    # Run the concatenation
    try:
        combined_df = concatenate_mhc_files(file_paths, req_cols, output_file)

        # Optional: Display basic statistics
        print(f"\n{'='*50}")
        print("BASIC STATISTICS")
        print(f"{'='*50}")
        print(f"Datasets: {combined_df['Dataset'].nunique()}")
        print(f"Total peptides: {len(combined_df):,}")

        # Show dataset distribution
        print("\nDataset distribution:")
        dataset_counts = combined_df['Dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count:,} ({count/len(combined_df)*100:.1f}%)")

    except Exception as e:
        print(f"Error: {str(e)}")

# Alternative function for direct usage (without the main wrapper)
def quick_concatenate():
    """Quick function to run the concatenation directly"""

    file_paths = [
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD038782/pxd038782_complete.tsv",
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD035344/pxd035344_complete.tsv",
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD022194/pxd022194_complete.tsv",
        "/content/drive/MyDrive/Colab_CCS_results/MHC_1/raw_data/New_data/PXD038273/pxd038273_complete.tsv"
    ]

    req_cols = ['Peptide', 'Mass', 'm/z', 'Length', '1/k0 Range']
    output_file = "/content/drive/MyDrive/Colab_CCS_results/MHC_1/combined_mhc_data.tsv"

    return concatenate_mhc_files(file_paths, req_cols, output_file)

# Run the function
if __name__ == "__main__":
    main()
