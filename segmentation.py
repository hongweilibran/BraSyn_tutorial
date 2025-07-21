import os
import argparse
from pathlib import Path
from brats import AdultGliomaPreTreatmentSegmenter, MeningiomaSegmenter, MetastasesSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms, MeningiomaAlgorithms, MetastasesAlgorithms

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run brain tumor segmentation on medical images.')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing patient folders')
    parser.add_argument('-o', '--output', required=True, help='Output directory for segmentation results')
    parser.add_argument('--gpu', default='0', help='GPU device number to use (default: 0)')
    args = parser.parse_args()

    # Define input and output directories from arguments
    input_dir = args.input
    output_dir = args.output
    cuda_device = args.gpu

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create dictionary mapping folder type patterns to appropriate segmenters and algorithms
    segmenters = {
        'GLI': {
            'segmenter': AdultGliomaPreTreatmentSegmenter(
                algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1, 
                cuda_devices=cuda_device
            )
        },
        'MEN': {
            'segmenter': MeningiomaSegmenter(
                algorithm=MeningiomaAlgorithms.BraTS23_1, 
                cuda_devices=cuda_device
            )
        },
        'MET': {
            'segmenter': MetastasesSegmenter(
                algorithm=MetastasesAlgorithms.BraTS23_1,
                cuda_devices=cuda_device
            )
        }
    }

    # Get all subfolders in the input directory
    input_path = Path(input_dir)
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]

    # Process each subfolder
    for subfolder in subfolders:
        folder_name = subfolder.name
        
        # Determine which segmenter to use based on folder name
        segmenter_key = None
        for key in segmenters.keys():
            if key in folder_name:
                segmenter_key = key
                break
        
        # Skip if folder doesn't match any of our target types
        if segmenter_key is None:
            print(f"Skipping folder {folder_name} - no matching segmentation algorithm")
            continue
        
        # Get the appropriate segmenter
        segmenter = segmenters[segmenter_key]['segmenter']
        
        # Get the input file paths
        t1c_file = subfolder / f"{folder_name}-t1c.nii.gz"
        t1n_file = subfolder / f"{folder_name}-t1n.nii.gz"
        t2f_file = subfolder / f"{folder_name}-t2f.nii.gz"
        t2w_file = subfolder / f"{folder_name}-t2w.nii.gz"
        
        # Check if all required files exist
        if not all(f.exists() for f in [t1c_file, t1n_file, t2f_file, t2w_file]):
            print(f"Skipping folder {folder_name} - missing required files")
            continue
        
        # Define output file path directly in the output directory
        output_file = Path(output_dir) / f"{folder_name}-seg.nii.gz"
        
        print(f"Processing {folder_name} with {segmenter_key} segmenter...")
        
        # Run the segmentation
        segmenter.infer_single(
            t1c=str(t1c_file),
            t1n=str(t1n_file),
            t2f=str(t2f_file),
            t2w=str(t2w_file),
            output_file=str(output_file)
        )
        
        print(f"Completed segmentation for {folder_name}")

    print("All segmentations complete!")

if __name__ == "__main__":
    main()