# The Brandenburg Dataset

The Brandeburg Dataset comprises camera trap videos of European wildife.

## What can be found here?

brandenburg-dataset: A PyTorch dataset class for the Brandenburg Dataset.
brandenburg-tracking: Python scripts to generate tracking information
data: Directory in which to place the Brandenburg dataset

## Basic pre-processing
This will sort the data from the Brandenburg dataset and generate tracklet information for all animal classes. 
Parameters can be adjusted from within the script. 
The `--make_video` option will generate videos shwoing the bounding boxes generated as part of the tracking.

```bash
python brandenburg_tracklet_processing.py {OPTIONAL --make_video}
```

## Basic usage

```bash
dataset = BrandenburgDataset(
    data_dir="path/to/data", 
    sequence_len=5,
    sample_itvl=1,
    stride=5,
    transform=transform
)

loader = DataLoader(dataset)

for sample, label in loader:
    # Training loop below
```
