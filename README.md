# The Brandenburg Dataset

The Brandeburg Dataset comprises camera trap videos of European wildife.

## What can be found here?

This is a PyTorch dataset class for the Brandenburg Dataset.

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
