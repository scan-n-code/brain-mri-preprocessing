# Brain MRI Image Contrast Enhancement & Digital Preprocessing

A small beginner project that improves the visibility of structures in raw brain MRI slices
using simple, classical image processing.

## Why this project
In brain MRI, the borders between different structures can be hard to see in a raw image.
I wanted to understand how basic preprocessing can make these borders clearer, and how to
check this with simple measurements instead of only by eye.

## Pipeline
1. **Normalization** - stretch intensities to 0-255 so all images use the same range.
2. **Median filter** - reduce speckle noise while keeping edges.
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - improve local contrast
   in small tiles; the clip limit stops noise from being over-amplified.

## Simple quality measures
| Measure | What it means |
|---|---|
| Contrast (std) | Spread of intensities. Higher = more contrast |
| Entropy | Amount of information in the histogram |
| Edge strength | Mean Sobel gradient. Higher = clearer borders |

## How to run
```bash
pip install -r requirements.txt
# put MRI slices in data/  (see data/README.md)
python preprocess.py --input data --output results
```
Optional settings: `--clip 2.0 --tile 8 --median 3`

## Output
- `results/enhanced/` - enhanced images
- `results/figures/` - original vs denoised vs enhanced, with histograms
- `results/metrics.csv` - measures before and after

## Limitations
- This is a basic project for learning. It is not a clinical tool.
- Higher contrast does not always mean better diagnostic quality. CLAHE can also make noise
  more visible, so the clip limit must be chosen carefully.
- The measures above are simple and do not replace a radiologist's review.
- Tested on a small number of 2D slices only.

## Next steps
- Compare other methods (Gaussian / non-local means denoising, gamma correction).
- Add the measures PSNR/SSIM using a reference image.
- Try 3D volumes (NIfTI) instead of 2D slices.

## Disclaimer
For educational purposes only. Do not use for medical decisions.
## Results
Example results on slices from the Brain Tumor MRI Dataset.

![Axial slice](images/Te-no_2_comparison.png)
![Sagittal slice](images/Te-gl_13_comparison.png)

On the example slices, the enhanced images show clearer borders between brain
structures, and the histograms are spread over a wider intensity range.
Measured values for all images are in `results/metrics.csv`.

Observations:
- CLAHE also slightly increases background noise in some images.
- In sagittal slices, bone and facial structures become very bright.
- The method improves general visibility but does not highlight tumors
  or replace a radiologist's review.

## Data
Images are from the "Brain Tumor MRI Dataset" by Masoud Nickparvar (Kaggle),
licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
The dataset itself is not included in this repository.