# XeniumAlign

Toolkit for registering Xenium and post-Xenium images (mIF, H&E) and evaluating alignment/segmentation quality against ground truth.

## Features

- Rigid (sITK) and affine registration between Xenium images and post-Xenium mIF / H&E images
- Import of cell/nucleus segmentations from QuPath (GeoJSON `detection` objects)
- Alignment quality metrics: Intersection over Union (IoU), IoGT, distance between nuclei centers
- Per-cell spatial visualization of alignment quality

## Project structure

```
├── analysis/           # Quarto tutorials (.qmd)
│   ├── 01_ToyDataset.qmd
│   └── 02_WholeSlide.qmd
├── docs/                # rendered Quarto documentation site
└── xenium_align/        # Python package (xa)
```

## Getting started

```bash
pip install -e .
```

Then follow the tutorials:

- **Registration on toy dataset** — `analysis/01_ToyDataset.qmd`
- **Registration on whole slide** — `analysis/02_WholeSlide.qmd`

## Documentation

Full docs and tutorials: build/preview the Quarto site with `quarto preview` from the project root.

## License

TODO
