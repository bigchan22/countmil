# NeurIPS Paper Draft

Main file:

```bash
paper/main.tex
```

Compile from repo root:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The draft currently compiles as a standard `article` unless `neurips_2026.sty` is present. For NeurIPS 2026 submission, place the official `neurips_2026.sty` in this directory; `main.tex` will load it automatically with the default main-track anonymous submission option.

Current priorities:

- Fill final result tables from aggregated CSVs.
- Replace TODO citations, especially Shukla/count loss and LLP-PVC.
- Add method derivations for posterior marginals and FFT-tree backend.
- Decide whether CIFAR-10 ResNet-18 results are strong enough for the main paper.
