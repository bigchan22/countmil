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

`main.tex` follows the NeurIPS 2026 submission shell and loads the official
`neurips_2026.sty` automatically when it is present. The current repo does not
include that style file, so local builds use a simple `article` fallback only for
drafting. For submission, place the unmodified official `neurips_2026.sty` and
the official `checklist.tex` in `paper/`; do not modify the style file.

Current priorities:

- Fill final result tables from aggregated CSVs.
- Keep claim strength aligned with `experiment_report_2026-05-06.md`.
- Add full seed-level appendices once provisional baselines finish.
- Verify the CIFAR-10 pretrained protocol details before making exact protocol-match claims.
