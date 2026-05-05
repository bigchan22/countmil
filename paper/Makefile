LATEX=pdflatex
BIBTEX=bibtex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib sections/*.tex
	$(LATEX) main
	$(BIBTEX) main || true
	$(LATEX) main
	$(LATEX) main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc main.pdf
