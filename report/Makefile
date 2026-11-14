MAIN = main
OUTDIR = build

.PHONY: all clean distclean

all:
	latexmk -pdf -bibtex -synctex=1 -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(MAIN).tex

clean:
	latexmk -c -outdir=$(OUTDIR) $(MAIN).tex

distclean:
	rm -rf $(OUTDIR)
