ROOT_FILE		= main
LATEX_BUILD_TOOL= xelatex 
BIB_TOOL		= biber
TEX_ARGS 		= -synctex=1 -shell-escape -interaction=nonstopmode

run:
	$(LATEX_BUILD_TOOL) $(TEX_ARGS) $(ROOT_FILE).tex 
	$(BIB_TOOL)	$(ROOT_FILE)
	$(LATEX_BUILD_TOOL) $(TEX_ARGS) $(ROOT_FILE).tex 
	$(LATEX_BUILD_TOOL) $(TEX_ARGS) $(ROOT_FILE).tex 
	
clean:
	rm -f *.aux  *.bbl *.blg *.bcf *.idx *.ind *.lof *.lot *.out *.toc *.acn *.acr *.alg *.glg *.glo *.gls *.ist *.fls *.log *.run.xml *.fdb_latexmk *.synctex.gz *.xdv