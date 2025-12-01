all:
	pdflatex main.tex -interaction=nonstopmode
	biber main
	pdflatex main.tex -interaction=nonstopmode

clean:
	rm *.log
	rm *.aux
	rm *.out
	
