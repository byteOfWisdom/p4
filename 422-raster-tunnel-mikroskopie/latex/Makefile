all:
	pdflatex main.tex -interaction=nonstopmode > /dev/null
	biber main > /dev/null
	pdflatex main.tex -interaction=nonstopmode | tail

clean:
	rm *.log
	rm *.aux
	rm *.out
	
