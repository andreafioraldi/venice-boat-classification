all:
	pdflatex report.tex
	pdflatex report.tex

clear:
	rm *.aux *.log *.out *.swp
