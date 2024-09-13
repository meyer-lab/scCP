.PHONY: clean test pyright

flist = $(wildcard sccp/figures/figure*.py)
allOutput = $(patsubst sccp/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

allThomson: $(filter output/figureThomson%, $(allOutput))

allLupus: $(filter output/figureLupus%, $(allOutput))

output/figure%.svg: sccp/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

# output/figureLupus%.svg: sccp/figures/figureLupus%.py
# 	@ mkdir -p ./output
# 	poetry run fbuild Lupus$*

# output/figureCITEseq%.svg: sccp/figures/figureCITEseq%.py
# 	@ mkdir -p ./output
# 	poetry run fbuild CITEseq$*

# output/figureThomson%.svg: sccp/figures/figureThomson%.py 
# 	@ mkdir -p ./output
# 	poetry run fbuild Thomson$*

test: .venv
	rye run pytest -s -v -x

.venv:
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=sccp --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright sccp

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache
