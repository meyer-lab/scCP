SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard sccp/figures/figure*.py)
allOutput = $(patsubst sccp/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

allThomson: $(filter output/figureThomson%, $(allOutput))

allLupus: $(filter output/figureLupus%, $(allOutput))

output/figure%.svg: sccp/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

output/figureLupus%.svg: sccp/figures/figureLupus%.py
	@ mkdir -p ./output
	poetry run fbuild Lupus$*

output/figureCITEseq%.svg: sccp/figures/figureCITEseq%.py
	@ mkdir -p ./output
	poetry run fbuild CITEseq$*

output/figureThomson%.svg: sccp/figures/figureThomson%.py 
	@ mkdir -p ./output
	poetry run fbuild Thomson$*

test:
	poetry run pytest -s -x -v

coverage.xml:
	poetry run pytest --cov=sccp --cov-report=xml

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports --check-untyped-defs sccp
