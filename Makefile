# =====================================================================
#  Entry points.
#
#    make verify       no GPU, no data, no download, ~1 min.  Checks that
#                      the numbers published in paper/generated/ are
#                      exactly what the generator produces, and runs the
#                      statistics unit tests.  Start here.
#    make smoke        exercise the whole training/evaluation pipeline on
#                      synthetic signals (~20 min, CPU only)
#    make tables       regenerate paper/generated/ from code/results/
#                      (use after `make experiments`, or after dropping a
#                      published results/ directory in place)
#    make experiments  the real run: needs PTB-XL and a GPU, ~45-55 GPU-h
#    make clean        remove caches and run outputs
# =====================================================================

PY   ?= python
CODE  = code

.PHONY: all help verify test smoke tables experiments clean

all: verify

help:
	@sed -n '2,15p' Makefile

verify:
	@echo "== 1/3  statistics unit tests ===================================="
	cd $(CODE) && $(PY) tests/test_stats.py
	@echo
	@echo "== 2/3  regenerating derived quantities =========================="
	cd $(CODE) && $(PY) scripts/link_budget.py --paper-dir ../paper
	cd $(CODE) && $(PY) scripts/emit_config_table.py --paper-dir ../paper
	@echo
	@echo "== 3/3  published numbers are the generator's output ============="
	cd $(CODE) && $(PY) tests/test_reproduce_tables.py \
	    --require-published-match
	@echo
	@echo "VERIFY COMPLETE."

test: verify

smoke:
	cd $(CODE) && $(PY) scripts/smoke_test.py --full

tables:
	cd $(CODE) && $(PY) scripts/aggregate_results.py \
	    --results-dir results --paper-dir ../paper
	cd $(CODE) && $(PY) scripts/link_budget.py --paper-dir ../paper
	cd $(CODE) && $(PY) scripts/emit_config_table.py --paper-dir ../paper

experiments:
	cd $(CODE) && ./run_all.sh

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(CODE)/results/runs $(CODE)/results/logs
