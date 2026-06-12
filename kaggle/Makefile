# ASRA-Security Kaggle submission helpers

.PHONY: validate test push

validate:
	../.venv/bin/aicomp validate attack.py

test:
	../.venv/bin/aicomp test attack.py --track redteam --budget-s 60 --agent deterministic --quick

push:
	kaggle kernels push -p .
