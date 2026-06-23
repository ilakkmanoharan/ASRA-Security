# ASRA-Security Kaggle submission helpers

.PHONY: validate test push push-test sync-attack

validate:
	../.venv/bin/aicomp validate attack.py

test:
	../.venv/bin/aicomp test attack.py --track redteam --budget-s 60 --agent deterministic --quick

sync-attack:
	cp ../attack.py attack.py

push:
	kaggle kernels push -p .

# Cloud GPT-OSS test kernel (T4 + internet ON). Uses REST API — see push-test-kernel.py
push-test:
	../.venv/bin/python push-test-kernel.py

# Submit kernel (T4, internet off)
push-submit: sync-attack
	../.venv/bin/python sync_submit_notebook.py --mode asra
	../.venv/bin/python push-submit-kernel.py

submit-day1:
	../.venv/bin/python submit_day1.py --all

submit-day2:
	../.venv/bin/python submit_day2.py --all
