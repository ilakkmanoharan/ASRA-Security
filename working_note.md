# ASRA-Security: Adaptive Search for Multi-Step Tool-Using Agent Failures

## 1. Motivation

Tool-using agents can fail across multiple steps when untrusted content influences later privileged actions. This submission treats agent security as sequential decision-making and searches for replayable multi-step attack paths in the competition sandbox.

## 2. Threat model within the benchmark

We target four sandbox predicates only:

- EXFILTRATION via `http.post`
- UNTRUSTED_TO_ACTION from `web.search` / `email.read`
- DESTRUCTIVE_WRITE on `secret.txt`
- CONFUSED_DEPUTY via `email.send`

## 3. Search algorithm

Go-Explore-style search with archive-backed state restoration, weighted cell selection, and prompt mutation over curated benchmark seeds.

## 4. Novelty archive

Each novel tool-call cell signature is archived with snapshot, score hint, depth, and detected real-attack annotations to guide branching.

## 5. Trace scoring

Heuristic scoring combines predicate severity, secret reads, egress, untrusted-to-risky transitions, and new real-attack detections.

## 6. Replayability

Findings are returned as `AttackCandidate(user_messages=...)` chains for evaluator replay.

## 7. Safety boundaries

All payloads and prompts are fictional benchmark fixtures. No real-world exploit instructions or external services.

## 8. Expected limitations

Performance depends on agent model and guardrail strictness. Search budget may not cover all predicate families equally on both target models.

## 9. Future work

Action semantics learning, security world models, and LLM-assisted trace mutation for broader unique-cell coverage.
