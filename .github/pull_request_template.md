## What and why

<!-- One paragraph: the problem and the chosen approach. Link related issues. -->

## How it was verified

- [ ] `make review` (build + cargo test + pytest + node check + selftest)
- [ ] `make lint` (fmt-check + clippy -D warnings + ruff)
- [ ] `make e2e` (CLI + HTTP against a real daemon)
- [ ] `make e2e-ai` if the neural/RAG layers changed

## Measurements (if performance/memory related)

<!-- Paste `mext bench` table or dashboard numbers before/after. -->

## Screenshots (UI changes)

## Notes for reviewers

<!-- Risky paths, follow-ups, deliberate trade-offs. -->
