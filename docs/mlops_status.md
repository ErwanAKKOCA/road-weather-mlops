# DevOps / MLOps status — 2026-08-12 update

## Completed in this update

- removed the unused parallel FastAPI router and duplicate in-memory metrics store;
- centralized the default abstention threshold at 0.90 in backend, frontend, Compose, and Azure runtime;
- repaired pytest against the real `/api/v1/*` API contract using the deterministic mock engine;
- retained Prometheus and added abstention, predicted-class, confidence, entropy, active-sequence, and model-init metrics;
- connected structured inference logging without persisting uploaded images;
- added a checksum manifest for the Mask2Former checkpoint and runtime verification;
- added a CLI release validator for both weather and segmentation artifacts;
- strengthened CI with release validation and a lightweight Docker smoke test;
- gated automatic Azure deployment on successful CI on `main`;
- documented application/model rollback using immutable Git-SHA image tags;
- switched frontend Docker builds from `npm install` to `npm ci`.

## Remaining optional / environment-dependent work

- publish the Mask2Former locked release to object storage if the production Git repository does not carry the checkpoint;
- configure Azure startup/readiness/liveness probes and production alerts in the Azure environment;
- optionally add dependency/image security scanning;
- optionally add frontend unit tests;
- select final screenshots and monitoring evidence for the thesis.
