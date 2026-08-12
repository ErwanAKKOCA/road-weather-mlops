# Rollback procedure

The deployment is intentionally immutable: API and UI images are tagged with the Git commit SHA.
Model files are protected by release manifests and SHA-256 checksums.

## Application rollback

1. Identify the last known-good Git SHA in GitHub Actions or Azure Container Registry.
2. Re-deploy the API image `phase7-weather-api:<GOOD_SHA>` and UI image `phase7-weather-ui:<GOOD_SHA>`.
3. Keep API `min-replicas=1` and `max-replicas=1` while EMA state is in memory.
4. Verify `/health/live`, `/health/ready`, `/api/v1/model`, and `/metrics`.
5. Run one known smoke-test sequence before declaring recovery complete.

## Model rollback

1. Restore the previous locked model release into the deployment build context.
2. Do not edit weights in place: restore the complete release (model + region/config + manifest).
3. Run `python backend/scripts/validate_release.py --artifacts backend/artifacts`.
4. Build a new immutable application image referencing the restored release.
5. Deploy and verify health, model metadata, confidence/abstention behavior, and a known smoke test.

## Safety rule

Never bypass a checksum mismatch. A checksum failure blocks deployment and should be treated as a corrupted or unintended artifact.
