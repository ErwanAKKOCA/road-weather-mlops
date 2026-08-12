# Required Phase VI release files

Copy or mount these files from `phase6_vkitti2_temporal_v4_1_release.tar.gz`:

- `dinov2_logistic_semantic_sky_v3_1.joblib`
- `region_definitions_v3_1.json`
- `release_manifest_v4_1.json` (recommended, for SHA-256 verification)

The canonical archive was produced by Phase VI and uploaded to:
`gs://thesis_weather_storage/phase6/releases/phase6_vkitti2_temporal_v4_1_release.tar.gz`.

Do not commit model artifacts to a public repository. Mount this directory read-only in
Docker Compose or inject it through a secured deployment process.

## Required Phase VII checkpoint files

Create the following directory from the three files exported by `save_pretrained()`:

```text
mask2former_kitti_best/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```
