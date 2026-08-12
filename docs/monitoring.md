# Monitoring and operational evidence

The API exposes Prometheus-format counters and latency histograms at `/metrics` and logs
startup/prediction failures to standard output, which Azure Container Apps can collect in
Log Analytics.

Recommended dashboard signals:

- request count split by `ok`, `invalid`, and `error`;
- p50, p95, and p99 inference latency;
- readiness failures and container restarts;
- replica CPU and memory;
- abstention count/rate (`weather_api_abstentions_total`);
- class-frequency (`weather_api_predicted_classes_total`);
- confidence and entropy distributions (`weather_api_confidence`, `weather_api_entropy`);
- active in-memory sequence states (`weather_api_active_sequences`);
- class-frequency and confidence drift, without storing uploaded images by default.

Recommended alerts:

- readiness unavailable for five minutes;
- error ratio above 5% over ten minutes;
- p95 latency above the agreed demo threshold;
- unexpected restart loop;
- checksum or model-initialization failure.

Privacy: the current implementation does not persist uploaded RGB frames or masks. Keep
that property unless a documented retention purpose, access policy, and deletion process
are introduced.


Structured inference events are emitted as JSON to standard output and optionally mirrored to `LOG_PATH`. Uploaded RGB frames and masks are not written to the log.
