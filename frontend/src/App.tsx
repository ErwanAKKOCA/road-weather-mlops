import { FormEvent, useEffect, useMemo, useState } from "react";
import { getModelInfo, predictFrame, resetSequence } from "./api";
import type { ModelInfo, Prediction, Probability } from "./types";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function ProbabilityBars({ rows }: { rows: Probability[] }) {
  return (
    <div className="probabilities">
      {rows.map((row) => (
        <div className="probability" key={row.label}>
          <div>
            <span>{row.label}</span>
            <strong>{percent(row.value)}</strong>
          </div>

          <div className="bar">
            <span style={{ width: percent(row.value) }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [rgb, setRgb] = useState<File | null>(null);
  const [mask, setMask] = useState<File | null>(null);
  const [sequenceId, setSequenceId] = useState("demo-route-01");
  const [threshold, setThreshold] = useState(0.90);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [firstFrame, setFirstFrame] = useState(true);

  useEffect(() => {
    getModelInfo()
      .then(setModel)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const rgbPreview = useMemo(
    () => (rgb ? URL.createObjectURL(rgb) : ""),
    [rgb]
  );

  const maskPreview = useMemo(
    () => (mask ? URL.createObjectURL(mask) : ""),
    [mask]
  );

  async function submit(event: FormEvent) {
    event.preventDefault();

    if (!rgb || !mask) {
      setError("Select both an RGB frame and its semantic mask.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const next = await predictFrame({
        rgb,
        mask,
        sequenceId,
        reset: firstFrame,
        threshold,
      });

      setPrediction(next);
      setFirstFrame(false);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Prediction failed"
      );
    } finally {
      setLoading(false);
    }
  }

  async function newSequence() {
    await resetSequence(sequenceId).catch(() => undefined);
    setPrediction(null);
    setFirstFrame(true);
  }

  return (
    <main>
      <header>
        <div>
          <span>MASTER THESIS · PHASE VII</span>
          <h1>Road Weather Lab</h1>
          <p>
            A deployable research interface for temporally stable weather
            recognition in driving scenes.
          </p>
        </div>

        <div className="status">
          {model ? "Phase VI model ready" : "Connecting to model"}
        </div>
      </header>

      <section className="notice">
        <strong>Research boundary</strong>
        <span>
          {model?.limitation ??
            "This demonstrator requires the matching Virtual KITTI 2 semantic mask."}
        </span>
      </section>

      <div className="workspace">
        <form className="panel input-panel" onSubmit={submit}>
          <div className="panel-heading">
            <span>01</span>

            <div>
              <h2>Sequence input</h2>
              <p>Send frames in temporal order.</p>
            </div>
          </div>

          <label>
            Sequence identifier
            <input
              value={sequenceId}
              maxLength={128}
              onChange={(event) => {
                setSequenceId(event.target.value);
                setFirstFrame(true);
              }}
            />
          </label>

          <div className="upload-grid">
            <label className="dropzone">
              {rgbPreview ? (
                <img src={rgbPreview} alt="RGB preview" />
              ) : (
                <b>
                  RGB frame
                  <small>PNG or JPEG</small>
                </b>
              )}

              <input
                type="file"
                accept="image/png,image/jpeg"
                onChange={(event) =>
                  setRgb(event.target.files?.[0] ?? null)
                }
              />
            </label>

            <label className="dropzone mask">
              {maskPreview ? (
                <img src={maskPreview} alt="Mask preview" />
              ) : (
                <b>
                  Semantic mask
                  <small>Matching colour GT</small>
                </b>
              )}

              <input
                type="file"
                accept="image/png"
                onChange={(event) =>
                  setMask(event.target.files?.[0] ?? null)
                }
              />
            </label>
          </div>

          <label className="threshold">
            <span>
              Abstention threshold <strong>{percent(threshold)}</strong>
            </span>

            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(event) =>
                setThreshold(Number(event.target.value))
              }
            />
          </label>

          {error && <p className="error">{error}</p>}

          <div className="actions">
            <button type="submit" disabled={loading}>
              {loading ? "Analysing…" : "Analyse frame"}
            </button>

            <button
              className="secondary"
              type="button"
              onClick={newSequence}
            >
              New sequence
            </button>
          </div>
        </form>

        <section className="panel result-panel">
          <div className="panel-heading">
            <span>02</span>

            <div>
              <h2>Temporal output</h2>
              <p>Comparison of raw and EMA-smoothed probabilities.</p>
            </div>
          </div>

          {prediction ? (
            <>
              <div
                className={`verdict ${
                  prediction.abstained ? "abstained" : ""
                }`}
              >
                <small>
                  {prediction.abstained
                    ? "ABSTAIN · LOW CONFIDENCE"
                    : "PREDICTED CONDITION"}
                </small>

                <strong>{prediction.predicted_class}</strong>

                <span>
                  {percent(prediction.confidence)} confidence after EMA
                </span>
              </div>

              <div className="probability-section">
                <h3>Raw probabilities — without EMA</h3>
                <ProbabilityBars rows={prediction.raw_probabilities} />
              </div>

              <div className="probability-section">
                <h3>Temporal probabilities — after EMA</h3>
                <ProbabilityBars
                  rows={prediction.temporal_probabilities}
                />
              </div>

              <div className="metrics">
                <div>
                  <span>Frame</span>
                  <b>{prediction.frame_index}</b>
                </div>

                <div>
                  <span>Entropy</span>
                  <b>{prediction.entropy.toFixed(3)}</b>
                </div>

                <div>
                  <span>Latency</span>
                  <b>{prediction.inference_ms.toFixed(0)} ms</b>
                </div>
              </div>
            </>
          ) : (
            <div className="empty">
              <span>◫</span>
              <p>
                Your prediction will appear here after the first frame.
              </p>
            </div>
          )}
        </section>
      </div>

      <section className="method-strip">
        <div>
          <span>Representation</span>
          <strong>DINOv2 ViT-S/14</strong>
        </div>

        <div>
          <span>Region</span>
          <strong>Semantic sky</strong>
        </div>

        <div>
          <span>Temporal method</span>
          <strong>EMA α = 0.6</strong>
        </div>

        <div>
          <span>Classes</span>
          <strong>Clone · Fog · Rain</strong>
        </div>
      </section>

      <footer>
        Erwan AKKOCA · Vision-based weather classification · Phase VII
        research demonstrator
      </footer>
    </main>
  );
}

export default App;