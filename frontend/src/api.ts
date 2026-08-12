import type { ModelInfo, Prediction } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export async function getModelInfo(): Promise<ModelInfo> {
  return parseResponse<ModelInfo>(await fetch(`${API_BASE}/api/v1/model`));
}

export async function predictFrame(input: {
  rgb: File;
  mask: File;
  sequenceId: string;
  reset: boolean;
  threshold?: number;
}): Promise<Prediction> {
  const body = new FormData();
  body.append("rgb", input.rgb);
  body.append("semantic_mask", input.mask);
  body.append("sequence_id", input.sequenceId);
  body.append("reset_sequence", String(input.reset));
  if (input.threshold !== undefined) body.append("abstention_threshold", String(input.threshold));
  return parseResponse<Prediction>(
    await fetch(`${API_BASE}/api/v1/predict/frame`, { method: "POST", body }),
  );
}

export async function resetSequence(sequenceId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/sequences/${encodeURIComponent(sequenceId)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error("Could not reset sequence state");
}
