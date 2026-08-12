export type Probability = { label: string; value: number };

export type Prediction = {
  sequence_id: string;
  frame_index: number;
  predicted_class: string;
  confidence: number;
  entropy: number;
  abstained: boolean;
  raw_probabilities: Probability[];
  temporal_probabilities: Probability[];
  inference_ms: number;
  model: string;
  region: string;
  temporal_method: string;
  oracle_mask_required: boolean;
};

export type ModelInfo = {
  status: string;
  model_mode: string;
  model: string;
  backbone: string;
  region: string;
  classes: string[];
  temporal_method: string;
  ema_alpha: number;
  oracle_mask_required: boolean;
  limitation: string;
};
