export type ForwardResponse = {
  prompt: string;
  tokens: number[];
  token_strings: string[];
  vocab_size: number;
  context_length: number;
  logits_shape: number[];
  logits: number[][];
};

const BASE = "/api";

export async function getHealth(): Promise<unknown> {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`health failed: ${r.status}`);
  return r.json();
}

export async function postForward(prompt: string): Promise<ForwardResponse> {
  const r = await fetch(`${BASE}/forward`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!r.ok) throw new Error(`forward failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export type AttentionResponse = {
  prompt: string;
  tokens: number[];
  token_strings: string[];
  num_layers: number;
  num_heads: number;
  seq_len: number;
  // [layer][head][query_pos][key_pos]
  attention: number[][][][];
};

export async function postAttention(prompt: string): Promise<AttentionResponse> {
  const r = await fetch(`${BASE}/attention`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!r.ok) throw new Error(`attention failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export type LogitLensCell = {
  token_id: number;
  token_string: string;
  prob: number;
};

export type LogitLensResponse = {
  prompt: string;
  tokens: number[];
  token_strings: string[];
  num_layers: number;
  seq_len: number;
  top_k: number;
  // layers[layer][pos] -> list of top-k cells, descending by prob
  layers: LogitLensCell[][][];
};

export async function postLogitLens(prompt: string): Promise<LogitLensResponse> {
  const r = await fetch(`${BASE}/logit_lens`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!r.ok) throw new Error(`logit_lens failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export type NeuronSummary = {
  idx: number;
  max_value: number;
  top_token: string;
};

export type NeuronContext = {
  value: number;
  global_pos: number;
  token_id: number;
  token_string: string;
  before_text: string;
  after_text: string;
};

export type LayerSummaryResponse = {
  layer: number;
  metadata: {
    num_layers: number;
    neurons_per_layer: number;
    top_k: number;
    context_before: number;
    context_after: number;
    num_tokens_scanned: number;
  };
  neurons: NeuronSummary[];
};

export type NeuronResponse = {
  layer: number;
  idx: number;
  metadata: LayerSummaryResponse["metadata"];
  max_value: number;
  top_token: string;
  contexts: NeuronContext[];
};

export async function getLayerSummary(layer: number): Promise<LayerSummaryResponse> {
  const r = await fetch(`${BASE}/neurons/${layer}`);
  if (!r.ok) throw new Error(`neurons/${layer} failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function getNeuron(layer: number, idx: number): Promise<NeuronResponse> {
  const r = await fetch(`${BASE}/neuron/${layer}/${idx}`);
  if (!r.ok) throw new Error(`neuron/${layer}/${idx} failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export type AblationTopK = {
  token_id: number;
  token_string: string;
  prob: number;
};

export type AblateResponse = {
  prompt: string;
  tokens: number[];
  token_strings: string[];
  ablated_heads: number[][];
  ablated_neurons: number[][];
  baseline_top_k: AblationTopK[];
  ablated_top_k: AblationTopK[];
  baseline_top_under_ablation: AblationTopK[];
};

export async function postAblate(
  prompt: string,
  ablateHeads: [number, number][],
  ablateNeurons: [number, number][] = [],
): Promise<AblateResponse> {
  const r = await fetch(`${BASE}/ablate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      ablate_heads: ablateHeads,
      ablate_neurons: ablateNeurons,
    }),
  });
  if (!r.ok) throw new Error(`ablate failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export type InductionScore = { layer: number; head: number; score: number };

export type InductionResponse = {
  num_seqs: number;
  seq_len: number;
  total_len: number;
  num_layers: number;
  num_heads: number;
  candidate_threshold: number;
  scores: InductionScore[];
  sample: {
    tokens: number[];
    // attention[layer][head][q][k]
    attention: number[][][][];
  };
};

export async function postInductionScan(): Promise<InductionResponse> {
  const r = await fetch(`${BASE}/induction_scan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  if (!r.ok) throw new Error(`induction_scan failed: ${r.status} ${await r.text()}`);
  return r.json();
}
