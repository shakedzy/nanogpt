import { useEffect, useState } from "react";
import {
  getLayerSummary,
  getNeuron,
  LayerSummaryResponse,
  NeuronResponse,
} from "../api";

const escape = (s: string) => s.replace(/\n/g, "↵").replace(/\t/g, "→");

function cellColor(v: number, maxOverall: number): string {
  if (maxOverall <= 0) return "white";
  const a = Math.pow(Math.max(0, Math.min(1, v / maxOverall)), 0.6);
  return `rgba(79, 70, 229, ${a})`;
}

export default function NeuronBrowser() {
  const [layer, setLayer] = useState(0);
  const [summary, setSummary] = useState<LayerSummaryResponse | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [detail, setDetail] = useState<NeuronResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingLayer, setLoadingLayer] = useState(false);
  const [loadingNeuron, setLoadingNeuron] = useState(false);

  // Load layer summary whenever layer changes
  useEffect(() => {
    let cancelled = false;
    setLoadingLayer(true);
    setError(null);
    setSummary(null);
    setSelected(null);
    setDetail(null);
    getLayerSummary(layer)
      .then((r) => {
        if (!cancelled) setSummary(r);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoadingLayer(false);
      });
    return () => {
      cancelled = true;
    };
  }, [layer]);

  // Load neuron detail whenever selection changes
  useEffect(() => {
    if (selected === null) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoadingNeuron(true);
    getNeuron(layer, selected)
      .then((r) => {
        if (!cancelled) setDetail(r);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoadingNeuron(false);
      });
    return () => {
      cancelled = true;
    };
  }, [layer, selected]);

  if (error && !summary) {
    return (
      <pre className="rounded bg-red-50 p-3 text-xs text-red-700">{error}</pre>
    );
  }

  const maxOverall = summary
    ? summary.neurons.reduce((m, n) => (n.max_value > m ? n.max_value : m), 0)
    : 1;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
        <label className="flex items-center gap-2">
          <span>layer</span>
          <select
            value={layer}
            onChange={(e) => setLayer(parseInt(e.target.value, 10))}
            className="rounded border border-slate-300 px-2 py-1 text-xs"
            disabled={loadingLayer}
          >
            {[0, 1, 2, 3].map((l) => (
              <option key={l} value={l}>
                L{l}
              </option>
            ))}
          </select>
        </label>
        {summary && (
          <span>
            {summary.metadata.neurons_per_layer} neurons · scanned{" "}
            {summary.metadata.num_tokens_scanned.toLocaleString()} tokens · top-
            {summary.metadata.top_k} per neuron. Color = relative max activation
            across this layer. Click any cell to see what fires it.
          </span>
        )}
      </div>

      {error && (
        <pre className="rounded bg-red-50 p-3 text-xs text-red-700">{error}</pre>
      )}

      {loadingLayer && (
        <div className="text-xs text-slate-500">Loading layer {layer}...</div>
      )}

      {summary && (
        <div className="grid grid-cols-[repeat(32,minmax(0,1fr))] gap-px rounded border border-slate-200 bg-slate-200 p-px">
          {summary.neurons.map((n) => {
            const isSelected = selected === n.idx;
            return (
              <button
                key={n.idx}
                onClick={() => setSelected(n.idx)}
                className={
                  "aspect-square text-[8px] " +
                  (isSelected
                    ? "ring-2 ring-red-500 ring-inset"
                    : "hover:ring-1 hover:ring-slate-400")
                }
                style={{ backgroundColor: cellColor(n.max_value, maxOverall) }}
                title={`n${n.idx} · max=${n.max_value.toFixed(2)} · top=${JSON.stringify(n.top_token)}`}
              />
            );
          })}
        </div>
      )}

      {selected !== null && (
        <section className="space-y-2 rounded border border-slate-200 p-3">
          <div className="flex items-baseline justify-between text-sm">
            <div className="font-mono">
              <span className="text-slate-500">L{layer} · neuron </span>
              <span className="font-semibold text-slate-900">{selected}</span>
              {detail && (
                <span className="ml-3 text-slate-500">
                  max={detail.max_value.toFixed(3)} · top_token=
                  <span className="font-semibold text-indigo-700">
                    {JSON.stringify(detail.top_token)}
                  </span>
                </span>
              )}
            </div>
            <button
              onClick={() => setSelected(null)}
              className="text-xs text-slate-500 hover:text-slate-800"
            >
              clear
            </button>
          </div>
          {loadingNeuron && (
            <div className="text-xs text-slate-500">Loading neuron...</div>
          )}
          {detail && (
            <ol className="space-y-1 font-mono text-xs">
              {detail.contexts.map((c, i) => (
                <li key={i} className="flex gap-2">
                  <span className="w-10 shrink-0 text-right text-slate-400">
                    {i + 1}.
                  </span>
                  <span className="w-16 shrink-0 text-right text-indigo-700">
                    {c.value.toFixed(3)}
                  </span>
                  <span className="grow whitespace-pre-wrap break-words text-slate-800">
                    <span className="text-slate-500">…{escape(c.before_text)}</span>
                    <span className="rounded bg-yellow-200 px-0.5 font-semibold">
                      {escape(c.token_string)}
                    </span>
                    <span className="text-slate-500">{escape(c.after_text)}…</span>
                  </span>
                </li>
              ))}
            </ol>
          )}
        </section>
      )}
    </div>
  );
}
