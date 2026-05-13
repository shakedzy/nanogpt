import { useEffect, useState } from "react";
import AttentionHeatmap from "./AttentionHeatmap";
import { AttentionResponse, AblateResponse, postAblate } from "../api";

type Props = { resp: AttentionResponse | null };

type Hover = { layer: number; head: number; q: number; k: number; value: number };

const escape = (s: string) => s.replace(/\s/g, "·");
const keyFor = (l: number, h: number) => `${l}-${h}`;

export default function AttentionView({ resp }: Props) {
  const [selected, setSelected] = useState<{ layer: number; head: number } | null>(null);
  const [normalize, setNormalize] = useState(false);
  const [hover, setHover] = useState<Hover | null>(null);

  // Ablation state — set of "L-H" head keys to zero out
  const [ablations, setAblations] = useState<Set<string>>(new Set());
  const [ablateResp, setAblateResp] = useState<AblateResponse | null>(null);
  const [ablateErr, setAblateErr] = useState<string | null>(null);
  const [ablateLoading, setAblateLoading] = useState(false);

  // Reset ablations when prompt changes (so stale results don't show)
  useEffect(() => {
    setAblations(new Set());
    setAblateResp(null);
    setAblateErr(null);
  }, [resp?.prompt]);

  // Fetch /ablate whenever ablation set changes (and we have a prompt)
  useEffect(() => {
    if (!resp || ablations.size === 0) {
      setAblateResp(null);
      return;
    }
    const heads: [number, number][] = [...ablations].map((k) => {
      const [l, h] = k.split("-").map(Number);
      return [l, h] as [number, number];
    });
    let cancelled = false;
    setAblateLoading(true);
    setAblateErr(null);
    postAblate(resp.prompt, heads)
      .then((r) => {
        if (!cancelled) setAblateResp(r);
      })
      .catch((e) => {
        if (!cancelled) setAblateErr(String(e));
      })
      .finally(() => {
        if (!cancelled) setAblateLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ablations, resp]);

  if (!resp) {
    return <div className="text-sm text-slate-500">Run a prompt to see per-head attention.</div>;
  }

  const toggleAblate = (l: number, h: number) => {
    setAblations((prev) => {
      const next = new Set(prev);
      const k = keyFor(l, h);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  };

  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between text-xs text-slate-500">
        <div>
          seq_len={resp.seq_len} · {resp.num_layers} layers × {resp.num_heads} heads.
          Hover a cell for its weight; click a heatmap to enlarge; click <em>ablate</em> to
          zero a head's output and see the prediction delta.
        </div>
        <label className="flex cursor-pointer items-center gap-1.5 text-slate-600">
          <input
            type="checkbox"
            className="h-3.5 w-3.5 accent-indigo-600"
            checked={normalize}
            onChange={(e) => setNormalize(e.target.checked)}
          />
          per-row normalize
        </label>
      </div>

      <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs">
        {hover ? (
          <span>
            <span className="text-slate-500">L{hover.layer}·H{hover.head}</span>{" "}
            <span className="text-slate-800">
              q={hover.q} ({escape(resp.token_strings[hover.q]!)}){" → "}
              k={hover.k} ({escape(resp.token_strings[hover.k]!)})
            </span>{" "}
            <span className="font-semibold text-indigo-700">
              p={hover.value.toFixed(4)}
            </span>
          </span>
        ) : (
          <span className="text-slate-400">
            {normalize
              ? "Color = value ÷ row max (stretches each row to its own max; numbers unchanged). Hover a cell to see its probability."
              : "Color = raw probability. Row 0 always saturated; lower rows fade as 1/(q+1). Hover a cell to see its probability."}
          </span>
        )}
      </div>

      <div
        className="grid gap-3"
        style={{ gridTemplateColumns: `repeat(${resp.num_heads}, minmax(0, 1fr))` }}
      >
        {resp.attention.map((layer, li) =>
          layer.map((mat, hi) => {
            const ablated = ablations.has(keyFor(li, hi));
            return (
              <div key={`${li}-${hi}`} className="flex flex-col gap-1">
                <div className={ablated ? "opacity-40 grayscale" : ""}>
                  <AttentionHeatmap
                    matrix={mat}
                    tokenStrings={resp.token_strings}
                    size={140}
                    title={`L${li} · H${hi}`}
                    normalize={normalize}
                    onClick={() => setSelected({ layer: li, head: hi })}
                    onHover={(cell) =>
                      setHover(cell ? { layer: li, head: hi, ...cell } : null)
                    }
                  />
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleAblate(li, hi);
                  }}
                  className={
                    "self-center rounded px-2 py-0.5 text-[10px] font-mono " +
                    (ablated
                      ? "bg-red-100 text-red-700 hover:bg-red-200"
                      : "text-slate-400 hover:bg-slate-100 hover:text-slate-700")
                  }
                >
                  {ablated ? "● ablated" : "○ ablate"}
                </button>
              </div>
            );
          }),
        )}
      </div>

      {ablations.size > 0 && (
        <AblationPanel
          ablations={ablations}
          resp={ablateResp}
          error={ablateErr}
          loading={ablateLoading}
          onReset={() => setAblations(new Set())}
        />
      )}

      {selected && (
        <div
          className="fixed inset-0 z-10 flex items-center justify-center bg-black/40 p-6"
          onClick={() => setSelected(null)}
        >
          <div
            className="rounded-lg bg-white p-5 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-800">
                Layer {selected.layer} · Head {selected.head}
              </div>
              <button
                className="rounded px-2 py-1 text-xs text-slate-500 hover:bg-slate-100"
                onClick={() => setSelected(null)}
              >
                close
              </button>
            </div>
            <AttentionHeatmap
              matrix={resp.attention[selected.layer]![selected.head]!}
              tokenStrings={resp.token_strings}
              size={Math.max(320, Math.min(560, resp.seq_len * 36))}
              showLabels
              normalize={normalize}
              onHover={(cell) =>
                setHover(
                  cell
                    ? { layer: selected.layer, head: selected.head, ...cell }
                    : null,
                )
              }
            />
          </div>
        </div>
      )}
    </div>
  );
}

function AblationPanel({
  ablations,
  resp,
  error,
  loading,
  onReset,
}: {
  ablations: Set<string>;
  resp: AblateResponse | null;
  error: string | null;
  loading: boolean;
  onReset: () => void;
}) {
  const headList = [...ablations]
    .map((k) => {
      const [l, h] = k.split("-").map(Number);
      return `L${l}H${h}`;
    })
    .sort();

  return (
    <section className="space-y-2 rounded border border-red-200 bg-red-50/40 p-3">
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <div className="font-mono text-slate-700">
          ablated heads: <span className="font-semibold">{headList.join(", ")}</span>
        </div>
        <button
          onClick={onReset}
          className="rounded px-2 py-0.5 text-xs text-slate-500 hover:bg-slate-200 hover:text-slate-800"
        >
          reset
        </button>
      </div>

      {error && <pre className="rounded bg-red-100 p-2 text-xs text-red-700">{error}</pre>}
      {loading && !resp && <div className="text-xs text-slate-500">Computing ablation...</div>}

      {resp && (
        <div className="grid grid-cols-2 gap-3 text-xs">
          <TopKColumn title="baseline" rows={resp.baseline_top_k} />
          <TopKColumn
            title="ablated"
            rows={resp.ablated_top_k}
            highlightFn={(row, i) => {
              const baseRow = resp.baseline_top_k[i];
              return baseRow?.token_id === row.token_id ? "" : "text-red-700";
            }}
          />
        </div>
      )}

      {resp && (
        <div className="text-[11px] text-slate-500">
          prompt: <span className="font-mono">{JSON.stringify(resp.prompt)}</span> — predictions
          are for the <em>next</em> token after the last input token.
        </div>
      )}
    </section>
  );
}

function TopKColumn({
  title,
  rows,
  highlightFn,
}: {
  title: string;
  rows: { token_id: number; token_string: string; prob: number }[];
  highlightFn?: (row: { token_id: number; token_string: string; prob: number }, i: number) => string;
}) {
  return (
    <div className="rounded border border-slate-200 bg-white p-2">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-slate-500">
        {title}
      </div>
      <ol className="space-y-0.5 font-mono">
        {rows.map((row, i) => (
          <li key={i} className={"flex justify-between " + (highlightFn?.(row, i) ?? "")}>
            <span>
              <span className="text-slate-400">{i + 1}.</span>{" "}
              {JSON.stringify(row.token_string)}
            </span>
            <span className="font-semibold text-indigo-700">{(row.prob * 100).toFixed(2)}%</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
