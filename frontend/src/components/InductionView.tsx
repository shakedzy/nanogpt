import { useEffect, useState } from "react";
import { postInductionScan, InductionResponse } from "../api";
import AttentionHeatmap from "./AttentionHeatmap";

export default function InductionView() {
  const [resp, setResp] = useState<InductionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<{ layer: number; head: number } | null>(null);

  const run = () => {
    setLoading(true);
    setError(null);
    setSelected(null);
    postInductionScan()
      .then(setResp)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  // Auto-run on first mount
  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return <pre className="rounded bg-red-50 p-3 text-xs text-red-700">{error}</pre>;
  }
  if (loading || !resp) {
    return <div className="text-sm text-slate-500">Running induction scan...</div>;
  }

  const maxScore = Math.max(...resp.scores.map((s) => s.score), resp.candidate_threshold);

  return (
    <div className="space-y-4">
      <div className="space-y-2 text-xs text-slate-600">
        <div>
          <span className="font-semibold text-slate-800">What this test does.</span>{" "}
          We build {resp.num_seqs} sequences that contain {resp.seq_len} random tokens, then
          repeat the same {resp.seq_len} tokens — so positions {resp.seq_len}…
          {resp.total_len - 1} are an exact copy of positions 0…{resp.seq_len - 1}. An{" "}
          <em>induction head</em> spots the repetition: at a second-half token, it looks
          back, finds where the token appeared before, and attends to the token that came
          right after. Below: how strongly each head does this, averaged across the{" "}
          {resp.num_seqs} sequences.
        </div>
        <div>
          <span className="font-semibold text-slate-800">How to read the score.</span>{" "}
          Random-attention baseline ≈{" "}
          <span className="font-mono">
            {(1 / (resp.seq_len + resp.seq_len / 2)).toFixed(3)}
          </span>{" "}
          (a head that pays equal attention to every legal position would land here).{" "}
          <span className="font-mono">≥ {resp.candidate_threshold}</span> means the head
          puts most of its attention mass exactly on the right spot — a real induction
          head (per Olsson et al., 2022). Between those is "trying but hasn't committed."
        </div>
        <div>
          <span className="font-semibold text-slate-800">Click a row</span> to see that
          head's attention pattern on a sample sequence below the table.
        </div>
        <div className="pt-1">
          <button
            onClick={run}
            className="rounded bg-indigo-600 px-2 py-0.5 text-xs text-white hover:bg-indigo-700"
          >
            re-run with new random sequences
          </button>
        </div>
      </div>

      <table className="w-full max-w-2xl border-collapse text-xs">
        <thead>
          <tr className="border-b border-slate-200">
            <th className="py-1 pr-2 text-left font-medium text-slate-500">rank</th>
            <th className="py-1 pr-2 text-left font-medium text-slate-500">head</th>
            <th className="py-1 pr-2 text-left font-medium text-slate-500">score</th>
            <th className="py-1 text-left font-medium text-slate-500"></th>
          </tr>
        </thead>
        <tbody>
          {resp.scores.map((s, i) => {
            const isCandidate = s.score >= resp.candidate_threshold;
            const isSelected =
              selected?.layer === s.layer && selected?.head === s.head;
            return (
              <tr
                key={`${s.layer}-${s.head}`}
                onClick={() => setSelected({ layer: s.layer, head: s.head })}
                className={
                  "cursor-pointer border-b border-slate-100 hover:bg-slate-50 " +
                  (isSelected ? "bg-indigo-50" : "")
                }
              >
                <td className="py-1 pr-2 font-mono text-slate-400">{i + 1}.</td>
                <td className="py-1 pr-2 font-mono">
                  L{s.layer}H{s.head}
                </td>
                <td className="py-1 pr-2 font-mono">
                  <span className={isCandidate ? "font-semibold text-emerald-700" : ""}>
                    {s.score.toFixed(4)}
                  </span>
                  {isCandidate && (
                    <span className="ml-1 text-emerald-700">★ candidate</span>
                  )}
                </td>
                <td className="py-1 pl-2">
                  <div
                    className="h-2 rounded"
                    style={{
                      width: `${(s.score / maxScore) * 240}px`,
                      minWidth: "1px",
                      backgroundColor: isCandidate
                        ? "rgb(16, 185, 129)"
                        : "rgb(79, 70, 229)",
                      opacity: isCandidate ? 1 : 0.6,
                    }}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {selected && (
        <section className="space-y-3 rounded border border-slate-200 p-3">
          <div className="text-sm font-mono text-slate-700">
            L{selected.layer}·H{selected.head} attention on a sample {resp.total_len}-token
            repeated sequence.
          </div>
          <div className="text-xs text-slate-600">
            <div className="mb-1">
              <span className="font-semibold text-slate-800">How to read it:</span>{" "}
              x-axis = key position (0 = start of sequence), y-axis = query position.
              Each cell shows how much attention the head at row <em>i</em> sends to
              column <em>k</em>. Brighter = more attention. The lower-left triangle is
              all that matters (the upper-right is masked out by the causal mask).
            </div>
            <div className="mb-1 font-semibold text-slate-800">
              Common patterns to look for:
            </div>
            <ul className="ml-4 list-disc space-y-0.5">
              <li>
                <span className="font-semibold text-emerald-700">Induction head</span> —
                a bright stripe in the lower-right quadrant, running parallel to the
                main diagonal but offset by{" "}
                <span className="font-mono">−{resp.seq_len - 1}</span> (i.e., row{" "}
                <em>i</em> ≥ {resp.seq_len} attends to column <em>i</em>−
                {resp.seq_len - 1}). What we're hunting for.
              </li>
              <li>
                <span className="font-semibold text-slate-800">BOS / attention sink</span>{" "}
                — a bright <em>vertical</em> column at the left edge (every row attends
                to position 0 or the first few positions). The head is ignoring content
                and parking its attention at the start of the sequence — softmax has to
                put probability mass <em>somewhere</em>, so the model uses BOS as a safe
                default. Common in many heads, especially after training.
              </li>
              <li>
                <span className="font-semibold text-slate-800">Previous-token head</span>{" "}
                — a thin diagonal stripe one cell below the main diagonal (row{" "}
                <em>i</em> attends to column <em>i</em>−1). The companion to induction
                heads: layer-0 previous-token info is what layer-N+ induction heads read
                from.
              </li>
              <li>
                <span className="font-semibold text-slate-800">Diffuse / noise</span> —
                scattered bright cells with no clear shape. The head isn't doing anything
                interpretable on this synthetic test (it may still be useful on real
                text).
              </li>
            </ul>
          </div>
          <AttentionHeatmap
            matrix={resp.sample.attention[selected.layer]![selected.head]!}
            tokenStrings={resp.sample.tokens.map((t, i) => `${i}:${t}`)}
            size={420}
          />
        </section>
      )}
    </div>
  );
}
