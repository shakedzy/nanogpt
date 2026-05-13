import { useState } from "react";
import { LogitLensResponse } from "../api";

type Props = { resp: LogitLensResponse | null };

const escape = (s: string) => s.replace(/\s/g, "·");

// alpha scaled non-linearly so tiny probs are still visible at all
function cellAlpha(p: number): number {
  const x = Math.max(0, Math.min(1, p));
  return Math.pow(x, 0.5);
}

export default function LogitLensView({ resp }: Props) {
  const [hover, setHover] = useState<{ layer: number; pos: number } | null>(null);

  if (!resp) {
    return <div className="text-sm text-slate-500">Run a prompt to see the logit lens.</div>;
  }

  const cells = resp.layers;

  return (
    <div className="space-y-3">
      <div className="text-xs text-slate-500">
        Each row is one transformer block's residual stream projected through the model's
        own LM head (final layer norm + final linear). Each cell is the top-1 predicted
        <em> next</em> token at that position, color-coded by probability (√-scaled).
        Last layer = the model's actual prediction.
      </div>

      <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs">
        {hover ? (
          <div>
            <div className="text-slate-500">
              L{hover.layer} · pos={hover.pos} (input "{escape(resp.token_strings[hover.pos]!)}") → top-{resp.top_k}:
            </div>
            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-slate-800">
              {cells[hover.layer]![hover.pos]!.map((c, i) => (
                <span key={i}>
                  <span className="text-slate-400">{i + 1}.</span>{" "}
                  "{escape(c.token_string)}"{" "}
                  <span className="font-semibold text-indigo-700">
                    {(c.prob * 100).toFixed(2)}%
                  </span>
                </span>
              ))}
            </div>
          </div>
        ) : (
          <span className="text-slate-400">Hover a cell to see its top-{resp.top_k} predictions.</span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              <th className="border-b border-slate-200 px-2 py-1 text-left font-medium text-slate-500">
                layer
              </th>
              {resp.token_strings.map((t, i) => (
                <th
                  key={i}
                  className="border-b border-slate-200 px-2 py-1 text-left font-mono font-normal text-slate-500"
                >
                  <div className="text-[10px] text-slate-400">pos {i}</div>
                  <div className="text-slate-700">"{escape(t)}"</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cells.map((layer, li) => (
              <tr key={li}>
                <td className="px-2 py-1 font-mono text-slate-500">L{li}</td>
                {layer.map((cell, pi) => {
                  const top = cell[0]!;
                  return (
                    <td
                      key={pi}
                      className="cursor-default border border-white px-2 py-1 font-mono align-top"
                      style={{ backgroundColor: `rgba(79, 70, 229, ${cellAlpha(top.prob)})` }}
                      onMouseEnter={() => setHover({ layer: li, pos: pi })}
                      onMouseLeave={() => setHover(null)}
                    >
                      <div className="text-slate-900">"{escape(top.token_string)}"</div>
                      <div className="text-[10px] text-slate-600">
                        {(top.prob * 100).toFixed(2)}%
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
