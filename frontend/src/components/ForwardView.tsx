import { ForwardResponse } from "../api";

type Props = { resp: ForwardResponse | null };

export default function ForwardView({ resp }: Props) {
  if (!resp) {
    return <div className="text-sm text-slate-500">Run a prompt to see token IDs and logits.</div>;
  }
  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-slate-700">Tokens</h2>
        <pre className="mt-1 overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
          {JSON.stringify(resp.tokens)}
        </pre>
      </div>
      <div>
        <h2 className="text-sm font-semibold text-slate-700">Decoded per-token</h2>
        <pre className="mt-1 overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
          {JSON.stringify(resp.token_strings)}
        </pre>
      </div>
      <div>
        <h2 className="text-sm font-semibold text-slate-700">Raw response</h2>
        <pre className="mt-1 max-h-96 overflow-auto rounded bg-slate-50 p-3 text-xs text-slate-700">
          {JSON.stringify(
            { ...resp, logits: `[${resp.logits_shape.join(" x ")} floats omitted]` },
            null,
            2,
          )}
        </pre>
      </div>
    </section>
  );
}
