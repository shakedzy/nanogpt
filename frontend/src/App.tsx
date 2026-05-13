import { useEffect, useState } from "react";
import {
  getHealth,
  postForward,
  postAttention,
  postLogitLens,
  ForwardResponse,
  AttentionResponse,
  LogitLensResponse,
} from "./api";
import PromptInput from "./components/PromptInput";
import ForwardView from "./components/ForwardView";
import AttentionView from "./components/AttentionView";
import LogitLensView from "./components/LogitLensView";
import NeuronBrowser from "./components/NeuronBrowser";
import InductionView from "./components/InductionView";

const TABS = [
  { id: "forward", label: "Forward" },
  { id: "attention", label: "Attention" },
  { id: "logit_lens", label: "Logit Lens" },
  { id: "neurons", label: "Neurons" },
  { id: "induction", label: "Induction" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>("attention");
  const [health, setHealth] = useState<unknown>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forward, setForward] = useState<ForwardResponse | null>(null);
  const [attention, setAttention] = useState<AttentionResponse | null>(null);
  const [logitLens, setLogitLens] = useState<LogitLensResponse | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setHealthErr(String(e)));
  }, []);

  const run = async (prompt: string) => {
    setLoading(true);
    setError(null);
    try {
      const [f, a, l] = await Promise.all([
        postForward(prompt),
        postAttention(prompt),
        postLogitLens(prompt),
      ]);
      setForward(f);
      setAttention(a);
      setLogitLens(l);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl p-6">
      <header className="mb-4 flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold text-slate-900">NanoGPT Interp</h1>
        <span className="text-xs text-slate-500">
          {healthErr ?? (health ? JSON.stringify(health) : "checking server...")}
        </span>
      </header>

      <div className="mb-4">
        <PromptInput onRun={run} loading={loading} />
      </div>

      {error && (
        <pre className="mb-4 rounded bg-red-50 p-3 text-xs text-red-700">{error}</pre>
      )}

      <nav className="mb-6 flex gap-1 border-b border-slate-200">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={
              "px-3 py-2 text-sm " +
              (tab === t.id
                ? "border-b-2 border-indigo-600 font-medium text-indigo-700"
                : "text-slate-500 hover:text-slate-800")
            }
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "forward" && <ForwardView resp={forward} />}
      {tab === "attention" && <AttentionView resp={attention} />}
      {tab === "logit_lens" && <LogitLensView resp={logitLens} />}
      {tab === "neurons" && <NeuronBrowser />}
      {tab === "induction" && <InductionView />}
    </div>
  );
}
