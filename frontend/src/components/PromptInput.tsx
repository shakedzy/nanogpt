import { useState } from "react";

type Props = {
  initial?: string;
  onRun: (prompt: string) => void;
  loading: boolean;
};

export default function PromptInput({ initial = "Once upon a time", onRun, loading }: Props) {
  const [value, setValue] = useState(initial);
  return (
    <div className="flex gap-2">
      <input
        className="flex-1 rounded border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !loading) onRun(value);
        }}
        placeholder="Type a prompt..."
      />
      <button
        className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        disabled={loading || !value.trim()}
        onClick={() => onRun(value)}
      >
        {loading ? "Running..." : "Run"}
      </button>
    </div>
  );
}
