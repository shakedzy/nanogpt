import { useState } from "react";

type Props = {
  matrix: number[][];      // (T, T), rows = query position, cols = key position
  tokenStrings: string[];  // length T
  size?: number;           // svg side in px
  showLabels?: boolean;
  normalize?: boolean;     // scale each row by its own max for visualization
  onClick?: () => void;
  onHover?: (cell: { q: number; k: number; value: number } | null) => void;
  title?: string;
};

function cellColor(alpha: number): string {
  // alpha in [0, 1] -> opacity on indigo-600
  const a = Math.max(0, Math.min(1, alpha));
  return `rgba(79, 70, 229, ${a})`;
}

export default function AttentionHeatmap({
  matrix,
  tokenStrings,
  size = 128,
  showLabels = false,
  normalize = false,
  onClick,
  onHover,
  title,
}: Props) {
  const T = matrix.length;
  const [hoverRow, setHoverRow] = useState<number | null>(null);
  const [hoverCell, setHoverCell] = useState<{ q: number; k: number } | null>(null);

  const labelPad = showLabels ? 80 : 0;
  const cell = (size - 2) / Math.max(T, 1);
  const rowMax = matrix.map((row) => row.reduce((m, v) => (v > m ? v : m), 0));

  return (
    <div
      className={
        "flex flex-col gap-1 " + (onClick ? "cursor-pointer" : "")
      }
      onClick={onClick}
      onMouseLeave={() => {
        setHoverRow(null);
        setHoverCell(null);
        onHover?.(null);
      }}
    >
      {title && (
        <div className="text-[10px] font-mono text-slate-500 text-center">{title}</div>
      )}
      <svg
        width={size + labelPad}
        height={size + labelPad}
        className="rounded border border-slate-200 bg-white"
        role="img"
      >
        {/* heatmap cells */}
        <g transform={`translate(${labelPad + 1}, ${labelPad + 1})`}>
          {matrix.map((row, q) =>
            row.map((v, k) => {
              const alpha = normalize && rowMax[q]! > 0 ? v / rowMax[q]! : v;
              return (
                <rect
                  key={`${q}-${k}`}
                  x={k * cell}
                  y={q * cell}
                  width={cell}
                  height={cell}
                  fill={cellColor(alpha)}
                  onMouseEnter={() => {
                    setHoverRow(q);
                    setHoverCell({ q, k });
                    onHover?.({ q, k, value: v });
                  }}
                >
                  <title>{`q=${q} (${tokenStrings[q]}) → k=${k} (${tokenStrings[k]}): ${v.toFixed(3)}`}</title>
                </rect>
              );
            }),
          )}
          {/* row highlight overlay */}
          {hoverRow !== null && (
            <rect
              x={0}
              y={hoverRow * cell}
              width={T * cell}
              height={cell}
              fill="none"
              stroke="rgb(220, 38, 38)"
              strokeWidth={1}
              pointerEvents="none"
            />
          )}
        </g>

        {/* labels */}
        {showLabels && (
          <>
            {/* column labels (key tokens) along the top, rotated */}
            {tokenStrings.map((t, k) => (
              <text
                key={`col-${k}`}
                x={labelPad + 1 + k * cell + cell / 2}
                y={labelPad - 4}
                fontSize={10}
                fontFamily="ui-monospace, monospace"
                fill="rgb(71, 85, 105)"
                textAnchor="start"
                transform={`rotate(-60, ${labelPad + 1 + k * cell + cell / 2}, ${labelPad - 4})`}
              >
                {t.replace(/\s/g, "·")}
              </text>
            ))}
            {/* row labels (query tokens) along the left */}
            {tokenStrings.map((t, q) => (
              <text
                key={`row-${q}`}
                x={labelPad - 4}
                y={labelPad + 1 + q * cell + cell / 2 + 3}
                fontSize={10}
                fontFamily="ui-monospace, monospace"
                fill="rgb(71, 85, 105)"
                textAnchor="end"
              >
                {t.replace(/\s/g, "·")}
              </text>
            ))}
          </>
        )}
      </svg>
      {showLabels && hoverCell && (
        <div className="text-xs font-mono text-slate-600">
          q={hoverCell.q} ({tokenStrings[hoverCell.q]!.replace(/\s/g, "·")}) →
          k={hoverCell.k} ({tokenStrings[hoverCell.k]!.replace(/\s/g, "·")}):{" "}
          <span className="font-semibold text-indigo-700">
            {matrix[hoverCell.q]![hoverCell.k]!.toFixed(4)}
          </span>
        </div>
      )}
    </div>
  );
}
