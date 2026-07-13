type HistoryPoint = {
  date: string;
  chlor_a_mean: number | null;
  sst_mean: number | null;
  bloom: number;
};

export function HistoryStrip({ history }: { history: HistoryPoint[] | null }) {
  if (!history || history.length === 0) return null;
  return (
    <div className="mt-4">
      <div className="text-xs uppercase tracking-widest text-teal-950/60 mb-2">
        Last 4 windows
      </div>
      <div className="flex gap-2">
        {history.map((p) => (
          <div
            key={p.date}
            className={`flex-1 rounded-lg border p-2 text-xs ${
              p.bloom ? 'border-red-600 bg-red-50' : 'border-teal-950/20 bg-white/30'
            }`}
          >
            <div className="font-semibold">{p.date}</div>
            <div className="opacity-70">
              chl: {p.chlor_a_mean == null ? '—' : p.chlor_a_mean.toFixed(2)}
            </div>
            <div className="opacity-70">
              sst: {p.sst_mean == null ? '—' : p.sst_mean.toFixed(1)}°C
            </div>
            {p.bloom ? (
              <div className="mt-1 font-semibold text-red-700">bloom</div>
            ) : (
              <div className="mt-1 opacity-50">clear</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
