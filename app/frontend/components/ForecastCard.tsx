type Forecast = {
  window_start: string;
  window_end: string;
  regions: Record<string, {
    P_bloom_next_week: number;
    risk_band: string;
    confidence_note: string;
  }>;
};

const bandColor: Record<string, string> = {
  'Low':       'bg-teal-950 text-cream',
  'Elevated':  'bg-amber-500 text-teal-950',
  'High':      'bg-orange-600 text-cream',
  'Very High': 'bg-red-700 text-cream',
};

export function ForecastCard({ forecast, region }: { forecast: Forecast | null; region: string }) {
  if (!forecast) {
    return (
      <div className="rounded-2xl bg-white/40 p-6 shadow-inner animate-pulse">
        <div className="h-4 bg-teal-950/20 rounded w-1/3 mb-3" />
        <div className="h-16 bg-teal-950/20 rounded w-1/2 mb-3" />
        <div className="h-3 bg-teal-950/20 rounded w-2/3" />
      </div>
    );
  }
  const r = forecast.regions[region];
  if (!r) return null;
  const pct = (r.P_bloom_next_week * 100).toFixed(1);
  const bg = bandColor[r.risk_band] || bandColor['Low'];

  return (
    <div className={`rounded-2xl ${bg} p-6 shadow-lg`}>
      <div className="text-xs uppercase tracking-widest opacity-80">
        Bloom risk · {region} coast
      </div>
      <div className="mt-2 flex items-end gap-3">
        <div className="text-6xl font-serif leading-none">{pct}%</div>
        <div className="mb-1 text-lg font-semibold uppercase tracking-wide">
          {r.risk_band}
        </div>
      </div>
      <div className="mt-3 text-sm opacity-90">
        Chance of a harmful algal bloom in the 8-day window from
        {' '}{forecast.window_start} to {forecast.window_end}.
      </div>
    </div>
  );
}
