const REGIONS = ['Kerala', 'Karnataka'] as const;

export function RegionSelector({
  region, onChange,
}: { region: string; onChange: (r: string) => void }) {
  return (
    <div className="inline-flex rounded-full bg-teal-950/10 p-1">
      {REGIONS.map((r) => (
        <button
          key={r}
          onClick={() => onChange(r)}
          className={`px-4 py-1.5 text-sm font-semibold rounded-full transition ${
            region === r ? 'bg-teal-950 text-cream' : 'text-teal-950 hover:bg-teal-950/10'
          }`}
        >
          {r}
        </button>
      ))}
    </div>
  );
}
