'use client';
import { useEffect, useState } from 'react';
import { ForecastCard } from '@/components/ForecastCard';
import { RegionSelector } from '@/components/RegionSelector';
import { HistoryStrip } from '@/components/HistoryStrip';
import { ChatBox } from '@/components/ChatBox';

type HistoryPoint = {
  date: string;
  chlor_a_mean: number | null;
  sst_mean: number | null;
  bloom: number;
};
type Forecast = {
  window_start: string;
  window_end: string;
  model_version: string;
  regions: Record<string, {
    P_bloom_next_week: number;
    risk_band: string;
    confidence_note: string;
  }>;
  recent_history?: Record<string, HistoryPoint[]>;
};

export default function Page() {
  const [region, setRegion] = useState<string>('Kerala');
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setForecast(null);
    setError(null);
    fetch('/api/backend/forecast')
      .then((r) => r.json())
      .then((d) => setForecast(d))
      .catch((e) => setError(e.message));
  }, []);

  return (
    <main className="mx-auto max-w-2xl px-5 py-8 md:py-14">
      {/* Header */}
      <header className="mb-6">
        <h1 className="text-3xl font-serif tracking-tight text-teal-950">
          Bloom<em>Watch</em>
          <span className="ml-1 text-base italic opacity-70">AI</span>
        </h1>
        <div className="text-xs uppercase tracking-widest text-teal-950/60 mt-1">
          Harmful algal bloom early warning · Kerala + Karnataka coasts
        </div>
      </header>

      {/* Region selector */}
      <div className="flex items-center justify-between mb-4">
        <RegionSelector region={region} onChange={setRegion} />
        {forecast && (
          <div className="text-xs text-teal-950/60 font-mono">
            {forecast.model_version}
          </div>
        )}
      </div>

      {/* Forecast card */}
      <ForecastCard forecast={forecast} region={region} />

      {/* Confidence caption */}
      {forecast && (
        <p className="mt-3 text-xs leading-relaxed text-teal-950/70">
          {forecast.regions[region]?.confidence_note}
        </p>
      )}

      {error && (
        <div className="mt-4 rounded-lg bg-red-100 text-red-800 p-3 text-sm">
          Could not reach the backend at localhost:8000 — is the API running?
          <br />Run <code className="font-mono">python api/main.py</code> in bloomwatch-app.
          <br />Detail: {error}
        </div>
      )}

      {/* Recent history strip */}
      <HistoryStrip history={forecast?.recent_history?.[region] || null} />

      {/* Chatbot */}
      <ChatBox region={region} />

      {/* Provenance strip */}
      <div className="mt-8 border-t border-teal-950/10 pt-4">
        <div className="text-xs uppercase tracking-widest text-teal-950/60 mb-2">
          Where the numbers come from
        </div>
        <ul className="text-xs text-teal-950/70 space-y-1">
          <li><b>Chlorophyll-a + SST</b>: NASA Suomi-NPP VIIRS 8-day composite (4 km, near real-time)</li>
          <li><b>Rainfall</b>: IMD district-level cumulative rainfall (mausam.imd.gov.in)</li>
          <li><b>Bloom history</b>: CMFRI Annual Reports 2016–2024, extracted event log</li>
          <li><b>Model</b>: XGBoost, 460 weekly training rows, 69 engineered features, 2024 AUC 0.83</li>
        </ul>
      </div>

      {/* Footer */}
      <footer className="mt-8 pb-6 text-xs text-teal-950/40 text-center">
        BloomWatch AI · Refreshed every Friday · Not a substitute for local advisories
      </footer>
    </main>
  );
}
