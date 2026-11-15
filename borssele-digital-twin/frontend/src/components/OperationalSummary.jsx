import React from 'react';
import { ShieldCheck, ShieldAlert } from 'lucide-react';
import { evaluateOperabilityClient } from './CoverageMap';

export default function OperationalSummary({
  horizonSnapshots,
  horizonHours,
  setHorizonHours,
  tzMode,
  localTzLabel,
  formatDisplayTimestamp,
}) {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
      {[1, 6, 12, 24].map((h) => {
        const snap = horizonSnapshots[`+${h}h`];
        const isSelected = horizonHours === h;
        const opInfo = snap ? evaluateOperabilityClient(snap.predicted) : null;

        return (
          <div
            key={h}
            onClick={() => setHorizonHours(h)}
            className={`cursor-pointer rounded-2xl p-5 border transition-all ${
              isSelected
                ? 'bg-slate-900 border-cyan-500 shadow-lg shadow-cyan-950/40 ring-1 ring-cyan-500/50'
                : 'bg-slate-900/70 border-slate-800 hover:border-slate-700'
            }`}
          >
            <div className="flex justify-between items-center mb-3">
              <span
                className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-md ${
                  isSelected ? 'bg-cyan-500 text-slate-950' : 'bg-slate-800 text-cyan-300'
                }`}
              >
                +{h}h {h === 24 ? '• Max Horizon' : 'Forecast'}
              </span>
              <span className="text-xs font-mono text-slate-400">
                {snap ? formatDisplayTimestamp(snap.timestamp, tzMode, localTzLabel, false) : '--'}
              </span>
            </div>

            {snap && opInfo ? (
              <div className="space-y-3">
                <div
                  className={`inline-flex items-center gap-1.5 text-xs font-mono font-bold px-3 py-1.5 rounded-xl border w-full justify-center ${opInfo.badgeClass}`}
                >
                  {opInfo.canOperate ? (
                    <ShieldCheck className="w-4 h-4 shrink-0" />
                  ) : (
                    <ShieldAlert className="w-4 h-4 shrink-0" />
                  )}
                  <span>{opInfo.shortLabel}</span>
                </div>

                <div className="flex items-baseline justify-between">
                  <div className="text-2xl font-mono font-black text-white">
                    {snap.predicted} <span className="text-xs font-normal text-cyan-400">m/s</span>
                  </div>
                  <div className="text-xs font-mono text-slate-400">
                    Margin: <strong className="text-slate-200">{snap.marginToCutOut} m/s</strong>
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-4 text-xs font-mono text-slate-500">Syncing +{h}h...</div>
            )}
          </div>
        );
      })}
    </section>
  );
}
