import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine
} from 'recharts';
import { Activity, CheckCircle2 } from 'lucide-react';

export default function ForecastChart({
  horizonHours,
  areaTitle,
  windowMetrics,
  forecastData,
  isLiveMode,
  proposalTelemetry,
  tzMode,
  localTzLabel,
  formatDisplayTimestamp,
}) {
  return (
    <div className="lg:col-span-6 bg-slate-900/85 backdrop-blur p-6 rounded-2xl border border-slate-800 shadow-xl flex flex-col justify-between">
      <div>
        <div className="flex flex-wrap justify-between items-center gap-2 mb-4">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>+{horizonHours}h Wind Forecast &amp; Shut-Down Thresholds — {areaTitle}</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              <span className="text-cyan-400 font-semibold">&lt; 3 m/s: Low Wind</span> •{' '}
              <span className="text-emerald-400 font-semibold">3–20 m/s: Good Conditions</span> •{' '}
              <span className="text-amber-400 font-semibold">20–25 m/s: High Wind Caution</span> •{' '}
              <span className="text-red-400 font-semibold">≥ 25 m/s: Cut-Out Shut-Down</span>
            </p>
          </div>

          {windowMetrics && windowMetrics.rmse_active !== null && (
            <span className="text-xs font-mono bg-slate-950 border border-slate-800 px-3 py-1 rounded-lg text-slate-300">
              Wind Accuracy: <strong className="text-emerald-400">{windowMetrics.operability_accuracy_pct}%</strong> • MAE:{' '}
              <strong className="text-amber-300">{windowMetrics.mae_active} m/s</strong> • RMSE:{' '}
              <strong className="text-cyan-300">{windowMetrics.rmse_active} m/s</strong>
            </span>
          )}
        </div>

        <div className="h-[340px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecastData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="horizon" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} unit=" m/s" domain={[0, 35]} />
              <Tooltip
                labelFormatter={(label, payload) => {
                  const ts = payload?.[0]?.payload?.timestamp;
                  return ts ? `${label} • ${formatDisplayTimestamp(ts, tzMode, localTzLabel, false)}` : label;
                }}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '12px',
                  color: '#fff',
                }}
              />
              <Legend />
              <ReferenceLine
                y={25.0}
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: 'Cut-Out Shut-Down (>25 m/s)', fill: '#f87171', fontSize: 10, position: 'insideTopRight' }}
              />
              <ReferenceLine
                y={20.0}
                stroke="#f59e0b"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                label={{ value: 'High Wind Caution (20–25 m/s)', fill: '#fbbf24', fontSize: 10, position: 'insideTopLeft' }}
              />
              <ReferenceLine
                y={3.0}
                stroke="#10b981"
                strokeWidth={1.2}
                strokeDasharray="2 4"
                label={{ value: 'Good Conditions (3–20 m/s) | Low Wind (<3 m/s)', fill: '#6ee7b7', fontSize: 10, position: 'insideBottomRight' }}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                name="Predicted 100m Wind (m/s)"
                stroke="#06b6d4"
                strokeWidth={3}
                dot={horizonHours <= 6}
              />
              <Line
                type="monotone"
                dataKey="actual"
                name={isLiveMode ? 'Secondary NWP Reference (m/s)' : 'ERA5 Verification Truth (m/s)'}
                stroke="#10b981"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={horizonHours <= 6}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Concise Grid Operator Decision & Proposal Performance Criteria Footer */}
      {proposalTelemetry && (
        <div className="mt-4 pt-3 border-t border-slate-800/80 bg-slate-950/70 rounded-xl p-3.5 border border-slate-800/80 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            <span className="text-cyan-400 font-bold flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              Grid Operator Decision (+{horizonHours}h Window)
            </span>
            <span className="text-emerald-400">
              {windowMetrics ? `${windowMetrics.operable_hours}/${horizonHours}h Operable` : ''} • Consensus:{' '}
              {proposalTelemetry.consensus_agreement_pct}%
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            {proposalTelemetry.grid_operator_strategy}
          </p>
          <div className="pt-1.5 border-t border-slate-800/70 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-[11px] font-mono text-slate-400">
            <span>
              Mode:{' '}
              <strong className={proposalTelemetry.auto_fallback_triggered ? 'text-amber-300' : 'text-slate-200'}>
                {proposalTelemetry.system_mode}
              </strong>
            </span>
            <span>
              Freshness: <strong className="text-emerald-400">{proposalTelemetry.data_freshness}</strong> • Cycle:{' '}
              <strong className="text-slate-200">1h</strong>
            </span>
            <span>
              Latency: <strong className="text-cyan-300">{proposalTelemetry.detection_to_alert_latency_ms} ms</strong> • Fallback:{' '}
              <strong className={proposalTelemetry.fallback_status?.startsWith('ACTIVE') ? 'text-amber-300' : 'text-emerald-400'}>
                {proposalTelemetry.fallback_status}
              </strong>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
