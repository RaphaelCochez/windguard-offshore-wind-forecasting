import React from 'react';
import { Wind, RefreshCw, MapPin, Clock, Layers, Calendar } from 'lucide-react';

export default function ControlsHeader({
  loading,
  resolvedTimestamp,
  tzMode,
  localTzLabel,
  handleToggleTzMode,
  lastAutoRefresh,
  selectedAreaId,
  handleAreaChange,
  areaOptions,
  horizonHours,
  setHorizonHours,
  scenarioId,
  handleScenarioChange,
  scenarioOptions,
  clampedYear,
  clampedMonth,
  clampedDay,
  clampedHour,
  availableYears,
  availableMonths,
  maxAllowedDay,
  availableHours,
  handleDatePartChange,
  handleApplyDateClick,
  effectiveTimestamp,
  setScenarioId,
  formatDisplayTimestamp,
}) {
  return (
    <header className="bg-slate-900/90 backdrop-blur p-6 rounded-2xl border border-slate-800 shadow-xl space-y-5">
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-cyan-400 font-semibold text-xs uppercase tracking-wider">
            <Wind className="w-4 h-4 animate-spin shrink-0" />
            <span>Copernicus C3S ERA5 &amp; NWP • Max Forecast Horizon: +24h</span>
          </div>
          <h1 className="text-2xl font-black text-white mt-1 tracking-tight">
            Offshore Wind Park Shut-Down Prediction Service
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 bg-slate-950/90 border border-slate-800 rounded-xl px-4 py-2.5 text-xs font-mono shrink-0">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-cyan-400' : 'text-emerald-400'}`} />
          <span className="text-emerald-400 font-bold">Auto-Sync Active</span>
          <span className="text-slate-500">•</span>
          <span className="text-slate-300">
            t₀: {resolvedTimestamp ? formatDisplayTimestamp(resolvedTimestamp, tzMode, localTzLabel, true) : 'Syncing...'}
          </span>
          <div className="inline-flex rounded-lg border border-slate-700 overflow-hidden ml-1">
            <button
              type="button"
              onClick={() => handleToggleTzMode('local')}
              className={`px-2 py-0.5 text-[11px] font-bold transition ${
                tzMode === 'local' ? 'bg-cyan-500 text-slate-950' : 'bg-slate-900 text-slate-400 hover:text-white'
              }`}
            >
              {localTzLabel}
            </button>
            <button
              type="button"
              onClick={() => handleToggleTzMode('utc')}
              className={`px-2 py-0.5 text-[11px] font-bold transition ${
                tzMode === 'utc' ? 'bg-cyan-500 text-slate-950' : 'bg-slate-900 text-slate-400 hover:text-white'
              }`}
            >
              UTC
            </button>
          </div>
          {lastAutoRefresh && (
            <>
              <span className="text-slate-500">•</span>
              <span className="text-slate-400">Updated {lastAutoRefresh}</span>
            </>
          )}
        </div>
      </div>

      {/* Unified Menu: Corridor + Horizon (+1h, +6h, +12h, +24h) + Scenario/Mode */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 pt-4 border-t border-slate-800/80 items-end">
        <div className="md:col-span-5 space-y-1.5">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <MapPin className="w-3.5 h-3.5 text-emerald-400" />
            Offshore Wind Corridor (Area)
          </label>
          <select
            value={selectedAreaId}
            onChange={(e) => handleAreaChange(e.target.value)}
            className="w-full bg-slate-950 text-white text-xs sm:text-sm font-mono font-semibold px-3.5 py-2.5 rounded-xl border border-slate-700 focus:border-cyan-500 focus:outline-none"
          >
            {areaOptions.map((a) => (
              <option key={a.id} value={a.id}>
                {a.label}
              </option>
            ))}
          </select>
        </div>

        <div className="md:col-span-3 space-y-1.5">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            Prediction Horizon (Max +24h)
          </label>
          <div className="grid grid-cols-4 bg-slate-950 p-1 rounded-xl border border-slate-700">
            {[1, 6, 12, 24].map((h) => (
              <button
                key={h}
                onClick={() => setHorizonHours(h)}
                className={`py-1.5 rounded-lg text-xs font-mono font-bold transition ${
                  horizonHours === h
                    ? 'bg-cyan-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                +{h}h
              </button>
            ))}
          </div>
        </div>

        <div className="md:col-span-4 space-y-1.5">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            Operational Mode / Historical Event
          </label>
          <select
            value={scenarioId}
            onChange={(e) => handleScenarioChange(e.target.value)}
            className="w-full bg-slate-950 text-cyan-300 text-xs sm:text-sm font-mono font-semibold px-3.5 py-2.5 rounded-xl border border-slate-700 focus:border-cyan-500 focus:outline-none"
          >
            {scenarioOptions.map((sc) => (
              <option key={sc.id} value={sc.id}>
                {sc.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Date & Hour (Year / Month / Day / Hour) Menu — Strictly capped at current date & hour */}
      <div className="pt-4 border-t border-slate-800/80 flex flex-col lg:flex-row items-stretch lg:items-end justify-between gap-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 flex-1">
          <div className="space-y-1">
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1">
              <Calendar className="w-3 h-3 text-cyan-400" />
              Year
            </label>
            <select
              value={clampedYear}
              onChange={(e) => handleDatePartChange('year', e.target.value)}
              className="w-full bg-slate-950 text-white text-xs sm:text-sm font-mono font-semibold px-3 py-2 rounded-xl border border-slate-700 focus:border-cyan-500 focus:outline-none"
            >
              {availableYears.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Month
            </label>
            <select
              value={clampedMonth}
              onChange={(e) => handleDatePartChange('month', e.target.value)}
              className="w-full bg-slate-950 text-white text-xs sm:text-sm font-mono font-semibold px-3 py-2 rounded-xl border border-slate-700 focus:border-cyan-500 focus:outline-none"
            >
              {availableMonths.map((m) => (
                <option key={m.val} value={m.val}>{m.label}</option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Day
            </label>
            <select
              value={clampedDay}
              onChange={(e) => handleDatePartChange('day', e.target.value)}
              className="w-full bg-slate-950 text-white text-xs sm:text-sm font-mono font-semibold px-3 py-2 rounded-xl border border-slate-700 focus:border-cyan-500 focus:outline-none"
            >
              {Array.from({ length: maxAllowedDay }, (_, i) => String(i + 1).padStart(2, '0')).map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Hour ({tzMode === 'utc' ? 'UTC' : localTzLabel} • Max Now)
            </label>
            <select
              value={clampedHour}
              onChange={(e) => handleDatePartChange('hour', e.target.value)}
              className="w-full bg-slate-950 text-cyan-300 text-xs sm:text-sm font-mono font-semibold px-3 py-2 rounded-xl border border-slate-700 focus:border-cyan-500 focus:outline-none"
            >
              {availableHours.map((hr) => (
                <option key={hr} value={hr}>{hr}:00 {tzMode === 'utc' ? 'UTC' : localTzLabel}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          <button
            onClick={handleApplyDateClick}
            className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono font-bold text-xs transition shadow"
          >
            Predict {clampedYear}-{clampedMonth}-{clampedDay} {clampedHour}:00 {tzMode === 'utc' ? 'UTC' : localTzLabel}
          </button>
          {effectiveTimestamp !== 'LIVE' && (
            <button
              onClick={() => setScenarioId('live_normal')}
              className="px-3.5 py-2 rounded-xl bg-slate-950 hover:bg-slate-800 text-emerald-400 border border-emerald-700/60 font-mono font-bold text-xs transition"
            >
              Live Now
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
