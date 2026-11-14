import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine
} from 'recharts';
import {
  Wind, RefreshCw, AlertTriangle, Activity, Clock, ShieldCheck, ShieldAlert, MapPin, CheckCircle2, Layers, Calendar
} from 'lucide-react';
import CoverageMap, { evaluateOperabilityClient } from './components/CoverageMap';

const AREA_OPTIONS = [
  { id: 'borssele', label: 'Borssele Corridor (NL • 51.72°N, 3.00°E)' },
  { id: 'hollandse_kust', label: 'Hollandse Kust Corridor (NL • 52.30°N, 4.10°E)' },
  { id: 'dogger_bank', label: 'Dogger Bank Corridor (UK • 54.75°N, 2.25°E)' },
  { id: 'hornsea', label: 'Hornsea Corridor (UK • 53.88°N, 1.80°E)' },
  { id: 'tahkoluoto', label: 'Tahkoluoto (FI • 61.63°N, 21.36°E)' },
];

const SCENARIO_OPTIONS = [
  { id: 'live_normal', label: 'Live Now (Auto-Synced ERA5 + ML Ensemble)', ts: 'LIVE', sysMode: 'normal' },
  { id: 'live_fallback', label: 'Live Safe Fallback (Secondary NWP Stream)', ts: 'LIVE', sysMode: 'degraded_fallback' },
  { id: 'eunice_shutdown', label: 'Historical Event 1: Storm Eunice (2022-02-18 • 31.3 m/s Cut-Out Shut-Down)', ts: '2022-02-18 06:00:00', sysMode: 'normal', area: 'borssele' },
  { id: 'ciaran_shutdown', label: 'Historical Event 2: Storm Ciarán (2023-11-02 • 32.6 m/s Cut-Out Shut-Down)', ts: '2023-11-02 00:00:00', sysMode: 'normal', area: 'borssele' },
  { id: 'event_3', label: 'Historical Event 3: New Year Gale (2025-01-01 • 24.5 m/s High Wind Caution)', ts: '2025-01-01 06:00:00', sysMode: 'normal', area: 'borssele' },
  { id: 'custom_datetime', label: 'Custom Date & Hour Selection (Year / Month / Day / Hour)', ts: 'CUSTOM', sysMode: 'normal' },
];

const YEAR_OPTIONS = ['2022', '2023', '2024', '2025', '2026'];
const MONTH_OPTIONS = [
  { val: '01', label: '01 (Jan)' },
  { val: '02', label: '02 (Feb)' },
  { val: '03', label: '03 (Mar)' },
  { val: '04', label: '04 (Apr)' },
  { val: '05', label: '05 (May)' },
  { val: '06', label: '06 (Jun)' },
  { val: '07', label: '07 (Jul)' },
  { val: '08', label: '08 (Aug)' },
  { val: '09', label: '09 (Sep)' },
  { val: '10', label: '10 (Oct)' },
  { val: '11', label: '11 (Nov)' },
  { val: '12', label: '12 (Dec)' },
];
const HOUR_OPTIONS = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0'));

function getDaysInMonth(yearStr, monthStr) {
  const y = parseInt(yearStr, 10) || 2025;
  const m = parseInt(monthStr, 10) || 1;
  return new Date(y, m, 0).getDate();
}

function getLocalTzLabel() {
  const offsetHours = -new Date().getTimezoneOffset() / 60;
  if (offsetHours === 0) return 'UTC';
  return `UTC${offsetHours > 0 ? `+${offsetHours}` : offsetHours}`;
}

function parseUtcTimestamp(tsStr) {
  if (!tsStr || tsStr === 'LIVE' || tsStr === 'CUSTOM') return null;
  const normalized = tsStr.trim().replace(' ', 'T');
  const iso = normalized.endsWith('Z') ? normalized : `${normalized}Z`;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatParts(d, useUtc) {
  const pad = (n) => String(n).padStart(2, '0');
  const year = String(useUtc ? d.getUTCFullYear() : d.getFullYear());
  const month = pad((useUtc ? d.getUTCMonth() : d.getMonth()) + 1);
  const day = pad(useUtc ? d.getUTCDate() : d.getDate());
  const hour = pad(useUtc ? d.getUTCHours() : d.getHours());
  const minute = pad(useUtc ? d.getUTCMinutes() : d.getMinutes());
  return { year, month, day, hour, minute };
}

function formatDisplayTimestamp(tsStr, tzMode, localTzLabel, includeSeconds = false) {
  const d = parseUtcTimestamp(tsStr);
  if (!d) return tsStr || '--';
  const useUtc = tzMode === 'utc';
  const p = formatParts(d, useUtc);
  const label = useUtc ? 'UTC' : localTzLabel;
  return includeSeconds
    ? `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:00 ${label}`
    : `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute} ${label}`;
}

function toUtcTimestampString(yearStr, monthStr, dayStr, hourStr, tzMode) {
  const y = parseInt(yearStr, 10) || 2026;
  const m = (parseInt(monthStr, 10) || 1) - 1;
  const d = parseInt(dayStr, 10) || 1;
  const h = parseInt(hourStr, 10) || 0;
  if (tzMode === 'utc') {
    const pad = (n) => String(n).padStart(2, '0');
    return `${y}-${pad(m + 1)}-${pad(d)} ${pad(h)}:00:00`;
  }
  const dt = new Date(y, m, d, h, 0, 0);
  const p = formatParts(dt, true);
  return `${p.year}-${p.month}-${p.day} ${p.hour}:00:00`;
}

export default function BorsseleDigitalTwin() {
  const localTzLabel = getLocalTzLabel();
  const [tzMode, setTzMode] = useState('local'); // Default to browser local time (e.g. UTC+2 in Finland)
  const [selectedAreaId, setSelectedAreaId] = useState('borssele');
  const [scenarioId, setScenarioId] = useState('live_normal');
  const [horizonHours, setHorizonHours] = useState(24); // Strictly 1 | 6 | 12 | 24 (max +24h)

  // Custom Year / Month / Day / Hour selector state
  const [selYear, setSelYear] = useState('2023');
  const [selMonth, setSelMonth] = useState('11');
  const [selDay, setSelDay] = useState('02');
  const [selHour, setSelHour] = useState('00');
  const [appliedCustomTs, setAppliedCustomTs] = useState('2023-11-02 00:00:00');

  const [activeAreaObj, setActiveAreaObj] = useState(null);
  const [resolvedTimestamp, setResolvedTimestamp] = useState('');
  const [isLiveMode, setIsLiveMode] = useState(true);
  const [proposalTelemetry, setProposalTelemetry] = useState(null);
  const [forecastData, setForecastData] = useState([]);
  const [horizonSnapshots, setHorizonSnapshots] = useState({});
  const [currentActual, setCurrentActual] = useState(null);
  const [windDirDeg, setWindDirDeg] = useState(245.0);
  const [windowMetrics, setWindowMetrics] = useState(null);

  const [loading, setLoading] = useState(false);
  const [lastAutoRefresh, setLastAutoRefresh] = useState('');
  const [error, setError] = useState(null);

  const nowObj = new Date();
  const useUtcSelectors = tzMode === 'utc';
  const curYear = useUtcSelectors ? nowObj.getUTCFullYear() : nowObj.getFullYear();
  const curMonth = (useUtcSelectors ? nowObj.getUTCMonth() : nowObj.getMonth()) + 1;
  const curDay = useUtcSelectors ? nowObj.getUTCDate() : nowObj.getDate();
  const curHour = useUtcSelectors ? nowObj.getUTCHours() : nowObj.getHours();

  const availableYears = YEAR_OPTIONS.filter((y) => parseInt(y, 10) <= curYear);
  const clampedYear = String(Math.min(parseInt(selYear, 10) || curYear, curYear));
  const isCurrentYear = parseInt(clampedYear, 10) === curYear;

  const maxAllowedMonth = isCurrentYear ? curMonth : 12;
  const availableMonths = MONTH_OPTIONS.filter((m) => parseInt(m.val, 10) <= maxAllowedMonth);
  const clampedMonth = String(Math.min(parseInt(selMonth, 10) || 1, maxAllowedMonth)).padStart(2, '0');
  const isCurrentMonth = isCurrentYear && parseInt(clampedMonth, 10) === curMonth;

  const daysInSelMonth = getDaysInMonth(clampedYear, clampedMonth);
  const maxAllowedDay = isCurrentMonth ? Math.min(daysInSelMonth, curDay) : daysInSelMonth;
  const clampedDay = String(Math.min(parseInt(selDay, 10) || 1, maxAllowedDay)).padStart(2, '0');
  const isCurrentDay = isCurrentMonth && parseInt(clampedDay, 10) === curDay;

  const maxAllowedHour = isCurrentDay ? curHour : 23;
  const availableHours = HOUR_OPTIONS.filter((hr) => parseInt(hr, 10) <= maxAllowedHour);
  const clampedHour = String(Math.min(parseInt(selHour, 10) || 0, maxAllowedHour)).padStart(2, '0');

  const activeScenario = SCENARIO_OPTIONS.find((s) => s.id === scenarioId) || SCENARIO_OPTIONS[0];
  const pendingCustomTs = toUtcTimestampString(clampedYear, clampedMonth, clampedDay, clampedHour, tzMode);

  const effectiveTimestamp =
    scenarioId === 'custom_datetime'
      ? appliedCustomTs
      : activeScenario.ts;

  const syncSelectorsFromTimestamp = (tsStr, targetTzMode = tzMode) => {
    const d = parseUtcTimestamp(tsStr);
    if (!d) return;
    const p = formatParts(d, targetTzMode === 'utc');
    setSelYear(p.year);
    setSelMonth(p.month);
    setSelDay(p.day);
    setSelHour(p.hour);
    const utcP = formatParts(d, true);
    setAppliedCustomTs(`${utcP.year}-${utcP.month}-${utcP.day} ${utcP.hour}:00:00`);
  };

  const handleToggleTzMode = (nextMode) => {
    setTzMode(nextMode);
    if (resolvedTimestamp) {
      syncSelectorsFromTimestamp(resolvedTimestamp, nextMode);
    }
  };

  const fetchForecast = async () => {
    setLoading(true);
    setError(null);
    const endpoints = ['http://127.0.0.1:8000/api/predict', '/api/predict'];
    let lastErr = null;

    for (const url of endpoints) {
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            timestamp: effectiveTimestamp,
            area_id: selectedAreaId,
            horizon_hours: horizonHours,
            mode: 'ensemble',
            system_mode: activeScenario.sysMode,
          }),
        });

        if (!response.ok) {
          const errJson = await response.json().catch(() => ({}));
          throw new Error(errJson.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setForecastData(data.forecast || []);
        setHorizonSnapshots(data.horizon_snapshots || {});
        setCurrentActual(data.current_actual);
        setResolvedTimestamp(data.timestamp || '');
        setIsLiveMode(Boolean(data.is_live));
        if (data.area) setActiveAreaObj(data.area);
        if (data.proposal_telemetry) setProposalTelemetry(data.proposal_telemetry);
        setWindDirDeg(data.current_wind_dir_deg ?? 245.0);
        setWindowMetrics(data.window_metrics ?? null);
        setLastAutoRefresh(new Date().toLocaleTimeString());
        if (data.timestamp && effectiveTimestamp === 'LIVE') {
          syncSelectorsFromTimestamp(data.timestamp);
        }
        setLoading(false);
        return;
      } catch (err) {
        lastErr = err;
      }
    }

    setError(lastErr?.message || 'Backend connection offline.');
    setLoading(false);
  };

  // Automatic background polling (every 60s) + instant update on any applied control change
  useEffect(() => {
    fetchForecast();
    const intervalId = setInterval(fetchForecast, 60000);
    return () => clearInterval(intervalId);
  }, [selectedAreaId, scenarioId, horizonHours, appliedCustomTs]);

  // Auto-apply custom Year/Month/Day/Hour 500ms after the user finishes selecting
  useEffect(() => {
    if (scenarioId !== 'custom_datetime') return;
    if (pendingCustomTs === appliedCustomTs) return;
    const timer = setTimeout(() => {
      setAppliedCustomTs(pendingCustomTs);
    }, 500);
    return () => clearTimeout(timer);
  }, [selYear, selMonth, selDay, selHour, scenarioId]);

  const handleAreaChange = (newAreaId) => {
    setSelectedAreaId(newAreaId);
  };

  const handleScenarioChange = (newScenarioId) => {
    const sc = SCENARIO_OPTIONS.find((s) => s.id === newScenarioId);
    if (sc?.area && sc.area !== selectedAreaId) {
      setSelectedAreaId(sc.area);
    }
    if (sc && sc.ts !== 'LIVE' && sc.ts !== 'CUSTOM') {
      syncSelectorsFromTimestamp(sc.ts);
    }
    setScenarioId(newScenarioId);
  };

  const handleDatePartChange = (part, value) => {
    if (part === 'year') setSelYear(value);
    if (part === 'month') setSelMonth(value);
    if (part === 'day') setSelDay(value);
    if (part === 'hour') setSelHour(value);
    if (scenarioId !== 'custom_datetime') {
      setScenarioId('custom_datetime');
    }
  };

  const handleApplyDateClick = () => {
    setScenarioId('custom_datetime');
    setAppliedCustomTs(pendingCustomTs);
  };

  const activeWindSpeed = currentActual !== null ? currentActual : 11.4;
  const selectedSnapshot = horizonSnapshots[`+${horizonHours}h`];
  const targetSpeed = selectedSnapshot ? selectedSnapshot.predicted : activeWindSpeed;
  const areaTitle = activeAreaObj ? activeAreaObj.name : 'Borssele Offshore Wind Zone (NL)';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-8 px-4 sm:px-8">
      <div className="max-w-7xl mx-auto space-y-7">

        {/* 1. COMPACT HEADER & UNIFIED CONTROL MENU */}
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
                {AREA_OPTIONS.map((a) => (
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
                {SCENARIO_OPTIONS.map((sc) => (
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

        {error && (
          <div className="bg-red-950/50 border border-red-800/80 text-red-300 px-5 py-3.5 rounded-xl flex items-center gap-3 text-sm">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* 2. INTERACTIVE HORIZON SUMMARY CARDS (+1h, +6h, +12h, +24h) */}
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

        {/* 3. MAIN WORKSPACE: COVERAGE MAP (LEFT) + UNIFIED FORECAST & THRESHOLD CHART (RIGHT) */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-7 items-stretch">

          {/* Left: Interactive Area Coverage Map & Wind Compass */}
          <div className="lg:col-span-6">
            <CoverageMap
              area={activeAreaObj}
              windSpeed={targetSpeed}
              windDirDeg={windDirDeg}
              horizonHours={horizonHours}
            />
          </div>

          {/* Right: Single Unified Forecast & Shut-Down Threshold Chart + Grid Operator Decision */}
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

        </section>

      </div>
    </div>
  );
}
