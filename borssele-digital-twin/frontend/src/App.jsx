import React, { useState, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import CoverageMap from './components/CoverageMap';
import ControlsHeader from './components/ControlsHeader';
import OperationalSummary from './components/OperationalSummary';
import ForecastChart from './components/ForecastChart';

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
  { id: 'ciara_2020_oos', label: 'Historical Event 4: Storm Ciara (2020-02-09 • Out-of-Sample Cut-Out)', ts: '2020-02-09 18:00:00', sysMode: 'normal', area: 'borssele' },
  { id: 'custom_datetime', label: 'Custom Date & Hour Selection (Year / Month / Day / Hour)', ts: 'CUSTOM', sysMode: 'normal' },
];

const YEAR_OPTIONS = ['2020', '2021', '2022', '2023', '2024', '2025', '2026'];
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
        <ControlsHeader
          loading={loading}
          resolvedTimestamp={resolvedTimestamp}
          tzMode={tzMode}
          localTzLabel={localTzLabel}
          handleToggleTzMode={handleToggleTzMode}
          lastAutoRefresh={lastAutoRefresh}
          selectedAreaId={selectedAreaId}
          handleAreaChange={handleAreaChange}
          areaOptions={AREA_OPTIONS}
          horizonHours={horizonHours}
          setHorizonHours={setHorizonHours}
          scenarioId={scenarioId}
          handleScenarioChange={handleScenarioChange}
          scenarioOptions={SCENARIO_OPTIONS}
          clampedYear={clampedYear}
          clampedMonth={clampedMonth}
          clampedDay={clampedDay}
          clampedHour={clampedHour}
          availableYears={availableYears}
          availableMonths={availableMonths}
          maxAllowedDay={maxAllowedDay}
          availableHours={availableHours}
          handleDatePartChange={handleDatePartChange}
          handleApplyDateClick={handleApplyDateClick}
          effectiveTimestamp={effectiveTimestamp}
          setScenarioId={setScenarioId}
          formatDisplayTimestamp={formatDisplayTimestamp}
        />

        {error && (
          <div className="bg-red-950/50 border border-red-800/80 text-red-300 px-5 py-3.5 rounded-xl flex items-center gap-3 text-sm">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* 2. INTERACTIVE HORIZON SUMMARY CARDS (+1h, +6h, +12h, +24h) */}
        <OperationalSummary
          horizonSnapshots={horizonSnapshots}
          horizonHours={horizonHours}
          setHorizonHours={setHorizonHours}
          tzMode={tzMode}
          localTzLabel={localTzLabel}
          formatDisplayTimestamp={formatDisplayTimestamp}
        />

        {/* 3. MAIN WORKSPACE: COVERAGE MAP (LEFT) + UNIFIED FORECAST & THRESHOLD CHART (RIGHT) */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-7 items-stretch">
          <div className="lg:col-span-6">
            <CoverageMap
              area={activeAreaObj}
              windSpeed={targetSpeed}
              windDirDeg={windDirDeg}
              horizonHours={horizonHours}
            />
          </div>

          <ForecastChart
            horizonHours={horizonHours}
            areaTitle={areaTitle}
            windowMetrics={windowMetrics}
            forecastData={forecastData}
            isLiveMode={isLiveMode}
            proposalTelemetry={proposalTelemetry}
            tzMode={tzMode}
            localTzLabel={localTzLabel}
            formatDisplayTimestamp={formatDisplayTimestamp}
          />
        </section>

      </div>
    </div>
  );
}
