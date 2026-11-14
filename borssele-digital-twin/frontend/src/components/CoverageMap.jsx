import React, { useState, useEffect } from 'react';
import { MapPin, Compass, Layers, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

const TILE_SIZE = 256;

function lonToWorldX(lon, zoom) {
  return ((lon + 180) / 360) * TILE_SIZE * Math.pow(2, zoom);
}

function latToWorldY(lat, zoom) {
  const rad = (lat * Math.PI) / 180;
  return (
    ((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) *
    TILE_SIZE *
    Math.pow(2, zoom)
  );
}

export function getCompassInfo(deg = 0) {
  const normalized = ((Number(deg) % 360) + 360) % 360;
  const dirs8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const names8 = [
    'North',
    'Northeast',
    'East',
    'Southeast',
    'South',
    'Southwest',
    'West',
    'Northwest',
  ];
  const dirs16 = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',
  ];
  const idx8 = Math.round(normalized / 45) % 8;
  const idx16 = Math.round(normalized / 22.5) % 16;
  const toIdx8 = (idx8 + 4) % 8;

  return {
    deg: normalized.toFixed(1),
    short8: dirs8[idx8],
    full8: names8[idx8],
    short16: dirs16[idx16],
    toShort8: dirs8[toIdx8],
    toFull8: names8[toIdx8],
    label: `${dirs8[idx8]} (${names8[idx8]})`,
    flowLabel: `From ${names8[idx8]} (${dirs16[idx16]}) → Toward ${names8[toIdx8]} (${dirs8[toIdx8]})`,
  };
}

export function evaluateOperabilityClient(windSpeed) {
  if (windSpeed >= 25.0) {
    return {
      canOperate: false,
      stateCode: 'CUT_OUT_SHUTDOWN',
      shortLabel: 'Cut-Out Shut-Down (≥25 m/s)',
      badgeClass: 'bg-red-950/90 text-red-300 border-red-700/70',
      dotColor: '#ef4444',
    };
  }
  if (windSpeed >= 20.0) {
    return {
      canOperate: true,
      stateCode: 'HIGH_WIND_CAUTION',
      shortLabel: 'High Wind Caution (20–25 m/s)',
      badgeClass: 'bg-amber-950/90 text-amber-300 border-amber-600/70',
      dotColor: '#fbbf24',
    };
  }
  if (windSpeed < 3.0) {
    return {
      canOperate: true,
      stateCode: 'LOW_WIND_STANDBY',
      shortLabel: 'Low Wind (<3 m/s • Can Operate)',
      badgeClass: 'bg-cyan-950/90 text-cyan-300 border-cyan-600/70',
      dotColor: '#38bdf8',
    };
  }
  return {
    canOperate: true,
    stateCode: 'GOOD_CONDITIONS',
    shortLabel: 'Good Conditions (3–20 m/s)',
    badgeClass: 'bg-emerald-950/90 text-emerald-300 border-emerald-600/70',
    dotColor: '#10b981',
  };
}

const DEFAULT_AREA = {
  id: 'borssele',
  name: 'Borssele Offshore Wind Zone (NL)',
  region: 'Southern North Sea • Netherlands',
  lat: 51.716667,
  lon: 3.0,
  grid_center: [51.75, 3.0],
  era5_box: {
    north: 51.875,
    south: 51.625,
    west: 2.875,
    east: 3.125,
    area_km2: 478,
  },
  zone_area_km2: 344,
  default_zoom: 9,
  map_center: [51.68, 3.18],
  zone_polygon: [
    [51.792, 2.915],
    [51.795, 3.085],
    [51.71, 3.112],
    [51.632, 3.058],
    [51.628, 2.942],
    [51.695, 2.888],
  ],
  landmarks: [
    { name: 'Westkapelle (NL Coast)', lat: 51.528, lon: 3.44 },
    { name: 'Vlissingen Port', lat: 51.442, lon: 3.573 },
    { name: 'Burgh-Haamstede', lat: 51.696, lon: 3.712 },
    { name: 'Zeebrugge (BE)', lat: 51.331, lon: 3.207 },
  ],
};

export default function CoverageMap({
  area = DEFAULT_AREA,
  windSpeed = 11.4,
  windDirDeg = 240,
  horizonHours = 24,
}) {
  const activeArea = area || DEFAULT_AREA;
  const [zoom, setZoom] = useState(activeArea.default_zoom || 9);
  const [mapStyle, setMapStyle] = useState('osm');
  const [showSeamarks, setShowSeamarks] = useState(true);

  useEffect(() => {
    setZoom(activeArea.default_zoom || 9);
  }, [activeArea.id]);

  const compass = getCompassInfo(windDirDeg);
  const zoneOp = evaluateOperabilityClient(windSpeed);

  const centerLat = activeArea.map_center ? activeArea.map_center[0] : activeArea.lat;
  const centerLon = activeArea.map_center ? activeArea.map_center[1] : activeArea.lon;
  const viewWidth = 760;
  const viewHeight = 360;

  const centerWorldX = lonToWorldX(centerLon, zoom);
  const centerWorldY = latToWorldY(centerLat, zoom);
  const topLeftWorldX = centerWorldX - viewWidth / 2;
  const topLeftWorldY = centerWorldY - viewHeight / 2;

  const project = (lat, lon) => ({
    x: lonToWorldX(lon, zoom) - topLeftWorldX,
    y: latToWorldY(lat, zoom) - topLeftWorldY,
  });

  const minTileX = Math.floor(topLeftWorldX / TILE_SIZE);
  const maxTileX = Math.floor((topLeftWorldX + viewWidth) / TILE_SIZE);
  const minTileY = Math.floor(topLeftWorldY / TILE_SIZE);
  const maxTileY = Math.floor((topLeftWorldY + viewHeight) / TILE_SIZE);

  const tiles = [];
  const maxIndex = Math.pow(2, zoom);
  for (let tx = minTileX; tx <= maxTileX; tx++) {
    for (let ty = minTileY; ty <= maxTileY; ty++) {
      if (ty >= 0 && ty < maxIndex) {
        const wrappedTx = ((tx % maxIndex) + maxIndex) % maxIndex;
        const left = tx * TILE_SIZE - topLeftWorldX;
        const top = ty * TILE_SIZE - topLeftWorldY;
        const baseTileUrl =
          mapStyle === 'dark'
            ? `https://a.basemaps.cartocdn.com/dark_all/${zoom}/${wrappedTx}/${ty}.png`
            : `https://tile.openstreetmap.org/${zoom}/${wrappedTx}/${ty}.png`;
        tiles.push({
          key: `${zoom}-${tx}-${ty}`,
          left,
          top,
          baseTileUrl,
          seamarkUrl: `https://tiles.openseamap.org/seamark/${zoom}/${wrappedTx}/${ty}.png`,
        });
      }
    }
  }

  const box = activeArea.era5_box || DEFAULT_AREA.era5_box;
  const era5PolygonCoords = [
    [box.north, box.west],
    [box.north, box.east],
    [box.south, box.east],
    [box.south, box.west],
  ];

  const era5Points = era5PolygonCoords
    .map(([lat, lon]) => {
      const p = project(lat, lon);
      return `${p.x},${p.y}`;
    })
    .join(' ');

  const zonePoints = (activeArea.zone_polygon || DEFAULT_AREA.zone_polygon)
    .map(([lat, lon]) => {
      const p = project(lat, lon);
      return `${p.x},${p.y}`;
    })
    .join(' ');

  const gridCenterLat = activeArea.grid_center ? activeArea.grid_center[0] : activeArea.lat;
  const gridCenterLon = activeArea.grid_center ? activeArea.grid_center[1] : activeArea.lon;
  const era5Center = project(gridCenterLat, gridCenterLon);
  const era5NW = project(box.north, box.west);
  const era5SE = project(box.south, box.east);

  const flowCompassBearing = (Number(windDirDeg) + 180) % 360;
  const flowSvgAngle = flowCompassBearing - 90;

  return (
    <div className="bg-slate-900/85 backdrop-blur p-6 rounded-2xl border border-slate-800 shadow-xl flex flex-col justify-between h-full">
      {/* Header & Map Layer Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <MapPin className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Covered Prediction Area: {activeArea.name}</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            ERA5 <span className="text-cyan-300 font-mono">0.25° × 0.25°</span> Grid (
            <span className="font-mono text-slate-200">
              {box.south.toFixed(3)}°N–{box.north.toFixed(3)}°N, {box.west.toFixed(3)}°E–{box.east.toFixed(3)}°E
            </span>{' '}
            • {box.area_km2} km²)
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setMapStyle(mapStyle === 'osm' ? 'dark' : 'osm')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-700 text-xs font-mono text-slate-200 transition"
            title="Switch Map Base Layer"
          >
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            {mapStyle === 'osm' ? 'OSM Map' : 'Dark Marine'}
          </button>
          <button
            onClick={() => setShowSeamarks(!showSeamarks)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-mono transition ${
              showSeamarks
                ? 'bg-emerald-950/80 border-emerald-600/60 text-emerald-300'
                : 'bg-slate-950 border-slate-800 text-slate-400'
            }`}
          >
            Seamarks
          </button>
          <div className="flex items-center bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
            <button
              onClick={() => setZoom((z) => Math.max(6, z - 1))}
              className="p-1.5 hover:bg-slate-800 text-slate-300"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-2 text-xs font-mono text-slate-400">z{zoom}</span>
            <button
              onClick={() => setZoom((z) => Math.min(11, z + 1))}
              className="p-1.5 hover:bg-slate-800 text-slate-300"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(activeArea.default_zoom || 9)}
              className="p-1.5 hover:bg-slate-800 text-slate-400 border-l border-slate-800"
              title="Reset View"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Interactive Map Viewport */}
      <div className="relative w-full h-[340px] rounded-xl overflow-hidden border border-slate-700/80 bg-[#0b1d33] select-none">
        <svg
          viewBox={`0 0 ${viewWidth} ${viewHeight}`}
          preserveAspectRatio="xMidYMid slice"
          className="w-full h-full"
        >
          <defs>
            <pattern id="era5GridPattern" width="16" height="16" patternUnits="userSpaceOnUse">
              <path
                d="M 16 0 L 0 0 0 16"
                fill="none"
                stroke="rgba(6, 182, 212, 0.22)"
                strokeWidth="1"
              />
            </pattern>
            <marker
              id="windArrowHead"
              markerWidth="6"
              markerHeight="6"
              refX="5"
              refY="3"
              orient="auto"
            >
              <path d="M0,0 L6,3 L0,6 Z" fill="#38bdf8" />
            </marker>
          </defs>

          {/* 1. OpenStreetMap / CartoDB Slippy Map Tiles */}
          <g>
            {tiles.map((t) => (
              <g key={t.key}>
                <image
                  href={t.baseTileUrl}
                  x={t.left}
                  y={t.top}
                  width={TILE_SIZE}
                  height={TILE_SIZE}
                  style={{
                    filter:
                      mapStyle === 'osm'
                        ? 'brightness(0.84) contrast(1.06) saturate(0.95)'
                        : 'none',
                  }}
                />
                {showSeamarks && (
                  <image
                    href={t.seamarkUrl}
                    x={t.left}
                    y={t.top}
                    width={TILE_SIZE}
                    height={TILE_SIZE}
                    opacity="0.85"
                  />
                )}
              </g>
            ))}
          </g>

          {/* 2. Highlighted ERA5 0.25° x 0.25° Prediction Grid Cell */}
          <g>
            <polygon
              points={era5Points}
              fill="url(#era5GridPattern)"
              stroke="#06b6d4"
              strokeWidth="2.5"
              strokeDasharray="7 4"
            />
            <polygon
              points={era5Points}
              fill="rgba(6, 182, 212, 0.12)"
              stroke="none"
            />

            {/* ERA5 Grid Corner Labels */}
            <rect
              x={era5NW.x + 4}
              y={era5NW.y + 4}
              width="202"
              height="20"
              rx="4"
              fill="rgba(15, 23, 42, 0.9)"
              stroke="rgba(6, 182, 212, 0.65)"
            />
            <text
              x={era5NW.x + 10}
              y={era5NW.y + 17}
              fill="#67e8f9"
              fontSize="10"
              fontFamily="monospace"
              fontWeight="bold"
            >
              ERA5 NW: {box.north.toFixed(3)}°N, {box.west.toFixed(3)}°E
            </text>

            <rect
              x={era5SE.x - 205}
              y={era5SE.y - 24}
              width="201"
              height="20"
              rx="4"
              fill="rgba(15, 23, 42, 0.9)"
              stroke="rgba(6, 182, 212, 0.65)"
            />
            <text
              x={era5SE.x - 199}
              y={era5SE.y - 11}
              fill="#67e8f9"
              fontSize="10"
              fontFamily="monospace"
              fontWeight="bold"
            >
              ERA5 SE: {box.south.toFixed(3)}°N, {box.east.toFixed(3)}°E
            </text>
          </g>

          {/* 3. Highlighted Wind Farm Zone Polygon */}
          <g>
            <polygon
              points={zonePoints}
              fill={zoneOp.canOperate ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'}
              stroke={zoneOp.canOperate ? '#10b981' : '#ef4444'}
              strokeWidth="2.5"
            />

            {/* ERA5 Grid Center Crosshair */}
            <circle
              cx={era5Center.x}
              cy={era5Center.y}
              r="7"
              fill={zoneOp.canOperate ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}
              stroke="#ffffff"
              strokeWidth="1.5"
            />
            <line
              x1={era5Center.x - 12}
              y1={era5Center.y}
              x2={era5Center.x + 12}
              y2={era5Center.y}
              stroke="#ffffff"
              strokeWidth="1.5"
            />
            <line
              x1={era5Center.x}
              y1={era5Center.y - 12}
              x2={era5Center.x}
              y2={era5Center.y + 12}
              stroke="#ffffff"
              strokeWidth="1.5"
            />

            {/* Covered Area Status Callout */}
            <rect
              x={era5Center.x - 132}
              y={era5Center.y - 44}
              width="264"
              height="26"
              rx="6"
              fill="rgba(15, 23, 42, 0.92)"
              stroke={zoneOp.canOperate ? '#10b981' : '#ef4444'}
              strokeWidth="1.5"
            />
            <text
              x={era5Center.x}
              y={era5Center.y - 27}
              textAnchor="middle"
              fill={zoneOp.canOperate ? '#6ee7b7' : '#fca5a5'}
              fontSize="10.5"
              fontFamily="monospace"
              fontWeight="bold"
            >
              +{horizonHours}h: {zoneOp.shortLabel} ({windSpeed} m/s)
            </text>
          </g>

          {/* 4. Coastal Reference Landmarks */}
          {(activeArea.landmarks || []).map((lm) => {
            const p = project(lm.lat, lm.lon);
            return (
              <g key={lm.name} transform={`translate(${p.x}, ${p.y})`}>
                <circle r="3.5" fill="#f59e0b" stroke="#0f172a" strokeWidth="1.5" />
                <text
                  x="7"
                  y="3"
                  fill="#fef3c7"
                  fontSize="10"
                  fontWeight="600"
                  style={{ textShadow: '0 1px 4px #000, 0 0 6px #000' }}
                >
                  {lm.name}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Pinned Top-Left 8-Point Reference Compass Rose Overlay (Never Cropped by SVG Aspect Slice) */}
        <div className="absolute top-2.5 left-2.5 z-10 pointer-events-none bg-slate-950/85 backdrop-blur-sm rounded-xl border border-slate-700/80 px-2.5 py-2 shadow-lg flex flex-col items-center">
          <svg width="96" height="96" viewBox="-50 -50 100 100" className="overflow-visible">
            <defs>
              <marker
                id="compassWindArrow"
                markerWidth="6"
                markerHeight="6"
                refX="5"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L6,3 L0,6 Z" fill="#38bdf8" />
              </marker>
            </defs>
            <circle
              r="44"
              fill="rgba(15, 23, 42, 0.92)"
              stroke="rgba(56, 189, 248, 0.55)"
              strokeWidth="1.5"
            />
            <circle
              r="30"
              fill="none"
              stroke="rgba(148, 163, 184, 0.28)"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
            {[
              { lbl: 'N', angle: 0, main: true },
              { lbl: 'NE', angle: 45, main: false },
              { lbl: 'E', angle: 90, main: true },
              { lbl: 'SE', angle: 135, main: false },
              { lbl: 'S', angle: 180, main: true },
              { lbl: 'SW', angle: 225, main: false },
              { lbl: 'W', angle: 270, main: true },
              { lbl: 'NW', angle: 315, main: false },
            ].map((pt) => {
              const rad = ((pt.angle - 90) * Math.PI) / 180;
              const rText = 36;
              const tx = Math.cos(rad) * rText;
              const ty = Math.sin(rad) * rText + 3;
              return (
                <text
                  key={pt.lbl}
                  x={tx}
                  y={ty}
                  textAnchor="middle"
                  fill={pt.lbl === 'N' ? '#38bdf8' : pt.main ? '#f8fafc' : '#94a3b8'}
                  fontSize={pt.main ? '9.5' : '7.5'}
                  fontFamily="monospace"
                  fontWeight={pt.main ? 'bold' : 'normal'}
                >
                  {pt.lbl}
                </text>
              );
            })}

            <g transform={`rotate(${flowSvgAngle})`}>
              <line
                x1="-20"
                y1="0"
                x2="18"
                y2="0"
                stroke="#38bdf8"
                strokeWidth="2.5"
                markerEnd="url(#compassWindArrow)"
              />
              <circle cx="-20" cy="0" r="3" fill="#f59e0b" />
            </g>
          </svg>
          <div className="mt-1 text-[10px] font-mono font-bold text-cyan-300 leading-none">
            {compass.short16} ({compass.deg}°)
          </div>
        </div>

        {/* OpenStreetMap Attribution */}
        <div className="absolute bottom-1.5 right-2 bg-slate-950/80 px-2 py-0.5 rounded text-[9px] text-slate-400">
          © OpenStreetMap contributors • OpenSeaMap
        </div>
      </div>

      {/* Spacious Bottom Legend & Compass Bar (No Map Overlap) */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <div className="bg-slate-950/70 rounded-xl p-3 border border-slate-800/80 space-y-1.5">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Highlighted Spatial Coverage
          </div>
          <div className="flex items-center gap-2 text-cyan-300 font-mono text-[11px]">
            <span className="w-3.5 h-2.5 border-2 border-dashed border-cyan-400 bg-cyan-500/20 inline-block rounded-sm shrink-0" />
            <span>ERA5 0.25° Grid Cell ({box.area_km2} km²)</span>
          </div>
          <div className="flex items-center gap-2 text-emerald-300 font-mono text-[11px]">
            <span
              className={`w-3.5 h-2.5 border-2 inline-block rounded-sm shrink-0 ${
                zoneOp.canOperate
                  ? 'border-emerald-400 bg-emerald-500/30'
                  : 'border-red-400 bg-red-500/30'
              }`}
            />
            <span>Wind Farm Lease Zone ({activeArea.zone_area_km2} km²)</span>
          </div>
        </div>

        <div className="bg-slate-950/70 rounded-xl p-3 border border-slate-800/80 space-y-1">
          <div className="flex items-center justify-between text-cyan-300 font-mono font-bold text-[11px]">
            <span className="flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              {compass.short16} • {compass.full8} ({compass.deg}°)
            </span>
            <span className="text-emerald-400">{windSpeed} m/s</span>
          </div>
          <div className="text-[11px] text-amber-300 font-mono">
            {compass.flowLabel}
          </div>
          <div className="text-[11px] text-slate-400 font-mono">
            Grid Center: {gridCenterLat.toFixed(3)}°N, {gridCenterLon.toFixed(3)}°E
          </div>
        </div>
      </div>
    </div>
  );
}
