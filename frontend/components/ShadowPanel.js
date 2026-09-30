"use client";

import { useState } from "react";
import { ApiError, estimateShadowHeights, resolveAssetUrl } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const MAX_SHOWN = 8; // keep the marker overlay and the list in sync

/**
 * Stage 2c UI: lets the user run the shadow-length + sun-angle
 * cross-check (backend/app/routers/shadow.py) against the same file
 * they already uploaded, and shows the resulting height estimates next
 * to the AI model's own result so they can be compared — per the spec's
 * "display both values in the UI when available".
 *
 * Sun elevation/azimuth are always manual input here (see
 * shadow_geometry.py's docstring for why the automatic pysolar path
 * isn't wired up yet) — the disclaimer below matches the spec's own
 * documented fallback for this exact situation.
 */
export default function ShadowPanel({ file, result }) {
  const [sunElevationDeg, setSunElevationDeg] = useState(45);
  const [sunAzimuthDeg, setSunAzimuthDeg] = useState(180);
  const [gsdMetersPerPx, setGsdMetersPerPx] = useState(0.3);
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const { token } = useAuth();

  if (!file || !result) return null;

  const needsGsd = !result.is_georeferenced;

  const run = async () => {
    setStatus("loading");
    setError(null);
    try {
      const body = await estimateShadowHeights(file, {
        token,
        sunElevationDeg: Number(sunElevationDeg),
        sunAzimuthDeg: Number(sunAzimuthDeg),
        gsdMetersPerPx: needsGsd ? Number(gsdMetersPerPx) : undefined,
      });
      setResponse(body);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setError(err instanceof ApiError ? err.message : "Shadow estimation failed.");
    }
  };

  return (
    <div className="w-full rounded-2xl border border-base-800 bg-base-900 p-5 text-sm text-slate-300">
      <h2 className="mb-1 font-semibold text-white">Double-check with shadows</h2>
      <p className="mb-3 text-xs text-slate-500">
        Estimates building heights from how long their shadows are — a
        second opinion, independent of the AI model above. Accuracy
        depends on how close your sun position estimate is to reality.
      </p>

      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Sun height (° above horizon)
          <input
            type="number"
            min={1}
            max={89}
            value={sunElevationDeg}
            onChange={(e) => setSunElevationDeg(e.target.value)}
            className="rounded bg-base-800 px-2 py-1 text-slate-100"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Sun direction (° from North)
          <input
            type="number"
            min={0}
            max={359}
            value={sunAzimuthDeg}
            onChange={(e) => setSunAzimuthDeg(e.target.value)}
            className="rounded bg-base-800 px-2 py-1 text-slate-100"
          />
        </label>
        {needsGsd && (
          <label className="col-span-2 flex flex-col gap-1 text-xs text-slate-400">
            Meters per pixel (estimate — this photo has no location data)
            <input
              type="number"
              min={0.01}
              step={0.01}
              value={gsdMetersPerPx}
              onChange={(e) => setGsdMetersPerPx(e.target.value)}
              className="rounded bg-base-800 px-2 py-1 text-slate-100"
            />
          </label>
        )}
      </div>

      <button
        onClick={run}
        disabled={status === "loading"}
        className="mt-3 w-full rounded-lg bg-accent-500 px-3 py-2 text-xs font-medium text-base-950 hover:bg-accent-400 disabled:opacity-50"
      >
        {status === "loading" ? "Measuring shadows…" : "Check shadow heights"}
      </button>

      {status === "error" && (
        <p className="mt-3 rounded-lg bg-red-950/50 px-3 py-2 text-xs text-red-300">{error}</p>
      )}

      {response && (
        <div className="mt-3">
          <p className="text-xs text-slate-500">
            Found {response.candidate_count} shadow{response.candidate_count === 1 ? "" : "s"} ·{" "}
            {response.gsd_source === "geotiff_transform" ? "scale from location data" : "scale estimated"}
          </p>

          {/* Numbered markers on the photo, so each list entry can be
              matched to the actual structure it's measuring — x/y are
              in the ORIGINAL upload's pixel space (image_width/height),
              converted to percentages so they line up regardless of how
              large the <img> is actually rendered. */}
          <div className="relative mt-2 overflow-hidden rounded-lg border border-base-800">
            <img
              src={resolveAssetUrl(result.image_url)}
              alt="Uploaded photo with detected shadow markers"
              className="block w-full"
            />
            {response.estimates.slice(0, MAX_SHOWN).map((estimate, i) => (
              <div
                key={i}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                style={{
                  left: `${(estimate.x / response.image_width) * 100}%`,
                  top: `${(estimate.y / response.image_height) * 100}%`,
                }}
                className={`absolute flex h-6 w-6 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 text-[11px] font-bold shadow-lg transition-transform ${
                  hoveredIndex === i
                    ? "z-10 scale-125 border-white bg-accent-500 text-base-950"
                    : "border-base-950 bg-accent-500/90 text-base-950"
                }`}
                title={`Structure ${i + 1}: ${estimate.estimated_height_m.toFixed(1)} m`}
              >
                {i + 1}
              </div>
            ))}
          </div>

          <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto text-xs">
            {response.estimates.slice(0, MAX_SHOWN).map((estimate, i) => (
              <li
                key={i}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                className={`flex justify-between rounded-lg px-3 py-1.5 transition-colors ${
                  hoveredIndex === i ? "bg-accent-500/20" : "bg-base-800"
                }`}
              >
                <span className="flex items-center gap-2 text-slate-500">
                  <span className="flex h-4 w-4 items-center justify-center rounded-full bg-accent-500 text-[10px] font-bold text-base-950">
                    {i + 1}
                  </span>
                  Structure {i + 1}
                </span>
                <span className="font-medium text-slate-100">
                  {estimate.estimated_height_m.toFixed(1)} m
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
