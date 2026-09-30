"use client";

import { resolveAssetUrl } from "@/lib/api";
import { formatCalibrationStage, formatGeoreferenced, formatModel } from "@/lib/format";

/**
 * Plain-language result card: what the file was, whether it's
 * calibrated to real measurements, and download links for the outputs.
 * Deliberately avoids surfacing raw backend enum values / identifiers
 * (calibration_stage strings, model_type+device) — see lib/format.js.
 */
export default function ResultSummary({ result }) {
  if (!result) return null;

  const calibration = formatCalibrationStage(result.calibration_stage);

  return (
    <div className="w-full rounded-2xl border border-base-800 bg-base-900 p-5 text-sm text-slate-300">
      <h2 className="mb-3 font-semibold text-white">Result</h2>

      <div className="space-y-2.5">
        <Row label="File" value={result.filename} />
        <Row label="Size" value={`${result.width} × ${result.height} px`} />
        <Row label="Location data" value={formatGeoreferenced(result.is_georeferenced)} />
        <Row
          label="Calibrated to real height"
          value={
            <span className={isCalibrated(result) ? "text-emerald-400" : "text-slate-400"}>
              {isCalibrated(result) ? "Yes" : "No"}
            </span>
          }
        />
        <Row label="AI model" value={formatModel(result.model_type, result.device)} />
      </div>

      {calibration.detail && (
        <p className="mt-3 rounded-lg bg-base-800/60 px-3 py-2 text-xs text-slate-400">
          {calibration.detail}
        </p>
      )}

      {result.dem_calibration_report && (
        <p className="mt-2 text-xs text-slate-500">
          Fit quality: R² {result.dem_calibration_report.r_squared.toFixed(2)} across{" "}
          {result.dem_calibration_report.sample_count} sample points.
        </p>
      )}

      {result.dem_calibration_error && (
        <p className="mt-3 rounded-lg bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
          Couldn't calibrate to real meters for this image: {result.dem_calibration_error}
        </p>
      )}

      <div className="mt-4 border-t border-base-800 pt-4">
        <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
          Downloads
        </h3>
        <div className="flex flex-wrap gap-2">
          <DownloadLink
            href={result.depth_heatmap_url}
            label="Depth map (PNG)"
            filename={`${result.job_id}_depth.png`}
          />
          {result.dsm_geotiff_url && (
            <DownloadLink
              href={result.dsm_geotiff_url}
              label="Elevation data (GeoTIFF)"
              filename={`${result.job_id}_dsm.tif`}
            />
          )}
          {result.relative_dsm_tif_url && (
            <DownloadLink
              href={result.relative_dsm_tif_url}
              label="Elevation data (.tif)"
              filename={`${result.job_id}_rdsm.tif`}
            />
          )}
        </div>
        <p className="mt-2 text-xs text-slate-500">
          3D model downloads (.glb / .obj) are available on the terrain view →
        </p>
      </div>
    </div>
  );
}

function isCalibrated(result) {
  return (
    (result.calibration_stage === "dem_calibrated" ||
      result.calibration_stage === "lora_calibrated") &&
    !!result.absolute_height_grid
  );
}

function DownloadLink({ href, label, filename }) {
  if (!href) return null;
  return (
    <a
      href={resolveAssetUrl(href)}
      download={filename}
      className="rounded-lg bg-base-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-base-700"
    >
      ⬇ {label}
    </a>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-medium text-slate-100">{value}</dd>
    </div>
  );
}
