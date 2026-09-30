"use client";

import { useCallback, useRef, useState } from "react";
import { ApiError, estimateDepth } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const ACCEPTED = ".png,.jpg,.jpeg,.tif,.tiff";

/**
 * Handles file selection (click or drag/drop), calls the backend, and
 * reports progress/results up to the parent page via callback props so
 * the 3D viewer and upload UI stay decoupled.
 */
export default function UploadPanel({ onResult }) {
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [fileName, setFileName] = useState(null);
  const [includeConfidence, setIncludeConfidence] = useState(false);
  const inputRef = useRef(null);
  const { token } = useAuth();

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;
      setFileName(file.name);
      setStatus("loading");
      setError(null);
      try {
        const result = await estimateDepth(file, { token, includeConfidence });
        setStatus("idle");
        onResult(result, file);
      } catch (err) {
        setStatus("error");
        setError(
          err instanceof ApiError
            ? err.message
            : "Something went wrong while processing that image."
        );
      }
    },
    [onResult, includeConfidence, token]
  );

  const onInputChange = (e) => handleFile(e.target.files?.[0]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  return (
    <div className="w-full max-w-xl">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-colors
          ${dragActive ? "border-accent-400 bg-base-800" : "border-base-700 bg-base-900"}
          hover:border-accent-500`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="hidden"
          onChange={onInputChange}
        />
        {status === "loading" ? (
          <div className="flex flex-col items-center gap-3">
            <span className="h-6 w-6 animate-spin rounded-full border-2 border-accent-400 border-t-transparent" />
            <p className="text-accent-400">
              Analyzing <span className="font-medium">{fileName}</span>…
            </p>
            <p className="text-xs text-slate-500">
              First run may take a minute while the AI model loads.
            </p>
          </div>
        ) : (
          <>
            <div className="mb-3 text-4xl">📤</div>
            <p className="text-lg font-medium text-white">
              Drop a satellite or aerial photo here
            </p>
            <p className="mt-1 text-sm text-slate-400">
              or click to browse · PNG, JPG, or GeoTIFF
            </p>
            <p className="mt-3 text-xs text-slate-500">
              Photos with location data (GeoTIFF) are automatically calibrated
              to real-world height in meters. Regular photos get an accurate
              3D shape without exact measurements.
            </p>
          </>
        )}
      </div>

      <label
        className="mt-3 flex items-center gap-2 text-sm text-slate-400"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          type="checkbox"
          checked={includeConfidence}
          onChange={(e) => setIncludeConfidence(e.target.checked)}
          className="accent-accent-500"
        />
        Show a confidence overlay (takes a bit longer)
      </label>

      {status === "error" && (
        <p className="mt-3 rounded-lg bg-red-950/50 px-4 py-3 text-sm text-red-300">
          {error}
        </p>
      )}
    </div>
  );
}
