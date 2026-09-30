/**
 * Human-readable labels for raw backend/API values. Kept in one place
 * so the UI never shows internal identifiers (calibration_stage enum
 * values, model_type/device strings) directly to non-technical users.
 */

const CALIBRATION_LABELS = {
  relative_uncalibrated: {
    label: "Not calibrated",
    detail: "Heights are relative only — good for shape, not real-world size.",
  },
  dem_calibrated: {
    label: "Calibrated to real meters",
    detail: "Matched against satellite elevation data (Copernicus DEM).",
  },
  lora_calibrated: {
    label: "Calibrated to real meters",
    detail: "Estimated directly by a fine-tuned AI model.",
  },
};

export function formatCalibrationStage(stage) {
  return (
    CALIBRATION_LABELS[stage] || {
      label: stage || "Unknown",
      detail: "",
    }
  );
}

const MODEL_LABELS = {
  "depth-anything-vits": "Depth Anything (Fast)",
  "depth-anything-vitl": "Depth Anything (High accuracy)",
};

export function formatModel(modelType, device) {
  const name = MODEL_LABELS[modelType] || modelType || "Unknown model";
  const hardware = (device || "").toLowerCase().includes("cuda") ? "GPU" : "CPU";
  return `${name} · ${hardware}`;
}

export function formatGeoreferenced(isGeoreferenced) {
  return isGeoreferenced ? "Yes — has location data" : "No location data";
}
