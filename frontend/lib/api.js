/**
 * Thin client for the FastAPI backend (backend/app/routers/*.py).
 * Keeping this isolated so the rest of the frontend never touches
 * fetch()/URLs directly — makes it a one-place change when Stage 2
 * calibration adds new response fields or endpoints.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseErrorDetail(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

async function postForm(path, formData, { token, signal } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      body: formData,
      signal,
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
  } catch (err) {
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. Is it running? (${err.message})`,
      0
    );
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

async function postJson(path, payload) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. Is it running? (${err.message})`,
      0
    );
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

/** POST /api/auth/signup — returns { access_token, email }. */
export function signup(email, password) {
  return postJson("/api/auth/signup", { email, password });
}

/** POST /api/auth/login — returns { access_token, email }. */
export function login(email, password) {
  return postJson("/api/auth/login", { email, password });
}

/**
 * Uploads an image to /api/depth/estimate and returns the parsed
 * DepthEstimateResponse (see backend/app/schemas.py for the shape:
 * job_id, is_georeferenced, geo, calibration_stage, image_url,
 * depth_heatmap_url, confidence_heatmap_url, height_grid,
 * height_grid_resolution, ...). Requires a signed-in session token.
 *
 * includeConfidence: Stage 2e (bonus, opt-in) — also runs MiDaS
 * backend-side for a MiDaS-vs-Depth-Anything disagreement heatmap.
 * Roughly doubles inference time, so it's off unless the caller asks.
 */
export function estimateDepth(file, { token, signal, includeConfidence = false } = {}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("include_confidence", includeConfidence ? "true" : "false");
  return postForm("/api/depth/estimate", formData, { token, signal });
}

/**
 * Uploads the SAME image to /api/shadow/estimate — Stage 2c, an
 * INDEPENDENT cross-check against the AI depth model (see
 * backend/app/services/shadow/shadow_geometry.py). Requires the sun's
 * elevation/azimuth as explicit input (see ShadowPanel.js for the
 * disclaimer on estimated vs. known angles) and, for non-georeferenced
 * images, a ground-sample-distance guess in meters/pixel. Requires a
 * signed-in session token.
 */
export function estimateShadowHeights(
  file,
  { token, sunElevationDeg, sunAzimuthDeg, gsdMetersPerPx, signal }
) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("sun_elevation_deg", String(sunElevationDeg));
  formData.append("sun_azimuth_deg", String(sunAzimuthDeg));
  if (gsdMetersPerPx != null) {
    formData.append("gsd_meters_per_px", String(gsdMetersPerPx));
  }
  return postForm("/api/shadow/estimate", formData, { token, signal });
}

/** Resolves a backend-relative URL (e.g. "/outputs/x.png") to an absolute one. */
export function resolveAssetUrl(path) {
  if (!path) return path;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
}

export { API_BASE_URL };
