"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { resolveAssetUrl } from "@/lib/api";
import { formatCalibrationStage } from "@/lib/format";

// Stage 1's height_grid is normalized [0, 1] relative depth, not
// meters — this constant is a purely visual multiplier so an
// uncalibrated mesh doesn't look flat. Only used as a fallback when
// Stage 2a's absolute_height_grid (real meters) isn't available — see
// the isCalibrated branch below.
const UNCALIBRATED_HEIGHT_SCALE = 40;
const PLANE_SIZE = 100;

// Disocclusion-gap mitigation (flagged by the 3D Ken Burns paper — see
// README > "Known gap flagged for Stage 3"): the mesh only has color
// data from the single top-down source photo, texture-mapped onto a
// height-displaced plane. There is no data at all for the *sides* of
// tall buildings/terrain, or for whatever those buildings occlude. A
// near-horizontal camera angle exposes those untextured faces as holes
// or stretched-looking artifacts. The cheap fix (this is it — the
// "constrain the camera" option from the README, applied before any
// fuller per-building smoothing or inpainting): keep the camera in an
// oblique "aerial tilt" range close to the original photo's viewing
// angle, and never let it pan away from the mesh, so it can't reach the
// grazing angles where the gaps would show.
const MAX_TILT_DEG = 58; // polar angle from straight-down; 90 = horizontal (never allowed)
const MIN_TILT_DEG = 8; // avoid exact top-lock gimbal jitter, stay just off nadir

// Flythrough mode deliberately does NOT apply the tilt clamp above — the
// spec asks for free flight "from any angle," which is the opposite of
// what the clamp is for. Trade-off, stated honestly: flying close to
// steep terrain at a grazing angle can expose the same untextured
// mesh-side artifacts the clamp exists to hide in orbit mode (see the
// disocclusion-gap comment above) — acceptable here since the whole
// point of this mode is unrestricted movement.
const FLY_SPEED = 45; // world units/second — crosses the ~100-unit PLANE_SIZE in ~2s
const FLY_LOOK_SENSITIVITY = 0.0025; // radians per pixel of mouse drag
const FLY_ALTITUDE_WHEEL_SPEED = 0.06; // world units per wheel-delta unit

const TEXTURE_MODES = [
  { key: "original", label: "Photo" },
  { key: "depth", label: "Depth map" },
  { key: "confidence", label: "Confidence" },
  { key: "slope", label: "Slope" },
];

// Stage 2d originally planned to get surface normals "for free" as a
// second output of Metric3D v2's forward pass (see README > Roadmap —
// Metric3D v2 isn't integrated: its official loader depends on the
// mmcv/openmmlab toolchain, which doesn't support Python 3.12, a real
// blocker hit while trying to wire it in, not a guess). This is the
// pragmatic substitute for Stage 2d/6's slope requirement: the terrain
// mesh already has real displaced geometry, and geometry.computeVertexNormals()
// — already running below for lighting — gives genuine per-vertex
// surface normals at zero extra cost, same rationale the original plan
// used for Metric3D v2, just sourced from OUR OWN height map instead of
// an independent model prediction. Honest limitation: because the
// normals are derived from the height map (not predicted independently
// from the image), they inherit whatever noise is already in Stage
// 1/2a's height values, rather than cross-checking it.
const SLOPE_FLAT_HUE = 0.33; // green
const SLOPE_STEEP_HUE = 0.0; // red
const SLOPE_CLAMP_DEG = 60; // slope at/above this angle renders fully "steep"

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

const VIEW_MODES = [
  { key: "orbit", label: "Orbit View" },
  { key: "flythrough", label: "Flythrough Mode" },
];

/**
 * First-person fly camera: WASD/arrow keys move (forward/back along the
 * full view direction, including pitch, so looking down + "forward"
 * dives toward the terrain — a deliberate choice over FPS-style
 * horizontal-only movement, since "fly over and through the mesh from
 * any angle" implies full 3D movement, not a ground-walker). Mouse-drag
 * (not pointer-lock) rotates look direction, matching the drag-to-look
 * interaction OrbitControls already uses elsewhere in this viewer.
 * Space / Shift / scroll-wheel move straight up/down along world Y,
 * independent of look direction, so altitude changes stay predictable
 * regardless of where the camera is pointed.
 *
 * Built custom rather than three.js's FlyControls: FlyControls' default
 * bindings (WASD + R/F for altitude + Q/E for roll) don't match the
 * spec's requested Space/Shift-altitude scheme, and its aircraft-style
 * roll would be disorienting for terrain inspection — this controller
 * only ever yaws/pitches (Euler order 'YXZ', the standard no-roll FPS
 * camera setup), never rolls.
 */
function createFlyControls(camera, domElement) {
  const moveState = { forward: false, back: false, left: false, right: false, up: false, down: false };
  let isDragging = false;
  let lastX = 0;
  let lastY = 0;

  // Continue smoothly from wherever the camera was already looking
  // (set via camera.lookAt just before this is constructed) instead of
  // snapping to a fixed orientation.
  const startEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
  let yaw = startEuler.y;
  let pitch = startEuler.x;

  function isTypingInField() {
    const tag = document.activeElement?.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  function onKeyDown(e) {
    if (isTypingInField()) return;
    switch (e.code) {
      case "KeyW":
      case "ArrowUp":
        moveState.forward = true;
        break;
      case "KeyS":
      case "ArrowDown":
        moveState.back = true;
        break;
      case "KeyA":
      case "ArrowLeft":
        moveState.left = true;
        break;
      case "KeyD":
      case "ArrowRight":
        moveState.right = true;
        break;
      case "Space":
        moveState.up = true;
        e.preventDefault(); // stop the page from scrolling
        break;
      case "ShiftLeft":
      case "ShiftRight":
        moveState.down = true;
        break;
      default:
        return;
    }
  }

  function onKeyUp(e) {
    switch (e.code) {
      case "KeyW":
      case "ArrowUp":
        moveState.forward = false;
        break;
      case "KeyS":
      case "ArrowDown":
        moveState.back = false;
        break;
      case "KeyA":
      case "ArrowLeft":
        moveState.left = false;
        break;
      case "KeyD":
      case "ArrowRight":
        moveState.right = false;
        break;
      case "Space":
        moveState.up = false;
        break;
      case "ShiftLeft":
      case "ShiftRight":
        moveState.down = false;
        break;
      default:
        return;
    }
  }

  function onMouseDown(e) {
    isDragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  }
  function onMouseUp() {
    isDragging = false;
  }
  function onMouseMove(e) {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    yaw -= dx * FLY_LOOK_SENSITIVITY;
    pitch -= dy * FLY_LOOK_SENSITIVITY;
    const maxPitch = Math.PI / 2 - 0.01;
    pitch = THREE.MathUtils.clamp(pitch, -maxPitch, maxPitch);
  }
  function onWheel(e) {
    e.preventDefault();
    camera.position.y -= e.deltaY * FLY_ALTITUDE_WHEEL_SPEED;
  }

  domElement.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  domElement.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);

  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();

  function update(deltaSeconds) {
    camera.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, "YXZ"));

    const speed = FLY_SPEED * deltaSeconds;
    forward.set(0, 0, -1).applyQuaternion(camera.quaternion);
    right.set(1, 0, 0).applyQuaternion(camera.quaternion);

    if (moveState.forward) camera.position.addScaledVector(forward, speed);
    if (moveState.back) camera.position.addScaledVector(forward, -speed);
    if (moveState.right) camera.position.addScaledVector(right, speed);
    if (moveState.left) camera.position.addScaledVector(right, -speed);
    if (moveState.up) camera.position.y += speed;
    if (moveState.down) camera.position.y -= speed;
  }

  function dispose() {
    domElement.removeEventListener("mousedown", onMouseDown);
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
    domElement.removeEventListener("wheel", onWheel);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
  }

  return { update, dispose };
}

/**
 * Builds a displacement-mapped terrain mesh from a DepthEstimateResponse
 * (see backend/app/schemas.py) and renders it with a choice of two
 * navigation modes:
 *   - "orbit" (default): OrbitControls, tilt-clamped per the
 *     disocclusion-gap comment above.
 *   - "flythrough": a custom WASD/arrow-key + mouse-drag-look + Space/
 *     Shift/scroll-altitude fly camera (see the FLY_* constants above),
 *     unclamped — can approach the mesh from any angle.
 * The photo texture (result.image_url) is draped onto the mesh as the
 * default view in both modes; depth/confidence/slope are togglable
 * overlays (see TEXTURE_MODES).
 */
export default function TerrainViewer({ result }) {
  const containerRef = useRef(null);
  const meshRef = useRef(null);
  const [hoverInfo, setHoverInfo] = useState(null);
  const [textureMode, setTextureMode] = useState("original");
  const [viewMode, setViewMode] = useState("orbit");
  const [showHint, setShowHint] = useState(true);

  useEffect(() => {
    setShowHint(true);
    const timer = setTimeout(() => setShowHint(false), 4000);
    return () => clearTimeout(timer);
  }, [result, viewMode]);

  useEffect(() => {
    if (!result || !containerRef.current) return;

    const textureUrlByMode = {
      original: result.image_url,
      depth: result.depth_heatmap_url,
      confidence: result.confidence_heatmap_url,
    };
    const activeTextureUrl = textureUrlByMode[textureMode] || result.image_url;

    const container = containerRef.current;
    const { height_grid_resolution: resolution } = result;

    // Stage 2a (DEM) or Stage 2b (LoRA, once a trained adapter exists —
    // see backend/training/README.md) calibration: absolute_height_grid
    // holds real meters — use those directly instead of the normalized
    // [0, 1] relative grid. Vertical scale is then 1 world-unit = 1
    // meter (elevationOffset shifts the lowest point to y=0 so the mesh
    // doesn't sit below the grid); the horizontal footprint (PLANE_SIZE)
    // is NOT to the same real-world scale — only relative heights across
    // the mesh are geographically meaningful, not absolute ground extent.
    const isCalibrated =
      (result.calibration_stage === "dem_calibrated" ||
        result.calibration_stage === "lora_calibrated") &&
      result.absolute_height_grid;
    const heightGrid = isCalibrated ? result.absolute_height_grid : result.height_grid;
    const elevationOffset = isCalibrated ? -Math.min(...heightGrid.flat()) : 0;
    const heightScale = isCalibrated ? 1 : UNCALIBRATED_HEIGHT_SCALE;

    let disposed = false;
    let animationId;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070a);
    scene.fog = new THREE.Fog(0x05070a, 150, 400);

    const camera = new THREE.PerspectiveCamera(
      55,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 70, 90);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(50, 80, 30);
    scene.add(sun);

    // --- Terrain mesh from the height grid ---
    const segments = resolution - 1;
    const geometry = new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE, segments, segments);
    geometry.rotateX(-Math.PI / 2); // horizontal, Y-up

    const positions = geometry.attributes.position;
    for (let row = 0; row < resolution; row++) {
      for (let col = 0; col < resolution; col++) {
        const vertexIndex = row * resolution + col;
        const h = heightGrid[row][col];
        positions.setY(vertexIndex, h * heightScale + elevationOffset);
      }
    }
    positions.needsUpdate = true;
    geometry.computeVertexNormals();

    // --- Slope analysis from the real mesh normals (see SLOPE_* comment above) ---
    const normalAttr = geometry.attributes.normal;
    const vertexCount = resolution * resolution;
    const slopeDegAtVertex = new Float32Array(vertexCount);
    const slopeColors = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i++) {
      // Y-up mesh: a flat surface has normal (0, 1, 0) -> slope 0.
      const ny = THREE.MathUtils.clamp(normalAttr.getY(i), -1, 1);
      const slopeDeg = THREE.MathUtils.radToDeg(Math.acos(ny));
      slopeDegAtVertex[i] = slopeDeg;
      const t = Math.min(slopeDeg / SLOPE_CLAMP_DEG, 1);
      const hue = SLOPE_FLAT_HUE + t * (SLOPE_STEEP_HUE - SLOPE_FLAT_HUE);
      const color = new THREE.Color().setHSL(hue, 0.85, 0.5);
      slopeColors[i * 3] = color.r;
      slopeColors[i * 3 + 1] = color.g;
      slopeColors[i * 3 + 2] = color.b;
    }
    const whiteColors = new Float32Array(vertexCount * 3).fill(1);
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(textureMode === "slope" ? slopeColors : whiteColors, 3)
    );

    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      side: THREE.DoubleSide,
      vertexColors: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "terrain";
    scene.add(mesh);
    meshRef.current = mesh;

    if (textureMode !== "slope") {
      const textureLoader = new THREE.TextureLoader();
      // Explicit (not relying on the library default): the image is
      // fetched cross-origin (frontend on :3000, backend on :8000).
      // Without this, the loaded texture can taint the canvas, which
      // silently breaks GLTFExporter's GLB image embedding (fix 3's
      // export button) even though on-screen rendering looks fine.
      // Backend CORS (app/main.py) already allows this origin.
      textureLoader.crossOrigin = "anonymous";
      textureLoader.load(resolveAssetUrl(activeTextureUrl), (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        material.map = tex;
        material.needsUpdate = true;
      });
    }

    // --- Navigation: orbit (default, clamped) or flythrough (free) ---
    const orbitControls =
      viewMode === "orbit" ? new OrbitControls(camera, renderer.domElement) : null;
    if (orbitControls) {
      orbitControls.enableDamping = true;
      orbitControls.dampingFactor = 0.08;
      orbitControls.enablePan = false; // never let the target drift off the mesh — see MAX_TILT_DEG comment above
      orbitControls.minPolarAngle = THREE.MathUtils.degToRad(MIN_TILT_DEG);
      orbitControls.maxPolarAngle = THREE.MathUtils.degToRad(MAX_TILT_DEG);
      orbitControls.minDistance = 25;
      orbitControls.maxDistance = 180;
    }
    const flyControls =
      viewMode === "flythrough" ? createFlyControls(camera, renderer.domElement) : null;

    // --- Height-on-hover ---
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    function onPointerMove(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(mesh);
      if (hits.length > 0) {
        const point = hits[0].point;
        const face = hits[0].face;
        const slopeDeg = face
          ? (slopeDegAtVertex[face.a] + slopeDegAtVertex[face.b] + slopeDegAtVertex[face.c]) / 3
          : null;
        setHoverInfo({
          x: point.x.toFixed(1),
          z: point.z.toFixed(1),
          isCalibrated,
          calibrationStage: result.calibration_stage,
          height: isCalibrated
            ? (point.y - elevationOffset).toFixed(1)
            : (point.y / heightScale).toFixed(3),
          slopeDeg: slopeDeg != null ? slopeDeg.toFixed(1) : null,
        });
      } else {
        setHoverInfo(null);
      }
    }
    renderer.domElement.addEventListener("pointermove", onPointerMove);

    function onResize() {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }
    window.addEventListener("resize", onResize);

    const clock = new THREE.Clock();

    function animate() {
      if (disposed) return;
      const delta = clock.getDelta();
      if (orbitControls) orbitControls.update();
      if (flyControls) flyControls.update(delta);
      renderer.render(scene, camera);
      animationId = requestAnimationFrame(animate);
    }
    animate();

    return () => {
      disposed = true;
      meshRef.current = null;
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      if (orbitControls) orbitControls.dispose();
      if (flyControls) flyControls.dispose();
      geometry.dispose();
      material.dispose();
      if (material.map) material.map.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [result, textureMode, viewMode]);

  const exportGLB = async () => {
    if (!meshRef.current) return;
    const { GLTFExporter } = await import("three/examples/jsm/exporters/GLTFExporter.js");
    const exporter = new GLTFExporter();
    exporter.parse(
      meshRef.current,
      (glb) => downloadBlob(new Blob([glb], { type: "model/gltf-binary" }), "terrain.glb"),
      (error) => console.error("GLB export failed", error),
      { binary: true }
    );
  };

  const exportOBJ = async () => {
    if (!meshRef.current) return;
    const { OBJExporter } = await import("three/examples/jsm/exporters/OBJExporter.js");
    const exporter = new OBJExporter();
    const objText = exporter.parse(meshRef.current);
    downloadBlob(new Blob([objText], { type: "text/plain" }), "terrain.obj");
  };

  if (!result) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-slate-500">
        <span className="text-3xl">🏔️</span>
        <p>Your 3D terrain will appear here after you upload an image.</p>
      </div>
    );
  }

  const calibration = formatCalibrationStage(result.calibration_stage);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />

      {/* Top-left: file + calibration status */}
      <div className="absolute left-4 top-4 max-w-xs rounded-xl bg-base-900/85 px-4 py-3 text-xs text-slate-300 shadow-lg backdrop-blur">
        <div className="truncate font-medium text-slate-100">{result.filename}</div>
        <div className="mt-1 flex items-center gap-1.5">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              isCalibratedStage(result.calibration_stage) ? "bg-emerald-400" : "bg-slate-500"
            }`}
          />
          <span>{calibration.label}</span>
        </div>
        {calibration.detail && <div className="mt-0.5 text-slate-500">{calibration.detail}</div>}
      </div>

      {/* Top-right: navigation mode + texture mode + export */}
      <div className="absolute right-4 top-4 flex flex-col items-end gap-2">
        <div className="flex gap-1 rounded-xl bg-base-900/85 p-1 shadow-lg backdrop-blur">
          {VIEW_MODES.map((mode) => (
            <button
              key={mode.key}
              onClick={() => setViewMode(mode.key)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                viewMode === mode.key
                  ? "bg-accent-500 text-base-950"
                  : "text-slate-300 hover:bg-base-700"
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>

        <div className="flex gap-1 rounded-xl bg-base-900/85 p-1 shadow-lg backdrop-blur">
          {TEXTURE_MODES.filter(
            (mode) => mode.key !== "confidence" || result.confidence_heatmap_url
          ).map((mode) => (
            <button
              key={mode.key}
              onClick={() => setTextureMode(mode.key)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                textureMode === mode.key
                  ? "bg-accent-500 text-base-950"
                  : "text-slate-300 hover:bg-base-700"
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>

        <div className="flex gap-1 rounded-xl bg-base-900/85 p-1 shadow-lg backdrop-blur">
          <button
            onClick={exportGLB}
            title="Download the 3D model as a GLB file"
            className="rounded-lg px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-base-700"
          >
            ⬇ 3D model (.glb)
          </button>
          <button
            onClick={exportOBJ}
            title="Download the 3D model as an OBJ file"
            className="rounded-lg px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-base-700"
          >
            ⬇ 3D model (.obj)
          </button>
        </div>
      </div>

      {/* One-time navigation hint */}
      {showHint && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-base-900/85 px-4 py-2 text-xs text-slate-300 shadow-lg backdrop-blur transition-opacity">
          {viewMode === "flythrough"
            ? "🕹️ WASD or arrows to fly · Drag to look · Space/Shift or scroll for altitude"
            : "🖱️ Drag to look around · Scroll to zoom"}
        </div>
      )}

      {hoverInfo && (
        <div className="pointer-events-none absolute bottom-4 left-4 rounded-xl bg-base-900/90 px-4 py-3 text-xs text-slate-200 shadow-lg backdrop-blur">
          <div>
            {hoverInfo.isCalibrated ? (
              <>
                <span className="font-medium text-slate-100">{hoverInfo.height} m</span>{" "}
                <span className="text-slate-500">elevation</span>
              </>
            ) : (
              <>
                <span className="font-medium text-slate-100">{hoverInfo.height}</span>{" "}
                <span className="text-slate-500">relative height (no real units yet)</span>
              </>
            )}
          </div>
          {hoverInfo.slopeDeg != null && (
            <div className="mt-0.5 text-slate-400">Slope: {hoverInfo.slopeDeg}°</div>
          )}
        </div>
      )}
    </div>
  );
}

function isCalibratedStage(stage) {
  return stage === "dem_calibrated" || stage === "lora_calibrated";
}
