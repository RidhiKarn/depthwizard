"use client";

import { useState } from "react";

/**
 * The long explanation of what DepthWizard does, tucked behind an info
 * icon instead of always-visible text — keeps the landing page focused
 * on the upload action for first-time, non-technical visitors, while
 * still making the detail available to anyone who wants it.
 */
export default function AboutPanel() {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label="About DepthWizard"
        className="flex h-8 w-8 items-center justify-center rounded-full bg-base-800 text-sm text-slate-300 hover:bg-base-700"
      >
        ℹ️
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-10 z-50 w-80 rounded-xl border border-base-800 bg-base-900 p-4 text-sm text-slate-300 shadow-2xl sm:w-96">
            <h3 className="mb-2 font-semibold text-white">What is DepthWizard?</h3>
            <p className="mb-2">
              Upload a single satellite or aerial photo and DepthWizard builds a
              navigable 3D model of it — turning flat imagery into terrain you
              can fly around and inspect.
            </p>
            <p className="mb-2">
              Photos that carry location data (GeoTIFF) are automatically
              matched against real satellite elevation data, so heights come
              out in true meters. Ordinary photos (PNG/JPG) still get an
              accurate 3D shape, just without exact real-world measurements
              unless a fine-tuned model has been trained for that.
            </p>
            <p>
              A shadow-based cross-check and a model-agreement overlay are
              also available, so you can sanity-check the result from more
              than one angle.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
