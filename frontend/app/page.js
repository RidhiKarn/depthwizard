"use client";

import Link from "next/link";
import { useAuth } from "@/lib/auth";

const FEATURES = [
  {
    icon: "🗻",
    title: "Elevation mapping",
    description:
      "Generates a digital surface model from a single photo — every upload gets an accurate relative height map, no stereo pairs or LiDAR needed.",
  },
  {
    icon: "📐",
    title: "Real-world calibration",
    description:
      "Photos with location data (GeoTIFF) are automatically matched against satellite elevation data, so heights come out in true meters.",
  },
  {
    icon: "🌇",
    title: "Shadow cross-check",
    description:
      "An independent, physics-based estimate from cast shadows lets you double-check building heights from a second angle.",
  },
  {
    icon: "🕹️",
    title: "3D flythrough",
    description:
      "Explore the result in an interactive viewer — orbit around it or fly through it freely — and download the terrain or elevation data.",
  },
];

export default function LandingPage() {
  const { ready, token } = useAuth();
  const signedIn = ready && !!token;

  return (
    <main className="min-h-screen bg-base-950">
      <header className="flex items-center justify-between px-6 py-5">
        <h1 className="text-lg font-semibold text-white">DepthWizard</h1>
        <nav className="flex items-center gap-3">
          {signedIn ? (
            <Link
              href="/app"
              className="rounded-lg bg-accent-500 px-4 py-2 text-sm font-medium text-base-950 hover:bg-accent-400"
            >
              Open App
            </Link>
          ) : (
            <>
              <Link
                href="/login"
                className="rounded-lg px-4 py-2 text-sm font-medium text-slate-300 hover:text-white"
              >
                Log In
              </Link>
              <Link
                href="/signup"
                className="rounded-lg bg-accent-500 px-4 py-2 text-sm font-medium text-base-950 hover:bg-accent-400"
              >
                Sign Up
              </Link>
            </>
          )}
        </nav>
      </header>

      <section className="mx-auto max-w-3xl px-6 pb-16 pt-12 text-center sm:pt-20">
        <h2 className="text-3xl font-semibold text-white sm:text-5xl">
          Turn one photo into a
          <span className="text-accent-400"> navigable 3D terrain</span>
        </h2>
        <p className="mx-auto mt-5 max-w-xl text-base text-slate-400">
          Upload a single satellite or aerial image and DepthWizard builds a
          realistic elevation map you can fly through — automatically
          calibrated to real-world height whenever location data is
          available.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          {signedIn ? (
            <Link
              href="/app"
              className="rounded-lg bg-accent-500 px-6 py-3 font-medium text-base-950 hover:bg-accent-400"
            >
              Open App
            </Link>
          ) : (
            <>
              <Link
                href="/signup"
                className="rounded-lg bg-accent-500 px-6 py-3 font-medium text-base-950 hover:bg-accent-400"
              >
                Sign Up
              </Link>
              <Link
                href="/login"
                className="rounded-lg bg-base-800 px-6 py-3 font-medium text-slate-200 hover:bg-base-700"
              >
                Log In
              </Link>
            </>
          )}
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-6 pb-20">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="rounded-2xl border border-base-800 bg-base-900 p-6"
            >
              <div className="text-3xl">{feature.icon}</div>
              <h3 className="mt-3 font-semibold text-white">{feature.title}</h3>
              <p className="mt-1.5 text-sm text-slate-400">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t border-base-800 px-6 py-6 text-center text-xs text-slate-600">
        DepthWizard · Built for Smart India Hackathon 2026, Problem Statement SIH26175
      </footer>
    </main>
  );
}
