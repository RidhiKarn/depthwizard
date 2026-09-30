"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import UploadPanel from "@/components/UploadPanel";
import TerrainViewer from "@/components/TerrainViewer";
import ShadowPanel from "@/components/ShadowPanel";
import ResultSummary from "@/components/ResultSummary";
import AboutPanel from "@/components/AboutPanel";
import { useAuth } from "@/lib/auth";

/**
 * The actual tool — upload, calibration, 3D viewer, shadow cross-check.
 * Lives behind login at /app; the public landing page (app/page.js)
 * explains what this does before anyone signs up.
 */
export default function ToolPage() {
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);
  const { ready, token, email, signOut } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (ready && !token) {
      router.replace("/login");
    }
  }, [ready, token, router]);

  const handleResult = (newResult, newFile) => {
    setResult(newResult);
    setFile(newFile);
  };

  if (!ready || !token) {
    return <div className="min-h-screen bg-base-950" />;
  }

  return (
    <main className="flex min-h-screen flex-col bg-base-950">
      <header className="flex items-center justify-between border-b border-base-800 px-6 py-4">
        <h1 className="text-xl font-semibold text-white">DepthWizard</h1>
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-400">{email}</span>
          <button
            onClick={() => {
              signOut();
              router.push("/");
            }}
            className="rounded-lg bg-base-800 px-3 py-1.5 text-sm text-slate-300 hover:bg-base-700"
          >
            Sign out
          </button>
          <AboutPanel />
        </div>
      </header>

      <section className="flex flex-1 flex-col gap-6 p-6 lg:flex-row">
        <div className="flex flex-col items-start gap-4 lg:w-[420px] lg:shrink-0">
          <UploadPanel onResult={handleResult} />
          <ResultSummary result={result} />
          <ShadowPanel file={file} result={result} />
        </div>

        <div className="min-h-[480px] flex-1 overflow-hidden rounded-2xl border border-base-800 bg-base-900">
          <TerrainViewer result={result} />
        </div>
      </section>
    </main>
  );
}
