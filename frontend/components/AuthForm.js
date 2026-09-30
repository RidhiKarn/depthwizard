"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, login as apiLogin, signup as apiSignup } from "@/lib/api";
import { useAuth } from "@/lib/auth";

/**
 * Shared form for /login and /signup — the two flows only differ in
 * which API call they make and the surrounding copy/links.
 */
export default function AuthForm({ mode }) {
  const isSignup = mode === "signup";
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const { signIn } = useAuth();
  const router = useRouter();

  const onSubmit = async (e) => {
    e.preventDefault();
    setStatus("loading");
    setError(null);
    try {
      const body = isSignup ? await apiSignup(email, password) : await apiLogin(email, password);
      signIn(body.access_token, body.email);
      router.push("/app");
    } catch (err) {
      setStatus("idle");
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-base-950 px-4">
      <div className="w-full max-w-sm rounded-2xl border border-base-800 bg-base-900 p-8">
        <h1 className="text-xl font-semibold text-white">
          {isSignup ? "Create your account" : "Welcome back"}
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          {isSignup
            ? "Sign up to start turning photos into 3D terrain."
            : "Sign in to continue to DepthWizard."}
        </p>

        <form onSubmit={onSubmit} className="mt-6 space-y-4">
          <label className="block text-sm">
            <span className="mb-1 block text-slate-400">Email</span>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg bg-base-800 px-3 py-2 text-slate-100 outline-none ring-accent-500 focus:ring-2"
              placeholder="you@example.com"
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-slate-400">Password</span>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg bg-base-800 px-3 py-2 text-slate-100 outline-none ring-accent-500 focus:ring-2"
              placeholder="At least 8 characters"
            />
          </label>

          {error && (
            <p className="rounded-lg bg-red-950/50 px-3 py-2 text-sm text-red-300">{error}</p>
          )}

          <button
            type="submit"
            disabled={status === "loading"}
            className="w-full rounded-lg bg-accent-500 px-4 py-2 font-medium text-base-950 hover:bg-accent-400 disabled:opacity-50"
          >
            {status === "loading" ? "Please wait…" : isSignup ? "Sign up" : "Sign in"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-slate-500">
          {isSignup ? (
            <>
              Already have an account?{" "}
              <Link href="/login" className="text-accent-400 hover:underline">
                Sign in
              </Link>
            </>
          ) : (
            <>
              New here?{" "}
              <Link href="/signup" className="text-accent-400 hover:underline">
                Create an account
              </Link>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
