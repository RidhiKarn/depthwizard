"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

const AuthContext = createContext(null);
const TOKEN_KEY = "depthwizard_token";
const EMAIL_KEY = "depthwizard_email";

/**
 * Session storage for the signed-in user. Kept in localStorage (not a
 * cookie) since the frontend and backend run on different origins in
 * dev and can be deployed separately — the token is just attached as an
 * Authorization header on API calls (see lib/api.js), never relied on
 * for anything security-critical beyond that.
 */
export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(null);
  const [email, setEmailState] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setTokenState(localStorage.getItem(TOKEN_KEY));
    setEmailState(localStorage.getItem(EMAIL_KEY));
    setReady(true);
  }, []);

  const signIn = useCallback((newToken, newEmail) => {
    localStorage.setItem(TOKEN_KEY, newToken);
    localStorage.setItem(EMAIL_KEY, newEmail);
    setTokenState(newToken);
    setEmailState(newEmail);
  }, []);

  const signOut = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(EMAIL_KEY);
    setTokenState(null);
    setEmailState(null);
  }, []);

  return (
    <AuthContext.Provider value={{ token, email, ready, signIn, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
