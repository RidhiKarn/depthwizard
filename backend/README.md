---
title: DepthWizard Backend
emoji: 🗻
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DepthWizard — backend

FastAPI + PyTorch inference service for DepthWizard (SIH26175). Deployed
here as a Docker Space — see the root project README (in the main repo,
not this Space) for full architecture, datasets, and calibration
details.

This Space only serves the API (`/docs` has the interactive reference).
The actual web app is the separate Next.js frontend, deployed on Vercel,
which talks to this Space's URL.

## Required Space secrets

Set these under this Space's **Settings → Variables and secrets**:

| Name | Value |
|---|---|
| `DEPTHWIZARD_JWT_SECRET` | any long random string — must be set, the code default is dev-only |
| `DEPTHWIZARD_CORS_ORIGINS` | your deployed Vercel URL, e.g. `https://your-app.vercel.app` |

## Local development

See the main repo's `backend/` folder and its own setup instructions —
this file is specifically for the deployed Space.
