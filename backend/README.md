# DepthWizard — backend

FastAPI + PyTorch inference service for DepthWizard (SIH26175). See the
root project README for full architecture, datasets, and calibration
details.

Deployed as a Docker web service (see `Dockerfile`) — originally planned
for Hugging Face Spaces, but as of the time this was written, Hugging
Face requires a paid plan to run Docker Spaces, so this is deployed on
Render's free tier instead. The same `Dockerfile` works unmodified on
either platform (or Railway, Fly.io, etc.) if that changes again.

This service only serves the API (`/docs` has the interactive
reference). The actual web app is the separate Next.js frontend,
deployed on Vercel, which talks to this service's URL.

## Required environment variables

Set these in your hosting platform's dashboard (e.g. Render →
Environment):

| Name | Value |
|---|---|
| `DEPTHWIZARD_JWT_SECRET` | any long random string — must be set, the code default is dev-only |
| `DEPTHWIZARD_CORS_ORIGINS` | your deployed frontend URL, e.g. `https://your-app.vercel.app` (comma-separate multiple) |

## Local development

See the main repo's `backend/` folder and its own setup instructions —
this file is specifically about the deployed service.
