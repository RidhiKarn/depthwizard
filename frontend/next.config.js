/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    // Depth heatmap / input-image previews are served from the FastAPI
    // backend (localhost in dev, the deployed HF Space/Render URL in
    // prod) — not from Next's own domain, so remote patterns must be
    // opened up for <Image>. We use plain <img> in TerrainViewer for the
    // texture itself (needs a raw HTMLImageElement for three.js anyway),
    // but keep this here for any <Image> previews added later.
    remotePatterns: [
      { protocol: "http", hostname: "localhost" },
      { protocol: "https", hostname: "**" },
    ],
  },
};

module.exports = nextConfig;
