/** @type {import('next').NextConfig} */
const nextConfig = {
  // Removed 'output: export' to enable API routes for webhooks
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig
