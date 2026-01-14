/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Segoe UI"', 'system-ui', 'sans-serif'],
        mono: ['Consolas', '"JetBrains Mono"', 'monospace'],
      },
      colors: {
        vs: {
          black: '#0d0d0d',
          bg: '#181818',
          surface: '#1f1f1f',
          border: '#2d2d2d',
          hover: '#2a2a2a',
          text: '#d4d4d4',
          muted: '#6e7681',
          blue: '#3794ff',
          green: '#4ec9b0',
          yellow: '#dcdcaa',
          orange: '#ce9178',
          purple: '#c586c0',
          red: '#f14c4c',
          cyan: '#9cdcfe',
        }
      },
      fontSize: {
        'vs': '13px',
        'vs-sm': '12px',
        'vs-xs': '11px',
      }
    }
  },
  plugins: [],
}
