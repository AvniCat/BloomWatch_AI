import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        teal: {
          950: '#0F3D3E',
          900: '#124e4f',
        },
        cream: {
          DEFAULT: '#F1DEAB',
          100: '#faf1d4',
        },
      },
      fontFamily: {
        serif: ['Georgia', 'serif'],
      },
    },
  },
  plugins: [],
};
export default config;
