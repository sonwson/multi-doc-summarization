/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: '#00d2b4'
      },
      boxShadow: {
        soft: '0 20px 45px -24px rgba(15, 23, 42, 0.22)'
      }
    }
  },
  plugins: []
};
