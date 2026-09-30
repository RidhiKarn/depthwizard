/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx}", "./components/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#05070a",
          900: "#0b0f16",
          800: "#131a24",
          700: "#1c2430",
        },
        accent: {
          400: "#5eead4",
          500: "#2dd4bf",
        },
      },
    },
  },
  plugins: [],
};
