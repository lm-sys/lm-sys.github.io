module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        sky: "#1d1d1f",
        paper: "#F6F6F6"
      },
    },
  },

  plugins: [require("tailwind-children")],
};
