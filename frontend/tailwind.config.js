import animate from "tailwindcss-animate";

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono Variable"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
        secondary: { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--secondary-foreground))" },
        destructive: { DEFAULT: "hsl(var(--destructive))", foreground: "hsl(var(--destructive-foreground))" },
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        accent: { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--accent-foreground))" },
        popover: { DEFAULT: "hsl(var(--popover))", foreground: "hsl(var(--popover-foreground))" },
        card: { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--card-foreground))" },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar))", foreground: "hsl(var(--sidebar-foreground))",
          border: "hsl(var(--sidebar-border))", accent: "hsl(var(--sidebar-accent))",
        },
        "status-good": "var(--status-good)", "status-good-bg": "var(--status-good-bg)",
        "status-warning": "var(--status-warning)", "status-warning-bg": "var(--status-warning-bg)",
        "status-sleeping": "var(--status-sleeping)", "status-sleeping-bg": "var(--status-sleeping-bg)",
        "status-critical": "var(--status-critical)", "status-critical-bg": "var(--status-critical-bg)",
        "status-neutral": "var(--status-neutral)", "status-neutral-bg": "var(--status-neutral-bg)",
        "chart-1": "var(--chart-1)", "chart-2": "var(--chart-2)", "chart-3": "var(--chart-3)", "chart-4": "var(--chart-4)",
        "chart-5": "var(--chart-5)", "chart-6": "var(--chart-6)", "chart-7": "var(--chart-7)", "chart-8": "var(--chart-8)",
        "chart-external": "var(--chart-external)", "chart-reserve": "var(--chart-reserve)", "chart-free": "var(--chart-free)",
      },
      borderRadius: { lg: "var(--radius)", md: "calc(var(--radius) - 2px)", sm: "calc(var(--radius) - 4px)" },
      maxWidth: { content: "1440px" },
    },
  },
  plugins: [animate],
};
