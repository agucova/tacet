import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightPageActions from "starlight-page-actions";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import tailwindcss from "@tailwindcss/vite";

// https://astro.build/config
export default defineConfig({
  site: "https://tacet.sh",

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },

  integrations: [
    starlight({
      title: "Tacet",
      description:
        "Detect timing side channels in cryptographic code with statistically rigorous methods",
      plugins: [
        starlightPageActions({
          baseUrl: "https://tacet.sh",
          actions: {
            chatgpt: true,
            claude: true,
            t3chat: true,
            markdown: true,
            v0: false,
          },
        }),
      ],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/agucova/tacet",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/agucova/tacet/edit/main/website/",
      },
      lastUpdated: true,
      components: {
        Header: "./src/components/Header.astro",
        SiteTitle: "./src/components/SiteTitle.astro",
        ThemeSelect: "./src/components/ThemeSelect.astro",
        Sidebar: "./src/components/Sidebar.astro",
        Head: "./src/components/CustomHead.astro",
      },
      customCss: [
        // Fonts
        "@fontsource/jetbrains-mono/400.css",
        "@fontsource/jetbrains-mono/500.css",
        "@fontsource/jetbrains-mono/600.css",
        // Design system
        "./src/styles/global.css",
        "./src/styles/tokens.css",
        "./src/styles/custom.css",
        // KaTeX for math
        "katex/dist/katex.min.css",
      ],
      sidebar: [
        {
          label: "Getting Started",
          autogenerate: { directory: "getting-started" },
        },
        {
          label: "Core Concepts",
          autogenerate: { directory: "core-concepts" },
        },
        {
          label: "Guides",
          autogenerate: { directory: "guides" },
        },
        {
          label: "Reference",
          autogenerate: { directory: "reference" },
        },
        {
          label: "API Reference",
          autogenerate: { directory: "api" },
        },
        {
          label: "Case Studies",
          autogenerate: { directory: "case-studies" },
        },
      ],
      expressiveCode: {
        themes: ["github-dark"],
        styleOverrides: {
          borderRadius: "8px",
        },
      },
      head: [
        // Fonts
        {
          tag: "link",
          attrs: {
            rel: "preconnect",
            href: "https://fonts.googleapis.com",
          },
        },
        {
          tag: "link",
          attrs: {
            rel: "preconnect",
            href: "https://fonts.gstatic.com",
            crossorigin: true,
          },
        },
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Recursive:CASL,CRSV,MONO@0,0,0&display=swap",
          },
        },
        // Favicon
        {
          tag: "link",
          attrs: {
            rel: "icon",
            type: "image/svg+xml",
            href: "/favicon.svg",
          },
        },
        // Basic meta
        {
          tag: "meta",
          attrs: {
            name: "keywords",
            content:
              "timing side channel, cryptography, security, rust, constant-time, tacet",
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "theme-color",
            content: "#0C0C0C",
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "author",
            content: "Agustín Covarrubias",
          },
        },
        // Open Graph
        {
          tag: "meta",
          attrs: {
            property: "og:type",
            content: "website",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:site_name",
            content: "Tacet",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:image",
            content: "https://tacet.sh/og-image.png",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:image:width",
            content: "1200",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:image:height",
            content: "630",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:locale",
            content: "en_US",
          },
        },
        // Twitter Card
        {
          tag: "meta",
          attrs: {
            name: "twitter:card",
            content: "summary_large_image",
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "twitter:image",
            content: "https://tacet.sh/og-image.png",
          },
        },
      ],
    }),
  ],

  vite: {
    plugins: [tailwindcss()],
  },
});
