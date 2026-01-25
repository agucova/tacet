import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightPageActions from 'starlight-page-actions';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://timingoracle.agus.sh',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: 'timing-oracle',
      description: 'Detect timing side channels in cryptographic code with statistically rigorous methods',
      plugins: [
        starlightPageActions({
          baseUrl: 'https://timingoracle.agus.sh',
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
        { icon: 'github', label: 'GitHub', href: 'https://github.com/agucova/timing-oracle' },
      ],
      editLink: {
        baseUrl: 'https://github.com/agucova/timing-oracle/edit/main/website/',
      },
      lastUpdated: true,
      customCss: [
        './src/styles/custom.css',
        'katex/dist/katex.min.css',
      ],
      sidebar: [
        {
          label: 'Getting Started',
          autogenerate: { directory: 'getting-started' },
        },
        {
          label: 'Core Concepts',
          autogenerate: { directory: 'core-concepts' },
        },
        {
          label: 'Guides',
          autogenerate: { directory: 'guides' },
        },
        {
          label: 'Reference',
          autogenerate: { directory: 'reference' },
        },
        {
          label: 'API Reference',
          autogenerate: { directory: 'api' },
        },
        {
          label: 'Case Studies',
          autogenerate: { directory: 'case-studies' },
        },
      ],
      expressiveCode: {
        themes: ['github-dark', 'github-light'],
      },
      head: [
        {
          tag: 'meta',
          attrs: {
            name: 'keywords',
            content: 'timing side channel, cryptography, security, rust, constant-time',
          },
        },
      ],
    }),
  ],
});
