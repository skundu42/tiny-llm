export const appName = 'tiny-llm';
export const appDescription = 'A ~500M-parameter LLM pretrained from scratch';
export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

const configuredSiteUrl = process.env.NEXT_PUBLIC_SITE_URL;
const vercelSiteUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL ?? process.env.VERCEL_URL;
export const siteUrl = configuredSiteUrl ?? (vercelSiteUrl ? `https://${vercelSiteUrl}` : 'http://localhost:3000');

export const gitConfig = {
  user: 'skundu42',
  repo: 'tiny-llm',
  branch: 'main',
};
