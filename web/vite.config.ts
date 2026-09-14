import { sites } from '@openai/sites-vite-plugin';
import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig, loadEnv } from 'vite';
import hostingConfig from './.openai/hosting.json';

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  '00000000-0000-4000-8000-000000000000';

const { d1, r2 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === 'seatbelt';

const localBindingConfig = {
  main: 'vinext/server/fetch-handler',
  compatibility_flags: ['nodejs_compat'],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: 'site-creator-d1',
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: 'site-creator-r2',
        },
      ]
    : [],
};

// URL.hostname은 IPv6를 대괄호째로 돌려준다.
const LOCAL_HOSTNAMES = new Set(['localhost', '127.0.0.1', '0.0.0.0', '[::1]']);

// 프로덕션 빌드에 NEXT_PUBLIC_* 값이 없으면 번들이 조용히 localhost로 폴백해
// 배포한 뒤에야 "Failed to fetch"로 드러난다. 빌드 단계에서 먼저 막는다.
function assertDeployableEnvironment(mode: string) {
  const env = loadEnv(mode, process.cwd(), 'NEXT_PUBLIC_');
  const missing = [
    'NEXT_PUBLIC_API_URL',
    'NEXT_PUBLIC_TURNSTILE_SITE_KEY',
  ].filter((name) => !env[name]);

  if (missing.length > 0) {
    throw new Error(
      `프로덕션 빌드에 ${missing.join(', ')}이(가) 없습니다. ` +
        'web/.env.production을 만들거나 빌드 환경변수로 등록하세요.',
    );
  }

  const raw = env.NEXT_PUBLIC_API_URL;
  let apiUrl: URL;
  try {
    apiUrl = new URL(raw);
  } catch {
    throw new Error(`NEXT_PUBLIC_API_URL(${raw})이 올바른 주소가 아닙니다.`);
  }

  if (LOCAL_HOSTNAMES.has(apiUrl.hostname)) {
    throw new Error(
      `NEXT_PUBLIC_API_URL이 로컬 주소(${raw})입니다. ` +
        '배포된 백엔드 주소를 넣으세요.',
    );
  }

  // HTTPS 사이트에서 HTTP API를 부르면 브라우저가 Mixed Content로 막는다.
  if (apiUrl.protocol !== 'https:') {
    throw new Error(
      `NEXT_PUBLIC_API_URL이 ${apiUrl.protocol.replace(':', '')} 주소입니다. ` +
        'HTTPS로 서비스되는 사이트에서는 차단되므로 https 주소를 넣으세요.',
    );
  }
}

export default defineConfig(async ({ command, mode }) => {
  if (command === 'build') {
    assertDeployableEnvironment(mode);
  }

  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= 'false';
  process.env.WRANGLER_LOG_PATH ??= '.wrangler/logs';
  process.env.MINIFLARE_REGISTRY_PATH ??= '.wrangler/registry';

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import('@cloudflare/vite-plugin');

  return {
    css: { postcss: { plugins: [tailwindcss()] } },
    server: isCodexSeatbeltSandbox
      ? { watch: { useFsEvents: false, usePolling: true } }
      : undefined,
    plugins: [
      vinext(),
      sites(),
      cloudflare({
        viteEnvironment: { name: 'rsc', childEnvironments: ['ssr'] },
        config: localBindingConfig,
      }),
    ],
  };
});
