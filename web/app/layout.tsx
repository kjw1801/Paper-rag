import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'Paper RAG · Jung Woo Kim',
  description: 'LangChain과 Gemini로 구현한 논문 근거 기반 질의응답 데모',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}

