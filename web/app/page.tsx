'use client';

import { useEffect, useRef, useState } from 'react';
import {
  ArrowUpRight,
  BookOpen,
  Bot,
  Code2,
  FileText,
  LoaderCircle,
  Search,
  ShieldCheck,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';

type Source = {
  page: number;
  snippet: string;
  relevance: number;
  cited: boolean;
};

type AskResponse = {
  answer: string;
  sources: Source[];
  grounded: boolean;
  cited_pages: number[];
};

type ErrorState = {
  message: string;
  code: string;
  status?: number;
};

type TurnstileApi = {
  render: (
    container: HTMLElement,
    options: {
      sitekey: string;
      action: string;
      theme: 'auto';
      size: 'flexible';
      appearance: 'interaction-only';
      callback: (token: string) => void;
      'expired-callback': () => void;
      'error-callback': () => void;
    },
  ) => string;
  reset: (widgetId: string) => void;
  remove: (widgetId: string) => void;
};

function getTurnstile(): TurnstileApi | undefined {
  return (window as Window & { turnstile?: TurnstileApi }).turnstile;
}

const suggestions = [
  '이 논문이 해결하려는 문제는 무엇인가요?',
  '그룹 추천은 어떤 단계로 구성되나요?',
  '제안 방법의 F-measure는 얼마인가요?',
  '실험 데이터는 어떻게 수집했나요?',
];

const performanceMetrics = [
  { label: '협업 필터링', value: 0.0977, color: 'bg-slate-400' },
  { label: '빈발 패턴', value: 0.12366, color: 'bg-cyan-500' },
  { label: '그룹 추천', value: 0.15435, color: 'bg-cyan-700' },
];

const performanceScaleMax = 0.16;

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';

export default function Home() {
  const [question, setQuestion] = useState(suggestions[0]);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<ErrorState | null>(null);
  const [loading, setLoading] = useState(false);
  const [turnstileReady, setTurnstileReady] = useState(false);
  const [turnstileError, setTurnstileError] = useState(false);
  const turnstileContainerRef = useRef<HTMLDivElement>(null);
  const turnstileTokenRef = useRef('');
  const turnstileWidgetIdRef = useRef<string | null>(null);
  const turnstileSiteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY ?? '';

  useEffect(() => {
    if (!turnstileSiteKey) return;

    const renderWidget = () => {
      const turnstile = getTurnstile();
      const container = turnstileContainerRef.current;
      if (!turnstile || !container || turnstileWidgetIdRef.current) return;
      turnstileWidgetIdRef.current = turnstile.render(container, {
        sitekey: turnstileSiteKey,
        action: 'paper_ask',
        theme: 'auto',
        size: 'flexible',
        appearance: 'interaction-only',
        callback: (token) => {
          turnstileTokenRef.current = token;
          setTurnstileReady(true);
          setTurnstileError(false);
        },
        'expired-callback': () => {
          turnstileTokenRef.current = '';
          setTurnstileReady(false);
        },
        'error-callback': () => {
          turnstileTokenRef.current = '';
          setTurnstileReady(false);
          setTurnstileError(true);
        },
      });
    };

    const scriptId = 'cloudflare-turnstile-script';
    let script = document.getElementById(scriptId) as HTMLScriptElement | null;
    if (script) {
      if (getTurnstile()) renderWidget();
      else script.addEventListener('load', renderWidget);
    } else {
      script = document.createElement('script');
      script.id = scriptId;
      script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
      script.async = true;
      script.defer = true;
      script.addEventListener('load', renderWidget);
      document.head.appendChild(script);
    }

    return () => {
      script?.removeEventListener('load', renderWidget);
      const turnstile = getTurnstile();
      if (turnstile && turnstileWidgetIdRef.current) turnstile.remove(turnstileWidgetIdRef.current);
      turnstileWidgetIdRef.current = null;
    };
  }, [turnstileSiteKey]);

  async function ask(event?: { preventDefault(): void }) {
    event?.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const turnstileToken = turnstileTokenRef.current;
    if (turnstileSiteKey && !turnstileToken) {
      setError({ message: '보안 확인을 완료한 뒤 질문해 주세요.', code: 'TURNSTILE_REQUIRED' });
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim(), top_k: 4, turnstile_token: turnstileToken || undefined }),
      });

      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as {
          detail?: string | { code?: string; message?: string };
        } | null;
        const detail = body?.detail;
        const message = typeof detail === 'string' ? detail : detail?.message ?? '답변 서버에 연결할 수 없습니다.';
        const code = typeof detail === 'object' && detail?.code
          ? detail.code
          : response.status === 429
            ? 'DEMO_RATE_LIMITED'
            : `HTTP_${response.status}`;
        setError({ message, code, status: response.status });
        return;
      }

      setResult((await response.json()) as AskResponse);
    } catch (caught) {
      setResult(null);
      setError({
        message: caught instanceof Error ? caught.message : '질문 처리 중 오류가 발생했습니다.',
        code: 'CLIENT_ERROR',
      });
    } finally {
      setLoading(false);
      if (turnstileSiteKey) {
        turnstileTokenRef.current = '';
        setTurnstileReady(false);
        const turnstile = getTurnstile();
        if (turnstile && turnstileWidgetIdRef.current) turnstile.reset(turnstileWidgetIdRef.current);
      }
    }
  }

  function selectSuggestion(suggestion: string) {
    setQuestion(suggestion);
    setResult(null);
    setError(null);
  }

  return (
    <main className="min-h-screen bg-grid text-slate-950">
      <header className="border-b border-slate-200/90 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
          <div className="flex items-center gap-3 font-semibold tracking-tight">
            <span className="grid size-9 place-items-center rounded-xl bg-slate-950 text-white">
              <Bot className="size-4" />
            </span>
            Paper RAG
          </div>
          <a
            href="/paper.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-9 items-center gap-1.5 rounded-full border border-slate-200 bg-white px-4 text-sm font-medium transition hover:bg-slate-100"
          >
            논문 PDF 보기 <ArrowUpRight className="size-4" />
          </a>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-5 pb-16 pt-12 sm:px-8 sm:pt-16">
        <div className="mb-10 max-w-3xl">
          <div className="mb-5 flex flex-wrap gap-2">
            <Badge className="h-7 bg-cyan-100 px-3 text-cyan-900">LANGCHAIN · FASTAPI · CHROMA</Badge>
            <Badge variant="outline" className="h-7 border-slate-300 bg-white px-3 text-slate-600">PDF 7 PAGES</Badge>
          </div>
          <h1 className="text-balance text-4xl font-semibold leading-[1.12] tracking-[-0.045em] sm:text-6xl">
            연구 논문에 질문하고,
            <span className="text-cyan-700"> 페이지 근거까지 확인하세요.</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600">
            「협업 필터링과 빈발 패턴을 이용한 개인화된 그룹 추천」을 페이지 단위로 분할하고 임베딩한 RAG 데모입니다.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_330px]">
          <Card className="border-0 bg-white py-0 shadow-xl shadow-slate-900/7 ring-slate-200">
            <CardHeader className="border-b border-slate-100 p-6 sm:p-8">
              <div className="flex items-center gap-3">
                <span className="grid size-11 place-items-center rounded-2xl bg-cyan-700 text-white">
                  <Search className="size-5" />
                </span>
                <div>
                  <CardTitle className="text-xl font-semibold">논문에 질문하기</CardTitle>
                  <CardDescription className="mt-1">검색된 문맥만 사용해 답변합니다.</CardDescription>
                  <p className="mt-2 max-w-lg text-xs leading-5 text-slate-500">
                    개인 프로젝트이며 무료 AI API 플랜으로 운영됩니다. 사용량에 따라 답변이 느리거나 일시 중단될 수 있습니다.
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6 sm:p-8">
              <form onSubmit={ask} className="flex flex-col gap-2 sm:flex-row">
                <Input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="논문에 대해 궁금한 내용을 입력하세요"
                  aria-label="논문 질문"
                  className="h-12 rounded-xl border-slate-300 px-4 text-base"
                />
                <Button
                  type="submit"
                  disabled={loading || !question.trim() || Boolean(turnstileSiteKey && !turnstileReady)}
                  className="h-12 w-full rounded-xl bg-slate-950 px-5 text-white hover:bg-slate-800 sm:w-auto"
                >
                  {loading ? <LoaderCircle className="animate-spin" /> : '질문'}
                </Button>
              </form>

              <div className="mt-4 flex flex-wrap gap-2">
                {suggestions.map((suggestion) => (
                  <Button
                    key={suggestion}
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => selectSuggestion(suggestion)}
                    className="h-auto rounded-full bg-slate-100 px-3 py-2 text-left text-xs text-slate-600 hover:bg-cyan-50 hover:text-cyan-800"
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>

              {turnstileSiteKey && (
                <div className="mt-4 min-h-8">
                  <div ref={turnstileContainerRef} />
                  {turnstileError && (
                    <p className="mt-2 text-xs text-orange-700">
                      보안 확인을 불러오지 못했습니다. 페이지를 새로고침해 주세요.
                    </p>
                  )}
                </div>
              )}

              <div className="mt-8 min-h-[300px] rounded-2xl border border-slate-200 bg-slate-50 p-5 sm:p-7">
                {loading && (
                  <div className="flex h-[240px] flex-col items-center justify-center text-slate-500">
                    <LoaderCircle className="mb-3 size-7 animate-spin text-cyan-700" />
                    관련 문장을 검색하고 있습니다.
                  </div>
                )}

                {!loading && error && (
                  <div className="flex h-[240px] flex-col items-center justify-center text-center">
                    <Bot className="mb-3 size-8 text-orange-500" />
                    <p className="font-medium text-slate-800">답변을 불러오지 못했습니다.</p>
                    <p className="mt-2 font-mono text-xs text-orange-700">
                      {error.code}{error.status ? ` · HTTP ${error.status}` : ''}
                    </p>
                    <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">{error.message}</p>
                  </div>
                )}

                {!loading && !error && !result && (
                  <div className="flex h-[240px] flex-col items-center justify-center text-center text-slate-500">
                    <BookOpen className="mb-3 size-8 text-cyan-700" />
                    <p className="font-medium text-slate-700">질문을 입력하면 답변과 근거가 이곳에 표시됩니다.</p>
                    <p className="mt-2 text-sm">논문에 없는 내용은 답하지 않습니다.</p>
                  </div>
                )}

                {!loading && result && (
                  <div>
                    <div className="mb-5 flex flex-wrap items-center gap-2">
                      <Badge className={result.grounded ? 'bg-cyan-100 text-cyan-900' : 'bg-orange-100 text-orange-900'}>
                        {result.grounded ? '근거 확인됨' : '근거 없음'}
                      </Badge>
                      {result.cited_pages.map((page) => (
                        <a
                          key={page}
                          href={`/paper.pdf#page=${page}`}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex h-6 items-center gap-1 rounded-full border border-cyan-200 bg-white px-2.5 text-xs font-medium text-cyan-800 hover:bg-cyan-50"
                        >
                          <FileText className="size-3" /> {page}페이지
                        </a>
                      ))}
                    </div>
                    <p className="whitespace-pre-line text-lg leading-8 text-slate-700">{result.answer}</p>
                    {result.sources.length > 0 && (
                      <div className="mt-7 space-y-3 border-t border-slate-200 pt-5">
                        <p className="text-sm font-semibold text-slate-900">검색 근거 <span className="font-normal text-slate-500">· 페이지를 누르면 PDF 해당 페이지가 열립니다</span></p>
                        {result.sources.map((source, index) => (
                          <div
                            key={`${source.page}-${index}`}
                            className={`rounded-xl border bg-white p-4 ${source.cited ? 'border-cyan-300' : 'border-slate-200'}`}
                          >
                            <div className="mb-2 flex items-center justify-between">
                              <a
                                href={`/paper.pdf#page=${source.page}`}
                                target="_blank"
                                rel="noreferrer"
                                className="flex items-center gap-2 text-sm font-semibold text-cyan-800 hover:underline"
                              >
                                <FileText className="size-4" /> PDF {source.page}페이지
                                <ArrowUpRight className="size-3.5" />
                                {source.cited && <span className="text-xs font-normal text-slate-500">답변에 인용됨</span>}
                              </a>
                              <span className="font-mono text-xs text-slate-400">{source.relevance.toFixed(2)}</span>
                            </div>
                            <p className="text-sm leading-6 text-slate-600">{source.snippet}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <aside className="space-y-5">
            <Card className="border-0 bg-white ring-slate-200">
              <CardHeader className="p-6 pb-3">
                <CardTitle className="text-lg font-semibold">추천 성능 비교</CardTitle>
                <CardDescription>평균 F-measure · 사용자 248명</CardDescription>
              </CardHeader>
              <CardContent className="px-6 pb-6">
                <figure className="space-y-4">
                  <figcaption className="sr-only">
                    협업 필터링 0.0977, 빈발 패턴 0.12366, 그룹 추천 0.15435의 평균 F-measure 비교
                  </figcaption>
                  {performanceMetrics.map((metric) => (
                    <div key={metric.label}>
                      <div className="mb-1.5 flex items-center justify-between gap-3 text-sm">
                        <span className="font-medium text-slate-700">{metric.label}</span>
                        <span className="font-mono text-xs text-slate-500">{metric.value}</span>
                      </div>
                      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full ${metric.color}`}
                          style={{ width: `${(metric.value / performanceScaleMax) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="flex justify-between border-t border-slate-100 pt-2 font-mono text-[11px] text-slate-400">
                    <span>0.00</span>
                    <span>0.08</span>
                    <span>0.16</span>
                  </div>
                </figure>
                <p className="mt-4 rounded-xl bg-cyan-50 px-3 py-2.5 text-sm leading-5 text-cyan-900">
                  그룹 추천은 협업 필터링보다 F-measure가 <strong>0.05665</strong> 높았습니다.
                </p>
                <a
                  href="/paper.pdf#page=5"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-cyan-800"
                >
                  논문 5페이지 표 1 확인 <ArrowUpRight className="size-3.5" />
                </a>
              </CardContent>
            </Card>

            <Card className="border-0 bg-slate-950 text-white ring-0">
              <CardHeader className="p-6 pb-3">
                <CardTitle className="text-lg font-semibold">처리 흐름</CardTitle>
                <CardDescription className="text-slate-400">질문 한 번에 실행되는 과정</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 px-6 pb-6">
                {[
                  ['01', '질문 임베딩'],
                  ['02', 'Chroma top-k 검색'],
                  ['03', 'Gemini 근거 답변'],
                  ['04', '문장·페이지 반환'],
                ].map(([number, label]) => (
                  <div key={number} className="flex items-center gap-4 border-t border-white/10 pt-4">
                    <span className="font-mono text-xs text-cyan-300">{number}</span>
                    <span className="text-sm text-slate-200">{label}</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="border-0 bg-cyan-50 ring-cyan-100">
              <CardContent className="p-6">
                <ShieldCheck className="mb-4 size-7 text-cyan-700" />
                <p className="font-semibold text-slate-900">근거 없는 답변 차단</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  검색 유사도 임계값과 제한 프롬프트를 함께 적용했습니다.
                </p>
              </CardContent>
            </Card>

            <div className="rounded-2xl border border-slate-200 bg-white p-5 text-sm leading-6 text-slate-500">
              <strong className="block text-slate-900">연구 논문</strong>
              <p className="mt-1">
                Personalized Group Recommendation Using Collaborative Filtering and Frequent Pattern
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <a
                  href="https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002130830"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-3 py-1.5 font-medium text-slate-700 transition hover:border-cyan-300 hover:bg-cyan-50 hover:text-cyan-800"
                >
                  KCI에서 보기 <ArrowUpRight className="size-3.5" />
                </a>
                <a
                  href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE06747562"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-3 py-1.5 font-medium text-slate-700 transition hover:border-cyan-300 hover:bg-cyan-50 hover:text-cyan-800"
                >
                  DBpia에서 보기 <ArrowUpRight className="size-3.5" />
                </a>
                <a
                  href="https://github.com/kjw1801/Paper-rag"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-3 py-1.5 font-medium text-slate-700 transition hover:border-cyan-300 hover:bg-cyan-50 hover:text-cyan-800"
                >
                  <Code2 className="size-3.5" /> GitHub 코드 보기
                </a>
              </div>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}
