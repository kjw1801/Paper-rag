'use client';

import { useState } from 'react';
import {
  ArrowUpRight,
  BookOpen,
  Bot,
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
};

type AskResponse = {
  answer: string;
  sources: Source[];
  grounded: boolean;
};

const suggestions = [
  '이 논문이 해결하려는 문제는 무엇인가요?',
  '그룹 추천은 어떤 단계로 구성되나요?',
  '제안 방법의 F-measure는 얼마인가요?',
  '실험 데이터는 어떻게 수집했나요?',
];

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';

export default function Home() {
  const [question, setQuestion] = useState(suggestions[0]);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function ask(event?: { preventDefault(): void }) {
    event?.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim(), top_k: 4 }),
      });

      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? '답변 서버에 연결할 수 없습니다.');
      }

      setResult((await response.json()) as AskResponse);
    } catch (caught) {
      setResult(null);
      setError(caught instanceof Error ? caught.message : '질문 처리 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  }

  function selectSuggestion(suggestion: string) {
    setQuestion(suggestion);
    setResult(null);
    setError('');
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
            rel="noreferrer"
            className="inline-flex h-9 items-center gap-1.5 rounded-full border border-slate-200 bg-white px-4 text-sm font-medium transition hover:bg-slate-100"
          >
            논문 원문 <ArrowUpRight className="size-4" />
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
            내 논문에 질문하고,
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
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6 sm:p-8">
              <form onSubmit={ask} className="flex gap-2">
                <Input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="논문에 대해 궁금한 내용을 입력하세요"
                  aria-label="논문 질문"
                  className="h-12 rounded-xl border-slate-300 px-4 text-base"
                />
                <Button
                  type="submit"
                  disabled={loading || !question.trim()}
                  className="h-12 rounded-xl bg-slate-950 px-5 text-white hover:bg-slate-800"
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
                    <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">{error}</p>
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
                    <div className="mb-5 flex items-center gap-2">
                      <Badge className={result.grounded ? 'bg-cyan-100 text-cyan-900' : 'bg-orange-100 text-orange-900'}>
                        {result.grounded ? '근거 확인됨' : '근거 없음'}
                      </Badge>
                    </div>
                    <p className="text-lg leading-8 text-slate-700">{result.answer}</p>
                    {result.sources.length > 0 && (
                      <div className="mt-7 space-y-3 border-t border-slate-200 pt-5">
                        <p className="text-sm font-semibold text-slate-900">검색 근거</p>
                        {result.sources.map((source, index) => (
                          <div key={`${source.page}-${index}`} className="rounded-xl border border-slate-200 bg-white p-4">
                            <div className="mb-2 flex items-center justify-between">
                              <span className="flex items-center gap-2 text-sm font-semibold text-cyan-800">
                                <FileText className="size-4" /> PDF {source.page}페이지
                              </span>
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
              <strong className="block text-slate-900">Paper</strong>
              Personalized Group Recommendation Using Collaborative Filtering and Frequent Pattern
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}
