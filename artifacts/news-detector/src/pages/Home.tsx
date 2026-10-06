import React, { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useAnalyzeNews,
  useSubmitFeedback,
  useGetStats,
  useGetMlStatus,
  getGetStatsQueryKey,
  getGetMlStatusQueryKey,
} from "@workspace/api-client-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Database,
  Zap,
  Search,
  ChevronRight,
  ChevronDown,
  ExternalLink,
  RotateCcw,
} from "lucide-react";
import { format } from "date-fns";

type VerdictType = "REAL" | "CLICKBAIT" | "UNCERTAIN";

const verdictConfig = {
  REAL: {
    icon: CheckCircle2,
    color: "text-emerald-500",
    bg: "bg-emerald-500",
    border: "border-emerald-500",
    glow: "bg-emerald-500/10 border-emerald-500/30",
    chip: "bg-emerald-500/10 border-emerald-500/40 text-emerald-400",
    badge: "bg-emerald-500/20 text-emerald-300",
  },
  CLICKBAIT: {
    icon: AlertTriangle,
    color: "text-amber-500",
    bg: "bg-amber-500",
    border: "border-amber-500",
    glow: "bg-amber-500/10 border-amber-500/30",
    chip: "bg-amber-500/10 border-amber-500/40 text-amber-400",
    badge: "bg-amber-500/25 text-amber-300",
  },
  UNCERTAIN: {
    icon: HelpCircle,
    color: "text-sky-400",
    bg: "bg-sky-400",
    border: "border-sky-400",
    glow: "bg-sky-400/10 border-sky-400/30",
    chip: "bg-sky-400/10 border-sky-400/40 text-sky-300",
    badge: "bg-sky-400/20 text-sky-300",
  },
} as const;

export default function Home() {
  const queryClient = useQueryClient();
  const [headline, setHeadline] = useState("");
  const [body, setBody] = useState("");
  const [currentResultId, setCurrentResultId] = useState<number | null>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  const { data: mlStatus } = useGetMlStatus({
    query: {
      queryKey: getGetMlStatusQueryKey(),
      refetchInterval: (query) => (query.state.data?.ready ? false : 5000),
    },
  });

  const { data: stats, isLoading: statsLoading } = useGetStats({
    query: { queryKey: getGetStatsQueryKey() },
  });

  const analyzeNews = useAnalyzeNews({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: getGetStatsQueryKey() });
      },
    },
  });

  const submitFeedback = useSubmitFeedback({
    mutation: {
      onSuccess: () => {
        setFeedbackSubmitted(true);
        queryClient.invalidateQueries({ queryKey: getGetStatsQueryKey() });
        queryClient.invalidateQueries({ queryKey: getGetMlStatusQueryKey() });
      },
    },
  });

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    if (!headline.trim()) return;
    setFeedbackSubmitted(false);
    setSourcesExpanded(false);
    analyzeNews.mutate(
      { data: { headline, body: body.trim() || undefined } },
      { onSuccess: (data) => setCurrentResultId(data.id) }
    );
  };

  const handleFeedback = (verdict: "REAL" | "CLICKBAIT") => {
    if (!currentResultId) return;
    submitFeedback.mutate({ data: { analysisId: currentResultId, correctLabel: verdict } });
  };

  const handleReset = () => {
    setHeadline("");
    setBody("");
    setCurrentResultId(null);
    setFeedbackSubmitted(false);
    setSourcesExpanded(false);
    analyzeNews.reset();
  };

  const result = analyzeNews.data?.id === currentResultId ? analyzeNews.data : null;
  const verdict = result?.verdict as VerdictType | undefined;
  const cfg = verdict ? verdictConfig[verdict] : null;

  const isAnalyzing = analyzeNews.isPending;
  const modelReady = mlStatus?.ready ?? false;

  const totalScanned = stats?.totalAnalyzed ?? 0;

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary selection:text-primary-foreground pb-20">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-sm bg-primary flex items-center justify-center text-primary-foreground font-mono font-bold text-lg">
              {"//"}
            </div>
            <h1 className="font-mono font-bold tracking-tight text-lg">Clickbait Detection</h1>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Database className="w-3.5 h-3.5" />
              <span>{mlStatus?.modelName || "LOADING_MODEL"}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative flex h-2 w-2">
                {modelReady ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500" />
                )}
              </div>
              <span className={modelReady ? "text-emerald-500" : "text-amber-500"}>
                {modelReady ? "SYS_READY" : "SYS_WARMUP"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* LEFT — Input + Results */}
        <div className="lg:col-span-8 space-y-6">

          <Card className="border-border/50 bg-card/30 shadow-none">
            <CardHeader>
              <CardTitle className="font-mono text-xl flex items-center gap-2">
                <Search className="w-5 h-5 text-muted-foreground" />
                NEWS_ANALYSIS
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleAnalyze} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="headline" className="font-mono text-xs text-muted-foreground">
                    TARGET_HEADLINE
                  </Label>
                  <Input
                    id="headline"
                    value={headline}
                    onChange={(e) => setHeadline(e.target.value)}
                    placeholder="Enter article headline..."
                    className="font-mono text-sm bg-background border-border/50 focus-visible:ring-primary focus-visible:border-primary h-12"
                    data-testid="input-headline"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="body" className="font-mono text-xs text-muted-foreground">
                    TARGET_BODY_CONTENT (OPTIONAL)
                  </Label>
                  <Textarea
                    id="body"
                    value={body}
                    onChange={(e) => setBody(e.target.value)}
                    placeholder="Paste full article text for higher precision..."
                    className="font-mono text-sm bg-background border-border/50 focus-visible:ring-primary min-h-[120px] resize-y"
                    data-testid="input-body"
                  />
                </div>
                <div className="flex gap-3 justify-end pt-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleReset}
                    disabled={isAnalyzing}
                    className="font-mono w-full md:w-auto"
                    data-testid="button-reset"
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    CLEAR
                  </Button>
                  <Button
                    type="submit"
                    disabled={!headline.trim() || isAnalyzing || !modelReady}
                    className="font-mono font-bold w-full md:w-auto min-w-[200px]"
                    data-testid="button-analyze"
                  >
                    {isAnalyzing ? (
                      <span className="flex items-center gap-2">
                        <Activity className="w-4 h-4 animate-spin" />
                        PROCESSING...
                      </span>
                    ) : !modelReady ? (
                      "MODEL LOADING..."
                    ) : (
                      <span className="flex items-center gap-2">
                        <Zap className="w-4 h-4" />
                        EXECUTE_SCAN
                      </span>
                    )}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          {/* Result */}
          {result && cfg && verdict && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <Card className="border-border/50 overflow-hidden relative">
                {/* Accent bar */}
                <div className={`absolute top-0 left-0 w-1 h-full ${cfg.bg}`} />

                <CardHeader className="pb-4 pl-6">
                  <div className="flex items-center justify-between mb-4">
                    <CardTitle className="font-mono text-sm text-muted-foreground">
                      ANALYSIS_REPORT_#{result.id}
                    </CardTitle>
                    <div className="text-xs font-mono text-muted-foreground">
                      {format(new Date(result.analyzedAt), "HH:mm:ss.SSS")}
                    </div>
                  </div>

                  <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div className="space-y-1">
                      <div className="text-xs font-mono text-muted-foreground mb-2">FINAL_VERDICT</div>
                      <Badge
                        variant="outline"
                        className={`text-2xl py-1 px-4 font-mono font-bold border-2 ${cfg.color} ${cfg.border} bg-transparent`}
                        data-testid={`status-verdict-${verdict}`}
                      >
                        {verdict}
                      </Badge>
                    </div>

                    <div className="flex-1 max-w-sm space-y-2">
                      <div className="flex justify-between font-mono text-xs">
                        <span className="text-muted-foreground">CONFIDENCE_SCORE</span>
                        <span className={`font-bold ${cfg.color}`}>
                          {result.confidence.toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                        <div
                          className={`h-full ${cfg.bg} transition-all duration-700`}
                          style={{ width: `${result.confidence}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-8 pl-6">
                  {/* Headline recap */}
                  <div className="p-4 bg-background/50 rounded border border-border/50 space-y-3">
                    <div>
                      <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase">Headline</div>
                      <div className="font-medium">{result.headline}</div>
                    </div>
                    {result.body && (
                      <div>
                        <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase">Body Snippet</div>
                        <div className="text-sm text-muted-foreground line-clamp-3 font-mono">{result.body}</div>
                      </div>
                    )}
                  </div>

                  {/* Probability matrix */}
                  <div className="space-y-4">
                    <h4 className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">
                      PROBABILITY_MATRIX
                    </h4>
                    <div className="grid gap-3">
                      {(["REAL", "CLICKBAIT"] as const).map((type) => {
                        const score = result.scores[type.toLowerCase() as keyof typeof result.scores] as number;
                        const c = verdictConfig[type];
                        return (
                          <div key={type} className="flex items-center gap-4">
                            <div className={`w-24 text-xs font-mono ${c.color}`}>{type}</div>
                            <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                              <div className={`h-full ${c.bg}`} style={{ width: `${score}%` }} />
                            </div>
                            <div className="w-12 text-right text-xs font-mono text-muted-foreground">
                              {score.toFixed(1)}%
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Trigger keywords — only for CLICKBAIT */}
                  {verdict === "CLICKBAIT" && result.keywords && result.keywords.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">
                        TRIGGER_KEYWORDS
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {result.keywords.map((kw, i) => (
                          <span
                            key={i}
                            className={`inline-flex items-center gap-1.5 pl-2.5 pr-1.5 py-1 rounded text-xs font-mono font-semibold border ${cfg.chip}`}
                          >
                            {kw.word}
                            <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold ${cfg.badge}`}>
                              {kw.score}%
                            </span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Detected signals */}
                  {result.indicators && result.indicators.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">
                        DETECTED_SIGNALS
                      </h4>
                      <ul className="grid gap-2">
                        {result.indicators.map((indicator, i) => (
                          <li
                            key={i}
                            className="flex items-start gap-2 text-sm font-mono bg-background p-2 rounded border border-border/30"
                          >
                            <ChevronRight className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                            <span className="text-muted-foreground">{indicator}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Source evidence */}
                  {result.sources && result.sources.length > 0 && (
                    <div className="space-y-3">
                      <button
                        type="button"
                        onClick={() => setSourcesExpanded((v) => !v)}
                        className="flex items-center justify-between w-full font-mono text-xs font-semibold tracking-wider text-muted-foreground hover:text-foreground transition-colors"
                        data-testid="button-toggle-sources"
                      >
                        <span>SOURCE_EVIDENCE ({result.sources.length})</span>
                        {sourcesExpanded ? (
                          <ChevronDown className="w-4 h-4" />
                        ) : (
                          <ChevronRight className="w-4 h-4" />
                        )}
                      </button>
                      {sourcesExpanded && (
                        <ul className="grid gap-2">
                          {result.sources.map((source, i) => (
                            <li
                              key={i}
                              className="text-sm font-mono bg-background p-3 rounded border border-border/30 space-y-1"
                            >
                              <span className="block text-[10px] uppercase tracking-wider text-primary">
                                {source.label}
                              </span>
                              {source.url ? (
                                <a
                                  href={source.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex items-start gap-1.5 text-foreground hover:text-primary hover:underline"
                                >
                                  <span>{source.title}</span>
                                  <ExternalLink className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                                </a>
                              ) : (
                                <span className="text-foreground">{source.title}</span>
                              )}
                              {source.publisher && (
                                <span className="block text-xs text-muted-foreground">
                                  — {source.publisher}
                                </span>
                              )}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Feedback */}
              {!feedbackSubmitted && (
                <Card className="border-primary/20 bg-primary/5">
                  <CardContent className="p-6 flex flex-col sm:flex-row items-center justify-between gap-6">
                    <div className="space-y-1 text-center sm:text-left">
                      <h3 className="font-mono font-semibold">
                        {verdict === "UNCERTAIN" ? "Do you know the real answer?" : "Was this correct?"}
                      </h3>
                      <p className="text-sm text-muted-foreground font-mono">
                        {verdict === "UNCERTAIN"
                          ? "We couldn't verify this confidently. If you know the answer, tell us."
                          : "Tell us the actual answer to help improve future results."}
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-2 justify-center">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleFeedback("REAL")}
                        disabled={submitFeedback.isPending}
                        className="font-mono text-emerald-500 border-emerald-500/30 hover:bg-emerald-500/10"
                        data-testid="button-feedback-real"
                      >
                        Real
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleFeedback("CLICKBAIT")}
                        disabled={submitFeedback.isPending}
                        className="font-mono text-amber-500 border-amber-500/30 hover:bg-amber-500/10"
                        data-testid="button-feedback-clickbait"
                      >
                        Clickbait
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>

        {/* RIGHT — Telemetry */}
        <div className="lg:col-span-4 space-y-6">
          <Card className="border-border/50 bg-card/30">
            <CardHeader className="pb-4">
              <CardTitle className="font-mono text-sm flex items-center gap-2 text-muted-foreground">
                <Database className="w-4 h-4" />
                SYSTEM_TELEMETRY
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {statsLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-24 w-full" />
                </div>
              ) : stats ? (
                <>
                  <div className="space-y-1">
                    <div className="text-[10px] font-mono text-muted-foreground uppercase">Avg Confidence</div>
                    <div className="text-2xl font-mono text-primary" data-testid="stat-confidence">
                      {stats.avgConfidence.toFixed(1)}%
                    </div>
                  </div>

                  {totalScanned > 0 && (
                    <div className="space-y-2">
                      <div className="text-[10px] font-mono text-muted-foreground uppercase">Distribution</div>
                      <div className="h-2 w-full flex rounded-full overflow-hidden bg-secondary">
                        <div
                          className="bg-emerald-500 h-full transition-all duration-500"
                          style={{ width: `${(stats.verdictCounts.REAL / totalScanned) * 100}%` }}
                        />
                        <div
                          className="bg-sky-400 h-full transition-all duration-500"
                          style={{ width: `${(stats.verdictCounts.UNCERTAIN / totalScanned) * 100}%` }}
                        />
                        <div
                          className="bg-amber-500 h-full transition-all duration-500"
                          style={{ width: `${(stats.verdictCounts.CLICKBAIT / totalScanned) * 100}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-[10px] font-mono mt-1">
                        <span className="text-emerald-500">{stats.verdictCounts.REAL} REAL</span>
                        <span className="text-sky-400">{stats.verdictCounts.UNCERTAIN} UNCERTAIN</span>
                        <span className="text-amber-500">{stats.verdictCounts.CLICKBAIT} CLICKBAIT</span>
                      </div>
                    </div>
                  )}
                </>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
