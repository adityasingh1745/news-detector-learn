import React, { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useAnalyzeNews,
  useSubmitFeedback,
  useGetHistory,
  useGetStats,
  useGetMlStatus,
  getGetHistoryQueryKey,
  getGetStatsQueryKey,
  getGetMlStatusQueryKey,
} from "@workspace/api-client-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Activity, AlertTriangle, CheckCircle2, History, Database, Shield, Zap, Search, ChevronRight, RotateCcw } from "lucide-react";
import { format } from "date-fns";

type VerdictType = "REAL" | "CLICKBAIT" | "FAKE";

const VerdictIcon = ({ verdict, className = "w-4 h-4" }: { verdict: VerdictType; className?: string }) => {
  if (verdict === "REAL") return <CheckCircle2 className={`text-emerald-500 ${className}`} />;
  if (verdict === "CLICKBAIT") return <AlertTriangle className={`text-amber-500 ${className}`} />;
  return <Shield className={`text-rose-500 ${className}`} />;
};

const getVerdictClasses = (verdict: VerdictType) => {
  if (verdict === "REAL") return "verdict-real";
  if (verdict === "CLICKBAIT") return "verdict-clickbait";
  return "verdict-fake";
};

const getVerdictColor = (verdict: VerdictType) => {
  if (verdict === "REAL") return "text-emerald-500";
  if (verdict === "CLICKBAIT") return "text-amber-500";
  return "text-rose-500";
};

export default function Home() {
  const queryClient = useQueryClient();
  const [headline, setHeadline] = useState("");
  const [body, setBody] = useState("");
  const [currentResultId, setCurrentResultId] = useState<number | null>(null);

  // Queries
  const { data: mlStatus } = useGetMlStatus({
    query: {
      queryKey: getGetMlStatusQueryKey(),
      refetchInterval: (query) => {
        return query.state.data?.ready ? false : 5000;
      },
    },
  });

  const { data: history, isLoading: historyLoading } = useGetHistory(
    { limit: 10 },
    { query: { queryKey: getGetHistoryQueryKey({ limit: 10 }) } }
  );

  const { data: stats, isLoading: statsLoading } = useGetStats({
    query: { queryKey: getGetStatsQueryKey() },
  });

  // Mutations
  const analyzeNews = useAnalyzeNews({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: getGetHistoryQueryKey() });
        queryClient.invalidateQueries({ queryKey: getGetStatsQueryKey() });
      },
    },
  });

  const submitFeedback = useSubmitFeedback({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: getGetHistoryQueryKey() });
        queryClient.invalidateQueries({ queryKey: getGetStatsQueryKey() });
        queryClient.invalidateQueries({ queryKey: getGetMlStatusQueryKey() });
      },
    },
  });

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    if (!headline.trim()) return;

    analyzeNews.mutate(
      { data: { headline, body: body.trim() || undefined } },
      {
        onSuccess: (data) => {
          setCurrentResultId(data.id);
        },
      }
    );
  };

  const handleFeedback = (verdict: VerdictType) => {
    if (!currentResultId) return;
    submitFeedback.mutate({
      data: { analysisId: currentResultId, correctLabel: verdict },
    });
  };

  const handleReset = () => {
    setHeadline("");
    setBody("");
    setCurrentResultId(null);
    analyzeNews.reset();
  };

  const currentResult = history?.find((h) => h.id === currentResultId);
  const currentResultFullData = analyzeNews.data?.id === currentResultId ? analyzeNews.data : null;

  const isAnalyzing = analyzeNews.isPending;
  const modelReady = mlStatus?.ready ?? false;

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary selection:text-primary-foreground pb-20">
      {/* Top Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-sm bg-primary flex items-center justify-center text-primary-foreground font-mono font-bold text-lg">
              {"//"}
            </div>
            <h1 className="font-mono font-bold tracking-tight text-lg">EVAL.OSINT</h1>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2 text-muted-foreground" data-testid="status-model-name">
              <Database className="w-3.5 h-3.5" />
              <span>{mlStatus?.modelName || "LOADING_MODEL"}</span>
            </div>
            <div className="flex items-center gap-2" data-testid="status-model-ready">
              <div className="relative flex h-2 w-2">
                {modelReady ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
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
        
        {/* LEFT COLUMN: Input & Results */}
        <div className="lg:col-span-8 space-y-6">
          
          <Card className="border-border/50 bg-card/30 shadow-none">
            <CardHeader>
              <CardTitle className="font-mono text-xl flex items-center gap-2">
                <Search className="w-5 h-5 text-muted-foreground" />
                NEW_ANALYSIS
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleAnalyze} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="headline" className="font-mono text-xs text-muted-foreground">TARGET_HEADLINE</Label>
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
                  <Label htmlFor="body" className="font-mono text-xs text-muted-foreground">TARGET_BODY_CONTENT (OPTIONAL)</Label>
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
                  {(headline || body || currentResultFullData) && (
                    <Button
                      type="button"
                      variant="outline"
                      onClick={handleReset}
                      disabled={isAnalyzing}
                      className="font-mono w-full md:w-auto"
                      data-testid="button-reset"
                    >
                      <RotateCcw className="w-4 h-4 mr-2" />
                      RESET
                    </Button>
                  )}
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

          {/* Result Display */}
          {currentResultFullData && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <Card className="border-border/50 overflow-hidden relative">
                <div className={`absolute top-0 left-0 w-1 h-full ${getVerdictClasses(currentResultFullData.verdict).split(' ')[1]}`} />
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between mb-4">
                    <CardTitle className="font-mono text-sm text-muted-foreground">ANALYSIS_REPORT_#{currentResultFullData.id}</CardTitle>
                    <div className="text-xs font-mono text-muted-foreground">
                      {format(new Date(currentResultFullData.analyzedAt), "HH:mm:ss.SSS")}
                    </div>
                  </div>
                  
                  <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div className="space-y-1">
                      <div className="text-xs font-mono text-muted-foreground mb-2">FINAL_VERDICT</div>
                      <Badge variant="outline" className={`text-2xl py-1 px-4 font-mono font-bold border-2 ${getVerdictClasses(currentResultFullData.verdict)}`} data-testid={`status-verdict-${currentResultFullData.verdict}`}>
                        {currentResultFullData.verdict}
                      </Badge>
                    </div>
                    
                    <div className="flex-1 max-w-sm space-y-2">
                      <div className="flex justify-between font-mono text-xs">
                        <span className="text-muted-foreground">CONFIDENCE_SCORE</span>
                        <span className={`font-bold ${getVerdictColor(currentResultFullData.verdict)}`}>
                          {(currentResultFullData.confidence).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${getVerdictClasses(currentResultFullData.verdict).split(' ')[0].replace('text-', 'bg-')}`} 
                          style={{ width: `${currentResultFullData.confidence}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-8">
                  {/* Target Content Recap */}
                  <div className="p-4 bg-background/50 rounded border border-border/50 space-y-3">
                    <div>
                      <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase">Headline</div>
                      <div className="font-medium">{currentResultFullData.headline}</div>
                    </div>
                    {currentResultFullData.body && (
                      <div>
                        <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase">Body Snippet</div>
                        <div className="text-sm text-muted-foreground line-clamp-3 font-mono">{currentResultFullData.body}</div>
                      </div>
                    )}
                  </div>

                  {/* Score Breakdown */}
                  <div className="space-y-4">
                    <h4 className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">PROBABILITY_MATRIX</h4>
                    <div className="grid gap-3">
                      {(["REAL", "CLICKBAIT", "FAKE"] as const).map(type => {
                        const score = currentResultFullData.scores[type.toLowerCase() as keyof typeof currentResultFullData.scores];
                        return (
                          <div key={type} className="flex items-center gap-4">
                            <div className={`w-24 text-xs font-mono ${getVerdictColor(type)}`}>{type}</div>
                            <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${type === 'REAL' ? 'bg-emerald-500' : type === 'CLICKBAIT' ? 'bg-amber-500' : 'bg-rose-500'}`} 
                                style={{ width: `${score}%` }} 
                              />
                            </div>
                            <div className="w-12 text-right text-xs font-mono text-muted-foreground">{score.toFixed(1)}%</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Trigger Keywords — only for FAKE or CLICKBAIT */}
                  {currentResultFullData.verdict !== "REAL" &&
                    currentResultFullData.keywords &&
                    currentResultFullData.keywords.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">
                        TRIGGER_KEYWORDS
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {currentResultFullData.keywords.map((kw, i) => (
                          <span
                            key={i}
                            className={`inline-flex items-center px-2.5 py-1 rounded text-xs font-mono font-semibold border ${
                              currentResultFullData.verdict === "CLICKBAIT"
                                ? "bg-amber-500/10 border-amber-500/40 text-amber-400"
                                : "bg-rose-500/10 border-rose-500/40 text-rose-400"
                            }`}
                          >
                            {kw}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Indicators */}
                  {currentResultFullData.indicators && currentResultFullData.indicators.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">DETECTED_SIGNALS</h4>
                      <ul className="grid gap-2">
                        {currentResultFullData.indicators.map((indicator, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm font-mono bg-background p-2 rounded border border-border/30">
                            <ChevronRight className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                            <span className="text-muted-foreground">{indicator}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Feedback Section */}
              {(!currentResult || !currentResult.hasFeedback) && (
                <Card className="border-primary/20 bg-primary/5">
                  <CardContent className="p-6 flex flex-col sm:flex-row items-center justify-between gap-6">
                    <div className="space-y-1 text-center sm:text-left">
                      <h3 className="font-mono font-semibold">Verify Result</h3>
                      <p className="text-sm text-muted-foreground font-mono">
                        Help calibrate {currentResultFullData.modelUsed}. Your feedback trains the model.
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
                        TRUE: REAL
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleFeedback("CLICKBAIT")}
                        disabled={submitFeedback.isPending}
                        className="font-mono text-amber-500 border-amber-500/30 hover:bg-amber-500/10"
                        data-testid="button-feedback-clickbait"
                      >
                        TRUE: CLICKBAIT
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleFeedback("FAKE")}
                        disabled={submitFeedback.isPending}
                        className="font-mono text-rose-500 border-rose-500/30 hover:bg-rose-500/10"
                        data-testid="button-feedback-fake"
                      >
                        TRUE: FAKE
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: History & Stats */}
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
                    <div className="text-2xl font-mono text-primary" data-testid="stat-confidence">{stats.avgConfidence.toFixed(1)}%</div>
                  </div>

                  <div className="space-y-2">
                    <div className="text-[10px] font-mono text-muted-foreground uppercase">Distribution</div>
                    <div className="h-2 w-full flex rounded-full overflow-hidden bg-secondary">
                      <div className="bg-emerald-500 h-full" style={{ width: `${(stats.verdictCounts.REAL / stats.totalAnalyzed) * 100}%` }} />
                      <div className="bg-amber-500 h-full" style={{ width: `${(stats.verdictCounts.CLICKBAIT / stats.totalAnalyzed) * 100}%` }} />
                      <div className="bg-rose-500 h-full" style={{ width: `${(stats.verdictCounts.FAKE / stats.totalAnalyzed) * 100}%` }} />
                    </div>
                    <div className="flex justify-between text-[10px] font-mono mt-1">
                      <span className="text-emerald-500">{stats.verdictCounts.REAL} R</span>
                      <span className="text-amber-500">{stats.verdictCounts.CLICKBAIT} C</span>
                      <span className="text-rose-500">{stats.verdictCounts.FAKE} F</span>
                    </div>
                  </div>

                  <div className="pt-4 border-t border-border/50 grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <div className="text-[10px] font-mono text-muted-foreground uppercase">Training Samples</div>
                      <div className="text-lg font-mono text-muted-foreground" data-testid="stat-feedback">{stats.totalFeedback.toLocaleString()}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-[10px] font-mono text-muted-foreground uppercase">Calibrated Acc</div>
                      <div className="text-lg font-mono text-muted-foreground" data-testid="stat-accuracy">
                        {stats.accuracyFromFeedback ? `${stats.accuracyFromFeedback.toFixed(1)}%` : "N/A"}
                      </div>
                    </div>
                  </div>
                </>
              ) : null}
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/30">
            <CardHeader className="pb-4">
              <CardTitle className="font-mono text-sm flex items-center gap-2 text-muted-foreground">
                <History className="w-4 h-4" />
                RECENT_SCANS
              </CardTitle>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="space-y-3">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : history && history.length > 0 ? (
                <div className="space-y-3">
                  {history.map((item) => (
                    <div 
                      key={item.id} 
                      className={`p-3 rounded border border-border/30 bg-background/50 flex flex-col gap-2 transition-colors cursor-pointer hover:border-primary/50 ${currentResultId === item.id ? 'ring-1 ring-primary border-primary' : ''}`}
                      onClick={() => setCurrentResultId(item.id)}
                      data-testid={`history-item-${item.id}`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <Badge variant="outline" className={`text-[10px] py-0 px-2 font-mono border ${getVerdictClasses(item.verdict)}`}>
                          {item.verdict}
                        </Badge>
                        <span className="text-[10px] font-mono text-muted-foreground">
                          {format(new Date(item.analyzedAt), "HH:mm")}
                        </span>
                      </div>
                      <div className="text-sm font-medium line-clamp-2 leading-snug">
                        {item.headline}
                      </div>
                      <div className="flex items-center gap-2 text-xs font-mono">
                        <span className={getVerdictColor(item.verdict)}>
                          {item.confidence.toFixed(1)}% CONF
                        </span>
                        {item.hasFeedback && (
                          <span className="text-muted-foreground flex items-center gap-1">
                            <span className="w-1 h-1 rounded-full bg-primary" /> VERIFIED
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-sm font-mono text-muted-foreground">
                  NO_DATA_FOUND
                </div>
              )}
            </CardContent>
          </Card>

        </div>
      </main>
    </div>
  );
}
