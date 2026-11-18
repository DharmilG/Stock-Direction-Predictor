import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ImagePlus, Upload, X } from 'lucide-react';

import { cvApi } from '../lib/cvApi';
import { useResource } from '../hooks/useResource';
import { POLL_INTERVALS } from '../lib/constants';
import { ConfusionMatrix, LineSeriesChart } from '../components/charts';
import { SignalBadge } from '../components/Signal';
import {
  Badge,
  Button,
  Callout,
  Empty,
  ErrorNote,
  KeyValue,
  Loading,
  PageHeader,
  Panel,
  ProgressBar,
  Section,
  SourceTag,
  Stat,
  StatGrid,
  Table,
  Warnings,
} from '../components/ui';

const ALLOWED = ['image/png', 'image/jpeg', 'image/webp'];
const MAX_BYTES = 5 * 1024 * 1024;
const SIGNAL_LABEL = { neutral: 'FLAT', bullish: 'UP', bearish: 'DOWN' };
const pct = (value) => (value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`);

export default function Cv() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const controllerRef = useRef(null);
  const inputRef = useRef(null);

  const healthResource = useResource(cvApi.health, { key: 'cv-health', pollMs: POLL_INTERVALS.cv });
  const training = useResource(cvApi.training, { key: 'cv-training', pollMs: POLL_INTERVALS.cv });
  const evaluation = useResource(cvApi.evaluation, { key: 'cv-evaluation', pollMs: POLL_INTERVALS.cv });

  const health = healthResource.envelope;
  const ready = Boolean(health?.data?.ready);

  useEffect(() => () => controllerRef.current?.abort(), []);
  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const chooseFile = useCallback(
    (candidate) => {
      if (!candidate) return;
      if (!ALLOWED.includes(candidate.type)) {
        setError(new Error('Use a PNG, JPEG or WEBP chart image.'));
        return;
      }
      if (candidate.size > MAX_BYTES) {
        setError(new Error('Image exceeds the 5 MB limit.'));
        return;
      }
      if (preview) URL.revokeObjectURL(preview);
      setFile(candidate);
      setPreview(URL.createObjectURL(candidate));
      setResult(null);
      setError(null);
    },
    [preview],
  );

  const analyze = useCallback(async () => {
    if (!file || busy) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy(true);
    setError(null);
    try {
      const envelope = await cvApi.analyze(file, { signal: controller.signal });
      setResult(envelope.data);
    } catch (err) {
      if (err?.name !== 'AbortError') setError(err);
    } finally {
      setBusy(false);
    }
  }, [file, busy]);

  const stages = useMemo(() => training.data?.stages || {}, [training.data]);
  const stage1 = useMemo(() => stages.train_stage1 || {}, [stages]);
  const stage2 = useMemo(() => stages.train_stage2 || {}, [stages]);
  const history2 = useMemo(() => stage2.history || [], [stage2]);
  const latest2 = history2[history2.length - 1] || null;

  const curves = useMemo(
    () =>
      history2.map((row) => ({
        epoch: row.epoch,
        train_loss: row.train?.loss,
        val_loss: row.val?.loss,
        train_acc: row.train?.direction_acc,
        val_acc: row.val?.direction_acc,
      })),
    [history2],
  );

  const perClassRows = useMemo(
    () =>
      Object.entries(latest2?.val?.direction_metrics?.per_class || {}).map(([name, metrics]) => ({
        name: SIGNAL_LABEL[name] || name,
        ...metrics,
      })),
    [latest2],
  );

  const scores = result
    ? Object.entries(result.pattern_scores || {}).map(([label, value]) => ({ label, value }))
    : [];
  const directionScores = result
    ? Object.entries(result.direction_probs || {}).map(([label, value]) => ({ label, value }))
    : [];

  const evalData = evaluation.data || {};
  const evalLabels = (evalData.labels || []).map((label) => SIGNAL_LABEL[label] || label);
  const evalReady = Boolean(evalData.ready);

  return (
    <>
      <PageHeader
        eyebrow="Computer vision"
        title="Chart Vision"
        description="Upload a candlestick chart image and read its pattern and price direction. An isolated research signal — it never feeds the tabular model."
        actions={<SourceTag envelope={health} prefix="CV status" />}
      />

      <Callout tone="warn" title="Research signal, not a trading edge">
        Synthetic pattern accuracy is inflated by construction and real direction accuracy is close to the
        3-class random baseline. Use this to inspect what the model sees, not to place trades.
      </Callout>

      {healthResource.error && (
        <div className="mb-5">
          <ErrorNote error={healthResource.error} label="CV health check failed" onRetry={healthResource.refresh} />
        </div>
      )}

      {health && !ready && (
        <div className="mb-5">
          <Callout tone="warn" title="CV model is not trained yet">
            Stage progress is shown below. Once stage 2 finishes, this page updates automatically and the upload
            form becomes active.
          </Callout>
        </div>
      )}

      <Section eyebrow="Training" title="Training run and output" note="Read-only · refreshes every 5 seconds">
        {training.error && (
          <div className="mb-5">
            <ErrorNote error={training.error} label="Training status unavailable" onRetry={training.refresh} />
          </div>
        )}
        <StatGrid columns={4}>
          <Stat label="Stage 1 status" value={stage1.status || 'not-started'} mono={false} />
          <Stat label="Stage 1 best pattern acc" value={pct(stage1.best_metric)} />
          <Stat label="Stage 2 status" value={stage2.status || 'not-started'} mono={false} />
          <Stat label="Stage 2 best direction acc" value={pct(stage2.best_metric)} />
        </StatGrid>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Panel title="Direction accuracy" eyebrow="Stage 2 · per epoch" info={{ what: 'Train and validation accuracy of the direction head after each epoch.' }}>
            {curves.length ? (
              <LineSeriesChart
                data={curves}
                xKey="epoch"
                series={[
                  { key: 'train_acc', label: 'Train' },
                  { key: 'val_acc', label: 'Validation' },
                ]}
                height={240}
                yDomain={[0, 1]}
                yTickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
            ) : (
              <Empty title="No epochs yet" detail="Stage 2 has not written a per-epoch metric." />
            )}
          </Panel>

          <Panel title="Loss" eyebrow="Stage 2 · per epoch" info={{ what: 'Class-weighted cross-entropy on train and validation folds.' }}>
            {curves.length ? (
              <LineSeriesChart
                data={curves}
                xKey="epoch"
                series={[
                  { key: 'train_loss', label: 'Train' },
                  { key: 'val_loss', label: 'Validation' },
                ]}
                height={240}
                yTickFormatter={(value) => Number(value).toFixed(2)}
              />
            ) : (
              <Empty title="No epochs yet" detail="Stage 2 has not written a per-epoch metric." />
            )}
          </Panel>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Panel title="Latest epoch" eyebrow="Stage 2 · validation">
            {latest2 ? (
              <KeyValue
                columns={2}
                items={[
                  { label: 'Epoch', value: latest2.epoch },
                  { label: 'Learning rate', value: latest2.learning_rate },
                  { label: 'Epoch seconds', value: latest2.epoch_seconds },
                  { label: 'Images / second', value: latest2.images_per_second },
                  { label: 'Macro F1', value: pct(latest2.val?.direction_metrics?.f1_macro) },
                  { label: 'AUC (one-vs-rest)', value: pct(latest2.val?.direction_metrics?.auc_ovr) },
                  { label: 'CPU %', value: latest2.cpu_percent },
                  { label: 'RAM available (MB)', value: latest2.ram_available_mb },
                ]}
              />
            ) : (
              <Empty title="No epoch data" detail="Metrics appear here after the first stage-2 epoch." />
            )}
          </Panel>

          <Panel title="Per-class validation quality" eyebrow="Precision / recall / F1">
            <Table
              columns={[
                { key: 'name', header: 'Class', strong: true },
                { key: 'precision', header: 'Precision', align: 'right', render: (row) => pct(row.precision) },
                { key: 'recall', header: 'Recall', align: 'right', render: (row) => pct(row.recall) },
                { key: 'f1', header: 'F1', align: 'right', render: (row) => pct(row.f1) },
                { key: 'support', header: 'Support', align: 'right', mono: true },
              ]}
              rows={perClassRows}
              rowKey={(row) => row.name}
              empty={<Empty title="No per-class data yet" detail="Available after the first stage-2 epoch." />}
            />
          </Panel>
        </div>
      </Section>

      <Section eyebrow="Evaluation" title="Held-out evaluation" note="Written by the evaluate command">
        {evaluation.error && (
          <div className="mb-5">
            <ErrorNote error={evaluation.error} label="Evaluation unavailable" onRetry={evaluation.refresh} />
          </div>
        )}
        {!evalReady ? (
          <Panel title="Not evaluated yet">
            <Empty title="No evaluation artifacts" detail="Run the evaluate command after training to see the confusion matrix and report." />
          </Panel>
        ) : (
          <>
            <StatGrid columns={3}>
              <Stat label="Accuracy" value={pct(evalData.accuracy)} />
              <Stat label="AUC (one-vs-rest)" value={pct(evalData.auc_ovr)} />
              <Stat label="Dataset" value={evalData.dataset} mono={false} />
            </StatGrid>
            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <Panel title="Confusion matrix" eyebrow="Predicted vs actual">
                <ConfusionMatrix matrix={evalData.matrix} labels={evalLabels} />
              </Panel>
              <Panel title="Classification report" eyebrow="Per-class detail">
                <pre className="mono overflow-x-auto text-[.75rem] leading-relaxed text-[var(--ink-2)]">
                  {evalData.classification_report}
                </pre>
              </Panel>
            </div>
          </>
        )}
      </Section>

      <Section eyebrow="Input" title="Upload a chart image" note="PNG, JPEG or WEBP · up to 5 MB">
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Chart image" eyebrow="Step 1">
            <div
              className={`flex min-h-[14rem] flex-col items-center justify-center gap-3 border border-dashed px-4 py-8 text-center ${
                dragging ? 'border-[var(--ink)] bg-[var(--surface-sunken)]' : 'border-[var(--line-strong)]'
              }`}
              onDragOver={(event) => event.preventDefault()}
              onDragEnter={() => setDragging(true)}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                chooseFile(event.dataTransfer.files?.[0]);
              }}
            >
              {preview ? (
                <div className="relative w-full max-w-sm">
                  <img src={preview} alt="Uploaded chart preview" className="w-full border border-[var(--line)]" />
                  <button
                    type="button"
                    className="absolute right-1 top-1 rounded border border-[var(--line)] bg-[var(--surface)] p-1"
                    aria-label="Remove image"
                    onClick={() => {
                      if (preview) URL.revokeObjectURL(preview);
                      setFile(null);
                      setPreview('');
                      setResult(null);
                      if (inputRef.current) inputRef.current.value = '';
                    }}
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <ImagePlus size={28} strokeWidth={1.5} className="text-[var(--ink-3)]" />
              )}
              <input
                ref={inputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                className="sr-only"
                onChange={(event) => chooseFile(event.target.files?.[0])}
              />
              <Button variant="secondary" icon={Upload} size="sm" onClick={() => inputRef.current?.click()}>
                {preview ? 'Choose another image' : 'Choose image'}
              </Button>
              <p className="mono text-[.8125rem] text-[var(--ink-3)]">
                {file ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB` : 'Or drop an image here'}
              </p>
            </div>
            <div className="mt-4 flex items-center gap-2">
              <Button variant="primary" onClick={analyze} disabled={!file || busy || !ready}>
                {busy ? 'Analyzing…' : 'Analyze chart'}
              </Button>
              {!ready && <Badge tone="badge-dashed">model not built</Badge>}
            </div>
          </Panel>

          <Panel title="What the model returns" eyebrow="Step 2" info={{ what: 'One forward pass produces a chart-pattern label, a price-direction label and a GradCAM attention map.' }}>
            <div className="space-y-3">
              <p className="text-[.875rem] leading-relaxed text-[var(--ink-2)]">
                The image is resized to 224×224, passed through a frozen ImageNet ResNet18 backbone, then through two
                small heads. Direction is mapped from bullish / bearish / neutral to UP / DOWN / FLAT.
              </p>
              <KeyValue
                columns={1}
                items={[
                  { label: 'Pattern classes', value: (health?.data?.pattern_classes || []).join(', ') || '—', mono: false },
                  { label: 'Signal classes', value: (health?.data?.signal_classes || []).join(' / ') || '—', mono: false },
                ]}
              />
              {busy && <Loading rows={4} label="Running inference" />}
              {error && <ErrorNote error={error} label="Analysis failed" onRetry={analyze} />}
              {!result && !busy && !error && (
                <Empty title="No analysis yet" detail="Upload a chart and click Analyze to see the prediction." />
              )}
            </div>
          </Panel>
        </div>
      </Section>

      {result && (
        <Section eyebrow="Output" title="Prediction" note="Independent research signal">
          <StatGrid columns={4}>
            <Stat label="Direction" value={<SignalBadge signal={result.signal} size="lg" />} mono={false} />
            <Stat label="Direction confidence" value={pct(result.signal_confidence)} />
            <Stat label="Detected pattern" value={result.pattern.replaceAll('_', ' ')} mono={false} />
            <Stat label="Pattern confidence" value={pct(result.pattern_confidence)} />
          </StatGrid>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Panel title="Pattern scores" eyebrow="6 classes">
              <div className="space-y-2">
                {scores.map((entry) => (
                  <ProgressBar
                    key={entry.label}
                    label={entry.label.replaceAll('_', ' ')}
                    value={Math.round(entry.value * 100)}
                    detail={pct(entry.value)}
                  />
                ))}
              </div>
              <div className="mt-4 border-t border-[var(--line)] pt-4">
                {directionScores.map((entry) => (
                  <div key={entry.label} className="flex items-center justify-between gap-2 py-1">
                    <SignalBadge signal={entry.label} size="sm" />
                    <span className="mono text-[.8125rem] text-[var(--ink-2)]">{pct(entry.value)}</span>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="GradCAM — what the model looked at" eyebrow="Attention heatmap">
              {result.gradcam_base64 ? (
                <div className="grid gap-3 sm:grid-cols-2">
                  <figure>
                    <img src={preview} alt="Original chart" className="w-full border border-[var(--line)]" />
                    <figcaption className="mt-1 text-[.75rem] text-[var(--ink-3)]">Original chart</figcaption>
                  </figure>
                  <figure>
                    <img
                      src={`data:image/png;base64,${result.gradcam_base64}`}
                      alt="GradCAM heatmap"
                      className="w-full border border-[var(--line)]"
                    />
                    <figcaption className="mt-1 text-[.75rem] text-[var(--ink-3)]">Attention overlay</figcaption>
                  </figure>
                </div>
              ) : (
                <Empty title="Heatmap unavailable" detail="The prediction succeeded but GradCAM could not be rendered." />
              )}
              {result.warnings?.length > 0 && <div className="mt-4"><Warnings warnings={result.warnings} /></div>}
            </Panel>
          </div>
        </Section>
      )}
    </>
  );
}
