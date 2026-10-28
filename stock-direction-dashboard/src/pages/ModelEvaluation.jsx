import { useMemo, useState } from 'react';
import { Braces, RefreshCw } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate, formatInstant } from '../lib/dates';
import { int, num, pct, signed } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { CalibrationChart, ConfusionMatrix, ImportanceBars } from '../components/charts';
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
  Segmented,
  SourceTag,
  Stat,
  StatGrid,
  Table,
  Warnings,
} from '../components/ui';

const CLASS_LABELS = ['DOWN', 'FLAT', 'UP'];
const SHAP_LIMITS = [
  { value: 15, label: '15' },
  { value: 25, label: '25' },
  { value: 40, label: '40' },
];

function JsonBlock({ value, label }) {
  if (!value) return <Empty title={`${label} not recorded`} />;
  return (
    <pre className="mono max-h-64 overflow-auto rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] p-3 text-[.6875rem] leading-relaxed text-[var(--ink-2)] scroll-thin">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export default function ModelEvaluation() {
  const metrics = useResource(api.metrics, { key: 'metrics', pollMs: POLL_INTERVALS.artifacts });
  const metadata = useResource(api.modelMetadata, { key: 'metadata', pollMs: POLL_INTERVALS.artifacts });
  const [shapLimit, setShapLimit] = useState(15);
  const shap = useResource((options) => api.shap(shapLimit, options), { key: `shap-${shapLimit}`, pollMs: POLL_INTERVALS.artifacts });

  const data = metrics.data;
  const m = data?.metrics;
  const report = data?.classification_report ?? {};

  const calibrationBins = useMemo(() => {
    const bins = m?.ece_bins ?? [];
    return bins
      .filter((bin) => bin.count > 0 && bin.confidence != null && bin.accuracy != null)
      .map((bin) => ({ mean_predicted: bin.confidence, mean_observed: bin.accuracy, count: bin.count, lo: bin.lo, hi: bin.hi }));
  }, [m]);

  const classRows = ['0', '1', '2'].map((key, index) => {
    const row = report[key] ?? {};
    return {
      label: CLASS_LABELS[index],
      precision: row.precision,
      recall: row.recall,
      f1: row['f1-score'] ?? row.f1,
      support: row.support,
    };
  });

  const meta = metadata.data?.metadata ?? {};

  return (
    <>
      <PageHeader
        eyebrow="Model evaluation"
        title="Test metrics and calibration"
        description="Scores are computed on the chronological holdout. The thresholded accuracy and the plain argmax accuracy are reported separately because they measure different decision rules."
        actions={
          <>
            <SourceTag envelope={metrics.envelope} />
            <Button size="sm" variant="quiet" icon={RefreshCw} onClick={() => { metrics.refresh(); metadata.refresh(); shap.refresh(); }}>
              Reload
            </Button>
          </>
        }
      />

      {metrics.error && (
        <div className="mb-4">
          <ErrorNote error={metrics.error} label="Metrics unavailable" onRetry={metrics.refresh} />
        </div>
      )}
      {metrics.envelope?.warnings?.length > 0 && (
        <div className="mb-4">
          <Warnings warnings={metrics.envelope.warnings} />
        </div>
      )}

      <StatGrid columns={3}>
        <Stat
          label="Accuracy (thresholded)"
          value={pct(m?.accuracy)}
          hint={`Argmax ${pct(data?.decision_rule?.argmax_accuracy)} · difference ${num(
            Math.abs((m?.accuracy ?? 0) - (data?.decision_rule?.argmax_accuracy ?? 0)) * 100,
            2,
          )} pp`}
          footnote={m?.beats_majority_baseline ? 'Beats the majority baseline' : 'Below the majority baseline'}
        />
        <Stat label="Balanced accuracy" value={pct(m?.balanced_accuracy)} hint={`Macro F1 ${num(m?.f1_macro, 4)}`} />
        <Stat label="MCC" value={num(m?.mcc, 4)} hint="Matthews correlation, 0 = chance" />
        <Stat label="ROC AUC (OvR macro)" value={num(m?.roc_auc_ovr_macro, 4)} hint="One-vs-rest, macro averaged" />
        <Stat label="Expected calibration error" value={num(m?.ece, 4)} hint={`Temperature ${num(data?.calibration?.temperature, 4)}`} />
        <Stat
          label="Directional accuracy"
          value={pct(m?.directional_accuracy_nonflat_actual)}
          hint={`When the model takes a direction (${pct(m?.directional_signal_coverage, 1)} of rows)`}
        />
      </StatGrid>

      <div className="mt-4 grid gap-4 lg:grid-cols-5">
        <Panel className="lg:col-span-3" eyebrow="Confusion" title="Holdout confusion matrix">
          {metrics.loading && !data ? <Loading rows={5} /> : <ConfusionMatrix matrix={m?.confusion_matrix} labels={CLASS_LABELS} />}
          <p className="mt-3 text-[.6875rem] text-[var(--ink-faint)]">
            Rows are the realised label, columns the prediction. The FLAT column dominates, which is why accuracy alone is misleading: compare it with
            the majority baseline of {pct(m?.majority_baseline_accuracy)}.
          </p>
        </Panel>

        <Panel className="lg:col-span-2" eyebrow="Per class" title="Precision / recall">
          <Table
            dense
            columns={[
              { key: 'label', header: 'Label', strong: true },
              { key: 'precision', header: 'Prec.', align: 'right', render: (row) => num(row.precision, 3) },
              { key: 'recall', header: 'Recall', align: 'right', render: (row) => num(row.recall, 3) },
              { key: 'f1', header: 'F1', align: 'right', render: (row) => num(row.f1, 3) },
              { key: 'support', header: 'n', align: 'right', render: (row) => int(row.support) },
            ]}
            rows={classRows}
            rowKey={(row) => row.label}
          />
          <div className="mt-3 space-y-2 border-t border-[var(--line)] pt-3">
            <div className="flex items-center justify-between">
              <span className="muted">Test rows</span>
              <span className="mono text-[.75rem]">{int(m?.n_test)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="muted">Test start</span>
              <span className="mono text-[.75rem]">{formatDate(m?.test_start)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="muted">LightGBM − baseline</span>
              <span className="mono text-[.75rem]">{num((m?.lightgbm_minus_baseline_accuracy ?? 0) * 100, 2)} pp</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="muted">Diebold–Mariano (UP)</span>
              <span className="mono text-[.75rem]">{num(m?.dm_up, 4)}</span>
            </div>
          </div>
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel eyebrow="Calibration" title="Reliability diagram">
          {calibrationBins.length ? (
            <CalibrationChart bins={calibrationBins} />
          ) : (
            <Empty title="No populated bins" detail="results/metrics.json has no ECE bins with rows in them." />
          )}
          <p className="mt-2 text-[.6875rem] text-[var(--ink-faint)]">
            Temperature scaling ({num(data?.calibration?.temperature, 4)}) is fitted on validation, not on the holdout shown here.
          </p>
        </Panel>

        <Panel eyebrow="Decision rule" title="Thresholded vs argmax">
          <div className="space-y-2">
            {(data?.decision_rule?.notes ?? []).map((note) => (
              <Callout key={note} tone="info">
                {note}
              </Callout>
            ))}
          </div>
          <div className="mt-3">
            <KeyValue
              items={[
                { label: 'Temperature', value: num(data?.decision_rule?.temperature, 4), mono: true },
                { label: 'Flat threshold', value: num(data?.decision_rule?.flat_threshold, 4), mono: true },
                { label: 'Confidence threshold', value: pct(data?.decision_rule?.confidence_threshold, 0), mono: true },
                { label: 'Thresholded accuracy', value: pct(data?.decision_rule?.thresholded_accuracy), mono: true },
                { label: 'Argmax accuracy', value: pct(data?.decision_rule?.argmax_accuracy), mono: true },
                { label: 'Predictions taken', value: int(m?.directional_predictions_taken), mono: true },
              ]}
            />
          </div>
        </Panel>
      </div>

      <Panel
        className="mt-4"
        eyebrow="Attribution"
        title="SHAP feature importance"
        actions={<Segmented label="Rows" options={SHAP_LIMITS} value={shapLimit} onChange={setShapLimit} />}
      >
        {shap.error ? (
          <ErrorNote error={shap.error} label="SHAP artifact unavailable" onRetry={shap.refresh} />
        ) : (
          <>
            <ImportanceBars rows={shap.data?.rows ?? []} limit={shapLimit} formatter={(value) => num(value, 4)} />
            <p className="mt-3 text-[.6875rem] text-[var(--ink-faint)]">
              Mean absolute SHAP value from results/shap_feature_importance.csv ({shap.data?.metric ?? 'column'}). This is an attribution of the
              trained model, not a causal claim.
            </p>
          </>
        )}
      </Panel>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel eyebrow="Model" title="Training metadata" actions={<Braces size={14} className="text-[var(--ink-faint)]" />}>
          <KeyValue
            items={[
              { label: 'Trained at', value: meta.trained_at ? formatInstant(meta.trained_at) : '—' },
              { label: 'Training rows', value: int(meta.training_rows), mono: true },
              { label: 'Test rows', value: int(meta.test_rows), mono: true },
              { label: 'Features', value: int(metadata.data?.feature_count), mono: true },
              { label: 'Train accuracy', value: pct(meta.train_accuracy), mono: true },
              { label: 'Final test accuracy', value: pct(meta.final_test_accuracy), mono: true },
            ]}
          />
          <div className="mt-3 flex flex-wrap gap-1.5">
            {(metadata.data?.feature_columns ?? []).slice(0, 14).map((feature) => (
              <Badge key={feature} tone="badge-ghost">
                {feature}
              </Badge>
            ))}
            {(metadata.data?.feature_columns ?? []).length > 14 && (
              <Badge tone="badge-dashed">+{metadata.data.feature_columns.length - 14} more</Badge>
            )}
          </div>
        </Panel>

        <Panel eyebrow="Checkpoint" title="Training context">
          <div className="space-y-3">
            <div>
              <p className="eyebrow mb-1">Best params</p>
              <JsonBlock value={metadata.data?.best_params} label="Best params" />
            </div>
            <div>
              <p className="eyebrow mb-1">Checkpoint index</p>
              <JsonBlock value={metadata.data?.checkpoint} label="Checkpoint" />
            </div>
          </div>
        </Panel>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Badge tone="badge-ghost">train rows {int(meta.training_rows)}</Badge>
        <Badge tone="badge-ghost">test rows {int(meta.test_rows)}</Badge>
        <Badge tone="badge-ghost">signed baseline delta {signed((m?.lightgbm_minus_baseline_accuracy ?? 0) * 100, 2)} pp</Badge>
      </div>
    </>
  );
}
