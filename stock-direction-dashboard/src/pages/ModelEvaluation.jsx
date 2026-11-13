import { useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate, formatInstant } from '../lib/dates';
import { int, num, pct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { modelKey, useModel } from '../state/ModelProvider';
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
  Section,
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
    <pre className="mono max-h-64 overflow-auto rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] p-3 text-[.8125rem] leading-relaxed text-[var(--ink-2)] scroll-thin">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export default function ModelEvaluation() {
  const { modelId, label, servable } = useModel();
  const metrics = useResource((options) => api.metrics(modelId, options), {
    key: modelKey(modelId, 'metrics'),
    pollMs: POLL_INTERVALS.artifacts,
  });
  const metadata = useResource((options) => api.modelMetadata(modelId, options), {
    key: modelKey(modelId, 'metadata'),
    pollMs: POLL_INTERVALS.artifacts,
  });
  const [shapLimit, setShapLimit] = useState(15);
  const shap = useResource((options) => api.shap(modelId, shapLimit, options), {
    key: modelKey(modelId, `shap-${shapLimit}`),
    pollMs: POLL_INTERVALS.artifacts,
  });

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
        eyebrow={`Model evaluation · ${label}`}
        title="Test metrics and calibration"
        description="Scores are computed on the chronological holdout. Thresholded accuracy and plain argmax accuracy are reported separately because they measure different decision rules. Every figure on this page belongs to the model selected in the header."
        actions={
          <>
            <SourceTag envelope={metrics.envelope} />
            <Badge tone={servable ? 'badge-ghost' : 'badge-dashed'} title={servable ? 'Served model' : 'Research model: measured here, never served.'}>
              {servable ? 'served' : 'research'}
            </Badge>
            <Button size="sm" variant="quiet" icon={RefreshCw} onClick={() => { metrics.refresh(); metadata.refresh(); shap.refresh(); }}>
              Reload
            </Button>
          </>
        }
      />

      {metrics.error && (
        <div className="mb-5">
          <ErrorNote error={metrics.error} label="Metrics unavailable" onRetry={metrics.refresh} />
        </div>
      )}
      {metrics.envelope?.warnings?.length > 0 && (
        <div className="mb-5">
          <Warnings warnings={metrics.envelope.warnings} />
        </div>
      )}

      <Section eyebrow="Headline" title="How the holdout scored">
        <StatGrid columns={3}>
          <Stat
            label="Accuracy (thresholded)"
            value={pct(m?.accuracy)}
            info={{
              what: 'Share of holdout rows where the thresholded decision rule named the right class.',
              reads: 'Same rows as the argmax score, but rows below the confidence threshold are pushed to FLAT before scoring.',
              ideal: `Above the majority baseline of ${pct(m?.majority_baseline_accuracy)} by more than two standard errors.`,
              when: 'At or below the baseline, the rule is no better than always naming the most common class.',
            }}
          />
          <Stat
            label="Balanced accuracy"
            value={pct(m?.balanced_accuracy)}
            info={{
              what: 'Accuracy re-weighted so each class counts equally, whatever its share of rows.',
              reads: 'Mean of the three per-class recalls. FLAT is rare, so this number sits far below plain accuracy.',
              ideal: 'Well above one third. Anything near 0.33 means at least one class is never found.',
            }}
          />
          <Stat
            label="MCC"
            value={num(m?.mcc, 4)}
            info={{
              what: 'One correlation number between prediction and truth across all three classes.',
              reads: 'Ranges from −1 to +1, where 0 is chance. Unlike accuracy it cannot be inflated by always naming the majority class.',
              ideal: 'Above 0.05 is a real, if small, signal on this target. Under 0.01 is indistinguishable from noise.',
            }}
          />
          <Stat
            label="ROC AUC (OvR macro)"
            value={num(m?.roc_auc_ovr_macro, 4)}
            info={{
              what: 'How well the probabilities rank rows, averaged over the three classes.',
              reads: 'One-vs-rest AUC per class, averaged with equal weight. It ignores the decision threshold entirely.',
              ideal: 'Above 0.55. Exactly 0.50 means the scores carry no ordering information at all.',
              when: 'Below 0.50 means the scores are ranking rows backwards.',
            }}
          />
          <Stat
            label="Expected calibration error"
            value={num(m?.ece, 4)}
            info={{
              what: 'How far a predicted probability sits from the frequency actually observed.',
              reads: `Weighted gap between confidence and accuracy across the reliability bins, after temperature scaling at ${num(data?.calibration?.temperature, 4)}.`,
              ideal: 'Below 0.05, the F-14 gate. At 0.04 a 60% call happens about 60% of the time.',
              when: 'Above 0.10 the numbers should not be read as probabilities.',
            }}
          />
          <Stat
            label="Directional accuracy"
            value={pct(m?.directional_accuracy_nonflat_actual)}
            info={{
              what: 'Accuracy on the rows where the model actually took a direction.',
              reads: `Only UP and DOWN calls on rows whose realised direction was also non-flat are scored, which is ${pct(m?.directional_signal_coverage, 1)} of the holdout.`,
              ideal: 'Comfortably above 50%. Acting on it needs roughly 57% or more, because a 31.5 bps trade eats into a move of about 1.6%.',
              when: 'Near 50% means the direction calls are coin flips wearing a confidence score.',
            }}
          />
        </StatGrid>
      </Section>

      <Section eyebrow="Holdout" title="Where the errors land">
        <div className="grid gap-6 lg:grid-cols-5">
          <Panel
            className="lg:col-span-3"
            title="Holdout confusion matrix"
            info={{
              what: 'Every holdout row cross-tabbed by what it was and what the model said it was.',
              reads: 'Rows are the realised label, columns the prediction, shade the count. The FLAT column carries most of the mass.',
              ideal: 'A dark diagonal with thin off-diagonals. Because FLAT is rare, a wide FLAT column is the usual failure shape, not a good one.',
              when: 'Accuracy looks high while the diagonal is dim: that is the majority class being predicted, not skill.',
            }}
          >
            {metrics.loading && !data ? <Loading rows={5} /> : <ConfusionMatrix matrix={m?.confusion_matrix} labels={CLASS_LABELS} />}
          </Panel>

          <Panel
            className="lg:col-span-2"
            title="Precision / recall"
            info={{
              what: 'Per-class precision, recall and F1 on the holdout.',
              reads: 'Straight from the classification report, with the class row count as n.',
              ideal: 'DOWN and UP recall both well above the FLAT share of rows. A UP recall near zero means bullish moves are never caught.',
              when: 'High precision with near-zero recall means the model only calls a class when it is certain, and misses most of them.',
            }}
          >
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
            <div className="mt-4 space-y-3 border-t border-[var(--line)] pt-4">
              <div className="flex items-center justify-between gap-3">
                <span className="muted">Test rows</span>
                <span className="mono text-[.875rem]">{int(m?.n_test)}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="muted">Test start</span>
                <span className="mono text-[.875rem]">{formatDate(m?.test_start)}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="muted">LightGBM − baseline</span>
                <span className="mono text-[.875rem]">{num((m?.lightgbm_minus_baseline_accuracy ?? 0) * 100, 2)} pp</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="muted">Diebold–Mariano (UP)</span>
                <span className="mono text-[.875rem]">{num(m?.dm_up, 4)}</span>
              </div>
            </div>
          </Panel>
        </div>
      </Section>

      <Section eyebrow="Calibration" title="Do the probabilities mean anything">
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="Reliability diagram"
            info={{
              what: 'Whether a predicted probability matches the frequency it actually occurs.',
              reads: 'Each dot is a confidence bin: predicted probability on the horizontal axis, observed accuracy on the vertical. The dashed diagonal is perfect calibration; bubble size is the bin count.',
              ideal: 'Dots hugging the diagonal with ECE under 0.05.',
              when: 'Dots below the diagonal mean the model says it is more sure than it is. Temperature scaling corrects exactly this, and it is fitted on validation rather than on the holdout shown here.',
            }}
          >
            {calibrationBins.length ? (
              <CalibrationChart bins={calibrationBins} />
            ) : (
              <Empty title="No populated bins" detail="results/metrics.json has no ECE bins with rows in them." />
            )}
          </Panel>

          <Panel
            title="Thresholded vs argmax"
            info={{
              what: 'The exact thresholds in force, and what each decision rule costs in accuracy.',
              reads: 'Two rules over the same rows: plain argmax, and argmax with a flat band plus a confidence threshold.',
              ideal: 'Thresholded and argmax accuracy close together. A wide gap means the confidence filter is throwing away signal rather than noise.',
            }}
          >
            <div className="space-y-3">
              {(data?.decision_rule?.notes ?? []).map((note) => (
                <Callout key={note} tone="info">
                  {note}
                </Callout>
              ))}
            </div>
            <div className="mt-5">
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
      </Section>

      <Section eyebrow="Attribution" title="What the model leans on">
        <Panel
          title="SHAP feature importance"
          info={{
            what: 'Which input columns the trained booster depends on most.',
            reads: 'Mean absolute SHAP value per feature, from results/shap_feature_importance.csv. This attributes the trained model; it is not a causal claim.',
            ideal: 'Importance spread over many features, and the top features are ones a trader could explain.',
            when: 'One column holding most of the gain, or a column that is really a clock, is the classic leak signature.',
          }}
          actions={<Segmented label="Rows" options={SHAP_LIMITS} value={shapLimit} onChange={setShapLimit} />}
        >
          {shap.error ? (
            <ErrorNote error={shap.error} label="SHAP artifact unavailable" onRetry={shap.refresh} />
          ) : (
            <ImportanceBars rows={shap.data?.rows ?? []} limit={shapLimit} formatter={(value) => num(value, 4)} />
          )}
        </Panel>
      </Section>

      <Section eyebrow="Model" title="What was trained">
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="Training metadata"
            info={{
              what: 'What was trained, when, and on how many rows.',
              reads: 'The sidecar written next to the booster: train and test row counts, feature count and the two accuracies.',
              ideal: 'Train accuracy and test accuracy in the same neighbourhood. A wide gap is memorisation.',
            }}
          >
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
            <div className="mt-4 flex flex-wrap gap-2">
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

          <Panel
            title="Training context"
            info={{
              what: 'The hyperparameters and checkpoint the final booster was built from.',
              reads: 'Raw JSON from the checkpoint sidecar and the saved best-parameter set, printed without editing.',
              ideal: 'Param counts and round counts that match the run you believe produced the artifacts you are reading.',
            }}
          >
            <div className="space-y-4">
              <div>
                <p className="eyebrow mb-2">Best params</p>
                <JsonBlock value={metadata.data?.best_params} label="Best params" />
              </div>
              <div>
                <p className="eyebrow mb-2">Checkpoint index</p>
                <JsonBlock value={metadata.data?.checkpoint} label="Checkpoint" />
              </div>
            </div>
          </Panel>
        </div>
      </Section>
    </>
  );
}
