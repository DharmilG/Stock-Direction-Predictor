import { useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate, formatInstant, humanDuration } from '../lib/dates';
import { int, num, pct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { CalibrationChart, ConfusionMatrix, DualAxisLine, LineSeriesChart } from '../components/charts';
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
  KeyValue,
  Loading,
  Panel,
  ProgressBar,
  Segmented,
  Stat,
  Table,
  Warnings,
} from '../components/ui';

/**
 * Q0 tabs: Live · Analytics · Cases.
 *
 * Every panel is a function of a file the run wrote. When a field is missing the
 * panel says so — it never draws a flat line, because a flat utilisation series
 * reads as "the GPU was idle", which is a very different claim from "this was
 * never measured".
 */

const WINDOWS = [
  { value: '30m', label: '30 m' },
  { value: '2h', label: '2 h' },
  { value: 'all', label: 'All' },
];

const THERMAL_TONE = { green: 0.16, amber: 0.42, prepare: 0.6, alarm: 0.82, rest: 0.3, cooldown: 0.95 };

function Missing({ title, detail }) {
  return <Empty title={title} detail={detail} />;
}

/* ------------------------------------------------------------------------- */
/* Live                                                                      */
/* ------------------------------------------------------------------------- */
export function Q0LiveTab() {
  const [window, setWindow] = useState('2h');
  const timeline = useResource((options) => api.q0Timeline({ window, step: 5 }, options), {
    key: `q0-timeline-${window}`,
    pollMs: POLL_INTERVALS.q0,
  });
  const data = timeline.data;
  const fields = data?.resource?.fields ?? [];
  const has = (name) => fields.includes(name);
  const series = data?.resource?.series ?? [];
  const progress = data?.progress?.rows ?? [];
  const history = data?.history?.rows ?? [];
  const psi = data?.psi?.rows ?? [];
  const live = data?.live;
  const memoryTotal = has('mem_total_mb') ? series.map((row) => row.mem_total_mb).filter((value) => value != null).pop() : null;

  // Derived from `data`, not from a local alias, so the memo only rebuilds when
  // the fetched payload actually changes.
  const eta = useMemo(
    () =>
      (data?.history?.rows ?? [])
        .filter((row) => row.eta_hours != null)
        .map((row) => ({ at: row.timestamp, eta_hours: row.eta_hours, rated: row.rated_total })),
    [data],
  );

  return (
    <div className="space-y-6">
      <Panel
        title={live?.running ? 'Runner is writing telemetry' : 'No run in flight'}
        info={{
          what: 'The live view of the run: what it is doing now, and how fresh that claim is.',
          reads: `Heartbeat vs the stale limit, the case in flight, and the sampling step behind the charts below${data?.resource?.source ? `, all read from ${data.resource.source}` : ''}.`,
          ideal: 'A phase that keeps changing and a heartbeat seconds old.',
          when: 'A heartbeat past its limit with the run still marked live means the process died without clearing its state.',
        }}
        actions={
          <>
            <Segmented label="Window" options={WINDOWS} value={window} onChange={setWindow} />
            <Badge tone={live?.running ? 'badge-solid' : 'badge-dashed'}>{live?.running ? 'LIVE' : 'IDLE'}</Badge>
          </>
        }
      >
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="Phase"
            value={live?.phase ?? '—'}
            info={{
              what: 'The research phase the runner is currently inside.',
              reads: live?.updated_at ? `Last heartbeat written ${formatInstant(live.updated_at)}.` : 'No heartbeat has been written yet.',
              ideal: 'A phase that advances every few minutes.',
            }}
          />
          <Stat
            label="Heartbeat age"
            value={humanDuration(live?.heartbeat_age_seconds)}
            info={{
              what: 'How long ago the runner last wrote its heartbeat.',
              reads: `The runner is treated as stale past ${int(live?.limit_seconds)} s.`,
              ideal: 'Seconds, not minutes. The charts below are all sampled on the same clock.',
              when: 'An age beyond the limit means the numbers on this tab describe a run that has stopped.',
            }}
          />
          <Stat
            label="Current case"
            value={live?.current_case ? String(live.current_case).split('|')[0] : '—'}
            info={{
              what: 'The case being judged at the moment of the last heartbeat.',
              reads: live?.current_case ? `Case reference ${String(live.current_case).split('|')[1] ?? 'not recorded'}.` : 'Nothing in flight; the runner is between cases.',
              ideal: 'Advancing from case to case rather than sitting on one, which means the model call is hanging.',
            }}
          />
          <Stat
            label="Window"
            value={window === 'all' ? 'all samples' : window}
            info={{
              what: 'How much history the charts on this tab are drawn from.',
              reads: `Down-sampled to a ${int(data?.resource?.stride_seconds ?? 5)} s step, ${int(series.length)} points.`,
              ideal: 'Wide enough that a throttling episode fits inside it, short enough that the line is still readable.',
              when: data?.resource?.timezone ? `The time axis is ${data.resource.timezone}, not UTC. Do not compare stamps with heartbeats.` : undefined,
            }}
          />
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel
          title="Temperature and SM clock"
          info={{
            what: 'Die temperature against the SM clock, drawn from the same five-second samples.',
            reads: 'Temperature on the left axis, streaming-multiprocessor clock on the right, with the clock the card reports as its maximum as the reference line.',
            ideal: 'A clock pinned at its cap while the die stays in its normal band.',
            when: 'A clock below the cap line while the die is hot is the throttling signature. The same drop on a cool die is a driver or power-limit decision instead.',
          }}
        >
          {series.length && has('temp_c') ? (
            has('clock_sm_mhz') ? (
              <>
                <DualAxisLine
                  data={series}
                  xKey="at"
                  left={{ key: 'temp_c', label: 'temp °C' }}
                  right={{ key: 'clock_sm_mhz', label: 'SM clock MHz', reference: memoryClockCap(series) }}
                  height={230}
                  leftFormatter={(value) => num(value, 0)}
                  rightFormatter={(value) => int(value)}
                  xTickFormatter={(value) => String(value).slice(11, 16)}
                />
              </>
            ) : (
              <>
                <LineSeriesChart
                  data={series}
                  xKey="at"
                  series={[{ key: 'temp_c', label: 'temp °C' }]}
                  height={230}
                  yTickFormatter={(value) => num(value, 0)}
                  xTickFormatter={(value) => String(value).slice(11, 16)}
                />
              </>
            )
          ) : (
            <Missing title="No temperature series" detail="logs/q0_resource_5s.csv and logs/gpu_5s.csv are both empty or missing." />
          )}
        </Panel>

        <Panel
          title="Utilisation and VRAM"
          info={{
            what: 'How hard the GPU is working and how much memory the model holds.',
            reads: 'Utilisation as a percentage of the sampling interval, and VRAM used against the card total on the same time axis.',
            ideal: 'Utilisation high and steady while cases are being judged, and VRAM flat rather than climbing.',
            when: 'VRAM climbing across a run means something is leaking; utilisation sitting flat at zero means the model call is idle or blocked.',
          }}
        >
          {has('util_pct') ? (
            <div className="space-y-4">
              <LineSeriesChart
                data={series}
                xKey="at"
                series={[{ key: 'util_pct', label: 'GPU utilisation %' }]}
                height={180}
                yDomain={[0, 100]}
                yTickFormatter={(value) => pct((value ?? 0) / 100, 0)}
                xTickFormatter={(value) => String(value).slice(11, 16)}
              />
              {has('mem_used_mb') && (
                <LineSeriesChart
                  data={series}
                  xKey="at"
                  series={[{ key: 'mem_used_mb', label: 'VRAM used MB' }]}
                  height={180}
                  yTickFormatter={(value) => int(value)}
                  xTickFormatter={(value) => String(value).slice(11, 16)}
                  reference={memoryTotal ? { value: memoryTotal, label: 'total' } : undefined}
                />
              )}
            </div>
          ) : (
            <Missing
              title="Utilisation was not recorded"
              detail="logs/q0_resource_5s.csv is missing, so only temperature exists. It appears after the next Q0 run; nothing is inferred here."
            />
          )}
        </Panel>

        <Panel
          title="Power draw and fan"
          info={{
            what: 'Power the card is pulling, against how fast the fan is running.',
            reads: 'Watts on the left axis, fan percentage on the right.',
            ideal: 'Power tracking utilisation, with the fan rising only once the die warms.',
            when: 'Power dropping while utilisation stays high is a power limit, not a thermal one. This card reports no fan speed on some runs, which is absent data rather than a flat line.',
          }}
        >
          {has('power_w') && has('fan_pct') ? (
            <DualAxisLine
              data={series}
              xKey="at"
              left={{ key: 'power_w', label: 'power W' }}
              right={{ key: 'fan_pct', label: 'fan %' }}
              height={230}
              leftFormatter={(value) => num(value, 0)}
              rightFormatter={(value) => num(value, 0)}
              xTickFormatter={(value) => String(value).slice(11, 16)}
            />
          ) : has('power_w') ? (
            <>
              <LineSeriesChart
                data={series}
                xKey="at"
                series={[{ key: 'power_w', label: 'power W' }]}
                height={230}
                yTickFormatter={(value) => num(value, 0)}
                xTickFormatter={(value) => String(value).slice(11, 16)}
              />
            </>
          ) : (
            <Missing title="Power and fan were not recorded" detail="These columns arrive with logs/q0_resource_5s.csv." />
          )}
        </Panel>

        <Panel
          title="Cumulative rated vs judged"
          info={{
            what: 'How many cases the run has rated, judged and kept, in ledger order.',
            reads: 'The x axis is the case index, not wall clock: the ledger is append-only and stores no rating timestamp, so a time axis here would be invented.',
            ideal: 'Three lines climbing together. Judged falling behind rated is the harness losing votes; kept falling behind judged is the drop rules working.',
          }}
        >
          {progress.length ? (
            <>
              <LineSeriesChart
                data={progress}
                xKey="seq"
                series={[
                  { key: 'rated', label: 'rated' },
                  { key: 'judged', label: 'judged' },
                  { key: 'kept', label: 'kept', dash: '2 3' },
                ]}
                height={230}
                yTickFormatter={(value) => int(value)}
                xTickFormatter={(value) => String(value)}
              />
            </>
          ) : (
            <Missing title="No ledger rows" detail="data/llm/qwen_rating_ledger.parquet is missing." />
          )}
        </Panel>

        <Panel
          title="Rolling case duration"
          info={{
            what: 'How long a case takes, rolled over the ledger.',
            reads: 'Median and 95th percentile seconds per case, against case index.',
            ideal: 'A flat median with the p95 above it. A creeping median means the machine is slowing or prompts are growing.',
            when: 'A p95 far above the median and rising is the retry tail: cases that had to be re-judged.',
          }}
        >
          {progress.some((row) => row.case_seconds_p50 != null) ? (
            <LineSeriesChart
              data={progress}
              xKey="seq"
              series={[
                { key: 'case_seconds_p50', label: 'p50 s' },
                { key: 'case_seconds_p95', label: 'p95 s', dash: '6 3' },
              ]}
              height={230}
              yTickFormatter={(value) => num(value, 0)}
              xTickFormatter={(value) => String(value)}
            />
          ) : (
            <Missing title="No durations" detail="Case seconds live in the trace files; none were readable." />
          )}
        </Panel>

        <Panel
          title="Rolling agreement and harness failures"
          info={{
            what: 'Whether the rater stays right as the run goes on, and how often its output fails to parse.',
            reads: 'Rolling agreement with a ±2·SE corridor, then the harness fail rate on the same case index.',
            ideal: 'Agreement holding level inside a narrowing corridor, with the fail rate flat near zero.',
            when: 'The corridor widens as the window empties, which is expected rather than a loss of confidence. A fail rate rising is the harness rejecting output it used to accept.',
          }}
        >
          {progress.length ? (
            <div className="space-y-4">
              <LineSeriesChart
                data={progress}
                xKey="seq"
                series={[
                  { key: 'qwen_matches', label: 'agreement' },
                  { key: 'agreement_hi', label: '+2 SE', dash: '4 3' },
                  { key: 'agreement_lo', label: '−2 SE', dash: '4 3' },
                ]}
                height={180}
                yDomain={[0, 1]}
                yTickFormatter={(value) => pct(value, 0)}
                xTickFormatter={(value) => String(value)}
              />
              <LineSeriesChart
                data={progress}
                xKey="seq"
                series={[{ key: 'harness_fail_rate', label: 'harness fail rate' }]}
                height={160}
                yDomain={[0, 1]}
                yTickFormatter={(value) => pct(value, 0)}
                xTickFormatter={(value) => String(value)}
              />
            </div>
          ) : (
            <Missing title="No ledger rows" detail="Rolling agreement needs the ledger." />
          )}
        </Panel>

        <Panel
          title="Rolling vote mix"
          info={{
            what: 'The share of UP, DOWN and FLAT votes as the run progresses.',
            reads: 'Each line is the share of votes for that class in the rolling window.',
            ideal: 'A mix that moves with the data and roughly matches the class balance of the label set.',
            when: 'A mix that barely moves is a rater ignoring the prompt; a mix dominated by one class is the model collapsing onto the majority answer.',
          }}
        >
          {progress.length ? (
            <>
              <LineSeriesChart
                data={progress}
                xKey="seq"
                series={[
                  { key: 'vote_up', label: 'UP' },
                  { key: 'vote_down', label: 'DOWN', dash: '6 3' },
                  { key: 'vote_flat', label: 'FLAT', dash: '2 3' },
                ]}
                height={230}
                yDomain={[0, 1]}
                yTickFormatter={(value) => pct(value, 0)}
                xTickFormatter={(value) => String(value)}
              />
            </>
          ) : (
            <Missing title="No votes to roll up" detail="The ledger has no rows yet." />
          )}
        </Panel>

        <Panel
          title="State band and ETA burn-down"
          info={{
            what: 'One stripe per heartbeat showing the thermal state the guard was in, then the projected finish time over the run.',
            reads: 'Stripe shade is the thermal state legend underneath; hover a stripe for its timestamp, state and peak temperature. The line beneath is ETA in hours against the same heartbeats.',
            ideal: 'Mostly green and amber stripes with a burn-down falling towards zero.',
            when: 'Clusters of rest, cooldown or alarm stripes line up with the saw-tooth dips on the temperature chart. An ETA that rises means the machine slowed, not that work was added.',
          }}
        >
          {history.length ? (
            <div className="space-y-3">
              <div>
                <div className="flex h-3 w-full gap-px overflow-hidden rounded-sm border border-[var(--line)]">
                  {history.map((row, index) => (
                    <div
                      key={`${row.timestamp}-${index}`}
                      className="flex-1"
                      style={{ background: `rgba(17,17,16,${THERMAL_TONE[row.thermal_state] ?? 0.12})` }}
                      title={`${formatInstant(row.timestamp)} · ${row.thermal_state ?? 'unknown'} · ${num(row.max_temperature_c, 0)} °C`}
                    />
                  ))}
                </div>
                <div className="mt-2 flex flex-wrap gap-x-3.5 gap-y-1.5 text-[.8125rem] text-[var(--ink-3)]">
                  {Object.entries(THERMAL_TONE).map(([state, opacity]) => (
                    <span key={state} className="flex items-center gap-1">
                      <span className="inline-block h-2 w-3 rounded-[1px] border border-[var(--line)]" style={{ background: `rgba(17,17,16,${opacity})` }} />
                      {state}
                    </span>
                  ))}
                </div>
              </div>
              {eta.length > 1 ? (
                <LineSeriesChart
                  data={eta}
                  xKey="at"
                  series={[{ key: 'eta_hours', label: 'ETA h' }]}
                  height={180}
                  yTickFormatter={(value) => num(value, 1)}
                  xTickFormatter={(value) => String(value).slice(11, 16)}
                />
              ) : (
                <p className="text-[.875rem] text-[var(--ink-3)]">
                  Not enough heartbeat history for an ETA burn-down yet: at least two readings are needed.
                </p>
              )}
            </div>
          ) : (
            <Missing title="No heartbeat history" detail="logs/q0_history.jsonl is written by the next Q0 run (30 s throttle)." />
          )}
        </Panel>
      </div>

      <Panel
        title="PSI history"
        info={{
          what: 'Feature drift measured after each monitoring run.',
          reads: 'One row per monitor run: the data date, when it was measured, how many features breached the PSI threshold, and the worst offender with its score.',
          ideal: 'Zero alerts, with scores below 0.1. Between 0.1 and 0.25 is worth watching; above 0.25 the model is seeing a population it was not trained on.',
        }}
        flush
      >
        {psi.length ? (
          <Table
            dense
            columns={[
              { key: 'latest_data', header: 'Data as of', mono: true },
              { key: 'generated_at', header: 'Measured', render: (row) => formatInstant(row.generated_at) },
              { key: 'alerts', header: 'Alerts', align: 'right', mono: true, render: (row) => int(row.alerts) },
              { key: 'feature_count', header: 'Features', align: 'right', mono: true, render: (row) => int(row.feature_count) },
              { key: 'leader', header: 'Top feature', render: (row) => row.top?.[0]?.feature ?? '—', mono: true },
              { key: 'leader_psi', header: 'PSI', align: 'right', mono: true, render: (row) => num(row.top?.[0]?.psi, 3) },
            ]}
            rows={[...psi].reverse()}
            rowKey={(row, index) => `${row.generated_at}-${index}`}
          />
        ) : (
          <Missing title="No PSI history" detail="artifacts/state/psi_history.jsonl appears after the next monitor run." />
        )}
      </Panel>

      {timeline.error && <ErrorNote error={timeline.error} label="Timeline unavailable" onRetry={timeline.refresh} />}
      {timeline.envelope?.warnings?.length > 0 && <Warnings warnings={timeline.envelope.warnings} />}
    </div>
  );
}

function memoryClockCap(series) {
  const caps = series.map((row) => row.clock_max_sm_mhz).filter((value) => value != null);
  return caps.length ? caps[caps.length - 1] : undefined;
}

/* ------------------------------------------------------------------------- */
/* Analytics                                                                 */
/* ------------------------------------------------------------------------- */
export function Q0AnalyticsTab() {
  const analytics = useResource(api.q0Analytics, { key: 'q0-analytics', pollMs: POLL_INTERVALS.artifacts });
  const data = analytics.data;

  if (analytics.error) return <ErrorNote error={analytics.error} label="Analytics unavailable" onRetry={analytics.refresh} />;
  if (analytics.loading && !data) return <Loading rows={6} />;
  if (!data) {
    return <Empty title="Nothing to analyse" detail="The rating ledger has not been written yet." />;
  }

  const bins = (data.calibration?.bins ?? []).map((bin) => ({ ...bin, mean_predicted: bin.confidence, mean_observed: bin.accuracy }));
  const matrix = (payload, title) => (
    <Panel
      title={title}
      info={{
        what: 'Which way the rater errs: rows are what the stock did, columns what the rater said.',
        reads: 'Cell shade is the count. A column-heavy block is that class being over-produced; a row-heavy block is it being missed.',
        ideal: 'A dark diagonal with the off-diagonal cells small and roughly balanced between the two.',
        when: 'One column carrying most of the mass means the rater has collapsed onto a single answer, which is the usual failure of a prompt-led classifier.',
      }}
    >
      <ConfusionMatrix matrix={payload?.counts} labels={payload?.cols} />
    </Panel>
  );

  return (
    <div className="space-y-6">
      <Panel
        title="Attempted → rated → judged → kept"
        info={{
          what: 'The four stages a case has to pass before its rating counts as data.',
          reads: 'Attempted is every ledger row; rated has a vote; judged has a vote that parses; kept survives the drop rules. Retention columns compare each stage with the one before it.',
          ideal: 'Attempted and rated equal, with judged close behind and a small, explained loss at kept.',
          when: 'A large drop between rated and judged is a harness failure, not a modelling problem. Look at the drop reasons on the Status tab.',
        }}
        actions={<Badge tone="badge-ghost">{int(data.rows)} ledger rows</Badge>}
      >
        <div className="space-y-4">
          <ProgressBar label="Judged" value={data.funnel?.find((step) => step.step === 'judged')?.n ?? 0} max={data.funnel?.[0]?.n ?? 0} detail={`judged rate ${pct(data.judged_rate)}`} />
          <ProgressBar label="Kept" value={data.funnel?.find((step) => step.step === 'kept')?.n ?? 0} max={data.funnel?.[0]?.n ?? 0} detail={`harness fail ${pct(data.harness_fail_rate)}`} />
        </div>
        <div className="mt-5">
          <Table
            dense
            columns={[
              { key: 'label', header: 'Step', strong: true },
              { key: 'n', header: 'Cases', align: 'right', mono: true, render: (row) => int(row.n) },
              { key: 'share_of_attempted', header: 'of attempted', align: 'right', mono: true, render: (row) => pct(row.share_of_attempted, 1) },
              { key: 'retention', header: 'of previous step', align: 'right', mono: true, render: (row) => (row.retention == null ? '—' : pct(row.retention, 1)) },
            ]}
            rows={data.funnel ?? []}
            rowKey={(row) => row.step}
          />
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        {matrix(data.votes_by_label, 'Qwen vote × realised label')}
        {matrix(data.finbert_by_label, 'FinBERT vote × realised label')}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel
          title="Does confidence mean anything?"
          info={{
            what: 'Whether a confident vote is more likely to be right than a hesitant one.',
            reads: `Each point buckets votes by stated confidence and plots that against how often the vote was actually correct, over ${int(data.calibration?.n)} judged rows. ECE is the average gap between the two.`,
            ideal: 'Points on the diagonal, with ECE under 0.05. A point below the diagonal is over-confidence: the vote was right less often than the model claimed.',
            when: 'A flat line at the base rate means confidence carries no information at all, so it cannot be used to rank cases.',
          }}
          actions={<Badge tone="badge-ghost">ECE {num(data.calibration?.ece, 4)}</Badge>}
        >
          {bins.length ? (
            <>
              <CalibrationChart bins={bins} height={240} />
            </>
          ) : (
            <Missing title="No calibration" detail="qwen_conf or vote_matches_label is missing from the ledger." />
          )}
        </Panel>

        <Panel
          title="Confidence distribution"
          info={{
            what: 'How sure the rater usually is.',
            reads: 'Share of votes falling in each confidence bucket, with the count printed on the right.',
            ideal: 'A spread across the upper buckets rather than a spike at one value.',
            when: 'Almost everything piling into one bucket means the model is emitting a near-constant confidence, which makes the calibration chart meaningless.',
          }}
        >
          {data.confidence_hist?.length ? (
            <div className="space-y-2.5">
              {data.confidence_hist.map((bin) => (
                <div key={`${bin.lo}-${bin.hi}`} className="grid grid-cols-[3.9rem_minmax(0,1fr)_3.4rem] items-center gap-3">
                  <span className="mono text-[.8125rem] text-[var(--ink-3)]">
                    {bin.lo}–{bin.hi}
                  </span>
                  <div className="h-2 w-full rounded-sm bg-[var(--surface-sunken)]">
                    <div className="h-full rounded-sm bg-[var(--ink)]" style={{ width: `${(bin.share ?? 0) * 100}%` }} />
                  </div>
                  <span className="mono text-right text-[.8125rem] text-[var(--ink-3)]">{int(bin.n)}</span>
                </div>
              ))}
            </div>
          ) : (
            <Missing title="No confidence recorded" detail="qwen_conf is missing from the ledger." />
          )}
        </Panel>
      </div>

      <Panel
        title="Who follows the label"
        info={{
          what: 'How the two raters split the cases between them.',
          reads: `Qwen ${pct(data.disagreement?.qwen_matches)} against FinBERT ${pct(data.disagreement?.finbert_matches)} on ${data.disagreement?.quoted_on}.`,
          ideal: 'Both agree dominating, with Qwen-only above FinBERT-only if Qwen is to win the gate.',
          when: 'The gate compares the two match rates, so a large "neither" bucket caps how far either rater can get.',
        }}
      >
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="Both agree"
            value={int(data.disagreement?.both_match)}
            info={{ what: 'Cases where both raters matched the label.', reads: 'The bucket that neither rater can win alone.' }}
          />
          <Stat
            label="Qwen only"
            value={int(data.disagreement?.qwen_only)}
            info={{ what: 'Cases only Qwen got right.', reads: 'FinBERT misses these.' }}
          />
          <Stat
            label="FinBERT only"
            value={int(data.disagreement?.finbert_only)}
            info={{ what: 'Cases only FinBERT got right.', reads: 'Qwen misses these.' }}
          />
          <Stat
            label="Neither"
            value={int(data.disagreement?.neither)}
            info={{
              what: 'Cases both raters got wrong.',
              reads: 'An upper bound on how bad the labels or the prompt are.',
              ideal: 'Small. This bucket is what remains when the text carries no usable signal for either model.',
            }}
          />
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel
          title="Seconds per case, by outcome"
          info={{
            what: 'How long a case takes, split by how it ended.',
            reads: `Duration percentiles per outcome, read from ${data.latency?.source ?? 'the trace files'}.`,
            ideal: 'Similar medians across outcomes; the outcome of a case should not decide how long it took.',
            when: 'Harness failures being the slowest group is expected: they are the retries that ran out of output.',
          }}
        >
          {data.latency?.available ? (
            <>
              <Table
                dense
                columns={[
                  { key: 'outcome', header: 'Outcome', strong: true },
                  { key: 'n', header: 'Cases', align: 'right', mono: true, render: (row) => int(row.n) },
                  { key: 'p50', header: 'p50 s', align: 'right', mono: true, render: (row) => num(row.p50, 1) },
                  { key: 'p90', header: 'p90 s', align: 'right', mono: true, render: (row) => num(row.p90, 1) },
                  { key: 'mean', header: 'mean s', align: 'right', mono: true, render: (row) => num(row.mean, 1) },
                ]}
                rows={data.latency.by_outcome ?? []}
                rowKey={(row) => row.outcome}
              />
            </>
          ) : (
            <Missing title="No durations" detail="results/traces/*.json were not readable." />
          )}
        </Panel>

        <Panel
          title="News vs no-news"
          info={{
            what: 'Whether the rater does better on cases that had news attached.',
            reads: 'Match rate for each rater and mean confidence, split by whether the case had headlines.',
            ideal: 'Qwen ahead on the news cohort and no worse than FinBERT on the no-news control.',
            when: 'No-news rows are the honest control: a rater that only wins on news is reading the text rather than the prior.',
          }}
        >
          <Table
            dense
            columns={[
              { key: 'cohort', header: 'Cohort', strong: true },
              { key: 'n', header: 'Judged', align: 'right', mono: true, render: (row) => int(row.n) },
              { key: 'qwen_matches', header: 'Qwen', align: 'right', mono: true, render: (row) => pct(row.qwen_matches, 1) },
              { key: 'finbert_matches', header: 'FinBERT', align: 'right', mono: true, render: (row) => pct(row.finbert_matches, 1) },
              { key: 'mean_confidence', header: 'conf', align: 'right', mono: true, render: (row) => num(row.mean_confidence, 3) },
            ]}
            rows={data.cohorts ?? []}
            rowKey={(row) => row.cohort}
          />
        </Panel>
      </div>

      <Panel
        title="Agreement by symbol"
        info={{
          what: 'Whether the rater is carried by a handful of names or works across the universe.',
          reads: 'Judged cases, Qwen match rate, and mean confidence per symbol, with the mean FinBERT score beside it.',
          ideal: 'Match rates spread across symbols rather than a few extremes on small samples.',
          when: 'A symbol with a handful of judged cases can show 0% or 100% by luck; read the judged count before the rate.',
        }}
        flush
      >
        <Table
          dense
          columns={[
            { key: 'symbol', header: 'Symbol', strong: true, mono: true },
            { key: 'n', header: 'Judged', align: 'right', mono: true, render: (row) => int(row.n) },
            { key: 'qwen_matches', header: 'Qwen agrees', align: 'right', mono: true, render: (row) => pct(row.qwen_matches, 1) },
            { key: 'mean_confidence', header: 'mean conf', align: 'right', mono: true, render: (row) => num(row.mean_confidence, 3) },
            { key: 'mean_finbert', header: 'mean FinBERT', align: 'right', mono: true, render: (row) => num(row.mean_finbert, 3) },
          ]}
          rows={data.per_symbol ?? []}
          rowKey={(row) => row.symbol}
          empty={<Empty title="No symbols" detail="The ledger has no judged rows." />}
        />
      </Panel>

      {analytics.envelope?.warnings?.length > 0 && <Warnings warnings={analytics.envelope.warnings} />}
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/* Cases                                                                     */
/* ------------------------------------------------------------------------- */
const VOTE_OPTIONS = [
  { value: 'all', label: 'Any' },
  { value: 'UP', label: 'UP' },
  { value: 'DOWN', label: 'DOWN' },
  { value: 'FLAT', label: 'FLAT' },
];
const OUTCOME_OPTIONS = [
  { value: 'all', label: 'Any' },
  { value: 'match', label: 'Matches label' },
  { value: 'mismatch', label: 'Misses label' },
];

function TraceDrawer({ caseId, onClose }) {
  const detail = useResource((options) => api.q0Case(caseId, options), { key: `q0-case-${caseId}` });
  const data = detail.data;
  const trace = data?.trace;
  const ledger = data?.ledger ?? {};
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/25" role="dialog" aria-modal="true" aria-label={caseId}>
      <div className="flex h-full w-full max-w-3xl flex-col overflow-hidden border-l border-[var(--line)] bg-white">
        <header className="flex items-start justify-between gap-3 border-b border-[var(--line)] px-4 py-3">
          <div className="min-w-0">
            <p className="eyebrow">Case trace</p>
            <h2 className="section-title mono mt-0.5 break-all">{caseId}</h2>
          </div>
          <Button size="sm" variant="quiet" icon={X} onClick={onClose}>
            Close
          </Button>
        </header>
        <div className="flex-1 space-y-6 overflow-auto px-5 py-5 scroll-thin">
          {detail.error && <ErrorNote error={detail.error} label="Trace unavailable" onRetry={detail.refresh} />}
          {detail.loading && !data && <Loading rows={6} />}
          {data && (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="badge-ghost">{ledger.symbol ?? '—'}</Badge>
                <Badge tone="badge-ghost">{formatDate(ledger.decision_date)}</Badge>
                <Badge tone={ledger.qwen_vote === 'UP' ? 'badge-solid' : 'badge-mid'}>qwen {ledger.qwen_vote ?? '—'}</Badge>
                {ledger.historical_label && <Badge tone="badge-dashed">label {String(ledger.historical_label)}</Badge>}
                {ledger.vote_matches_label != null && (
                  <Badge tone={ledger.vote_matches_label ? 'badge-ghost' : 'badge-alert'}>{ledger.vote_matches_label ? 'matches' : 'misses'}</Badge>
                )}
                {data.trace_available ? null : <Badge tone="badge-alert">no trace file</Badge>}
              </div>
              <Panel
                title="Recorded row"
                info={{
                  what: 'What the ledger stored for this case id, before any interpretation.',
                  reads: 'Pool, model stage and prompt version, the vote and its confidence, the FinBERT comparison, the thermal state at the time, and whether the row was kept for training.',
                  ideal: 'Kept yes with no drop reason. The thermal state should not be an alarm or rest for a case that was counted.',
                }}
              >
                <KeyValue
                  columns={3}
                  items={[
                    { label: 'Pool', value: String(ledger.pool ?? '—'), mono: true },
                    { label: 'Stage', value: String(ledger.model_stage ?? '—'), mono: true },
                    { label: 'Prompt', value: String(ledger.prompt_version ?? '—'), mono: true },
                    { label: 'Confidence', value: num(ledger.qwen_conf, 3), mono: true },
                    { label: 'FinBERT', value: `${String(ledger.finbert_vote ?? '—')} (${num(ledger.finbert_score, 3)})`, mono: true },
                    { label: 'Thermal', value: String(ledger.thermal_state ?? '—'), mono: true },
                    { label: 'GPU start → end', value: `${num(ledger.gpu_temp_start, 0)} → ${num(ledger.gpu_temp_end, 0)} °C`, mono: true },
                    { label: 'Kept', value: ledger.kept_for_training ? 'yes' : 'no', mono: true },
                    { label: 'Drop reason', value: String(ledger.drop_reason ?? '—'), mono: true },
                  ]}
                />
              </Panel>
              {trace ? (
                <>
                  <Panel
                    title="Vote and content"
                    info={{
                      what: 'What the model actually returned for this case, and the sampler settings it ran under.',
                      reads: 'Vote, confidence, seconds and model are read back from the trace header; the settings are what the runner passed at the time.',
                      ideal: 'A vote that parses and a trace that was not truncated.',
                    }}
                  >
                    <KeyValue
                      columns={3}
                      items={[
                        { label: 'Vote', value: String(trace.vote ?? '—'), mono: true },
                        { label: 'Confidence', value: num(trace.confidence, 3), mono: true },
                        { label: 'Seconds', value: num(trace.seconds, 1), mono: true },
                        { label: 'Model', value: String(trace.model ?? '—'), mono: true },
                        { label: 'num_ctx / num_predict', value: `${int(trace.num_ctx)} / ${int(trace.num_predict)}`, mono: true },
                        { label: 'Temperature / top_p', value: `${num(trace.temperature, 2)} / ${num(trace.top_p, 2)}`, mono: true },
                      ]}
                    />
                    {trace.truncated && (
                      <p className="mt-4 border-l-2 border-[var(--ink)] pl-3 text-[.875rem] text-[var(--ink-2)]">
                        The answer was truncated, so the vote came from an incomplete trace. The ledger counts these separately.
                      </p>
                    )}
                  </Panel>
                  <Panel
                    title="What the rater was shown"
                    info={{
                      what: 'The full prompt for this case: the instructions, the feature summary and any headlines.',
                      reads: 'Verbatim from the trace file, unedited.',
                      when: 'A prompt with missing headlines explains a vote that ignores the news.',
                    }}
                  >
                    <pre className="mono max-h-56 overflow-auto whitespace-pre-wrap rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-3 py-2.5 text-[.8125rem]">
                      {trace.prompt || '—'}
                    </pre>
                  </Panel>
                  <Panel
                    title="Reasoning tokens"
                    info={{
                      what: 'The model reasoning before it answered, when the run recorded one.',
                      reads: 'Verbatim from the trace. A model that skips this block votes on the prompt alone.',
                      ideal: 'Reasoning that names the features or headlines it used, rather than restating the question.',
                    }}
                  >
                    <pre className="mono max-h-64 overflow-auto whitespace-pre-wrap rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-3 py-2.5 text-[.8125rem]">
                      {trace.thinking || '— no thinking block recorded —'}
                    </pre>
                  </Panel>
                  <Panel
                    title="Raw answer"
                    info={{
                      what: 'Exactly what the model emitted, before the harness parsed it.',
                      reads: 'This is the text the harness had to turn into a vote. A malformed answer is visible here first.',
                      ideal: 'A short answer matching the requested schema, with nothing after it.',
                    }}
                  >
                    <pre className="mono max-h-56 overflow-auto whitespace-pre-wrap rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-3 py-2.5 text-[.8125rem]">
                      {trace.content || '— empty —'}
                    </pre>
                  </Panel>
                </>
              ) : (
                <Empty title="Trace file missing" detail="The ledger row exists but results/traces has no file for it. Nothing is reconstructed here." />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function Q0CasesTab() {
  const [symbol, setSymbol] = useState('');
  const [vote, setVote] = useState('all');
  const [outcome, setOutcome] = useState('all');
  const [offset, setOffset] = useState(0);
  const [active, setActive] = useState(null);
  const limit = 25;

  const filters = useMemo(() => ({ symbol, vote, outcome, limit, offset }), [symbol, vote, outcome, offset]);
  const cases = useResource((options) => api.q0Cases(filters, options), { key: `q0-cases-${symbol}-${vote}-${outcome}-${offset}` });
  const data = cases.data;
  const reset = (setter) => (value) => {
    setter(value);
    setOffset(0);
  };

  return (
    <div className="space-y-6">
      <Panel
        title="Ledger rows"
        info={{
          what: 'Filter the append-only ledger down to the cases you want to inspect.',
          reads: 'Vote is what the rater said, outcome is whether that matched the realised label, symbol narrows to one name.',
          ideal: 'Wide filters when auditing a run, one symbol when chasing a specific miss.',
        }}
        actions={<Badge tone="badge-ghost">{data ? `${int(data.total)} rows` : '—'}</Badge>}
      >
        <div className="flex flex-wrap items-end gap-3">
          <label className="block">
            <span className="eyebrow">Vote</span>
            <div className="mt-1">
              <Segmented label="Vote" options={VOTE_OPTIONS} value={vote} onChange={reset(setVote)} />
            </div>
          </label>
          <label className="block">
            <span className="eyebrow">Outcome</span>
            <div className="mt-1">
              <Segmented label="Outcome" options={OUTCOME_OPTIONS} value={outcome} onChange={reset(setOutcome)} />
            </div>
          </label>
          <label className="block">
            <span className="eyebrow">Symbol</span>
            <input
              className="field mt-1 w-44"
              placeholder="ADANIPORTS.NS"
              value={symbol}
              onChange={(event) => reset(setSymbol)(event.target.value.toUpperCase())}
            />
          </label>
        </div>
      </Panel>

      <Panel
        title="Cases"
        info={{
          what: 'One row per ledger case, with the verdict each rater reached and whether it matched the label.',
          reads: 'Click a row to open its trace: the prompt, the reasoning, the raw answer and the recorded ledger row.',
          ideal: 'A match rate in line with the Analytics tab. A run of misses on one symbol is worth opening case by case.',
          when: 'A row with no trace file shows the ledger entry but nothing to read, which is a missing artifact rather than a failed case.',
        }}
        flush
        bodyClass="px-0 py-0"
      >
        {cases.error ? (
          <div className="p-4">
            <ErrorNote error={cases.error} label="Case table unavailable" onRetry={cases.refresh} />
          </div>
        ) : cases.loading && !data ? (
          <div className="p-4">
            <Loading rows={6} />
          </div>
        ) : (
          <>
            <Table
              dense
              columns={[
                { key: 'symbol', header: 'Symbol', strong: true, mono: true },
                { key: 'decision_date', header: 'Decision date', mono: true, render: (row) => formatDate(row.decision_date) },
                { key: 'vote', header: 'Vote', render: (row) => row.vote ?? '—' },
                { key: 'label', header: 'Label', render: (row) => row.label ?? '—' },
                { key: 'confidence', header: 'conf', align: 'right', mono: true, render: (row) => num(row.confidence, 3) },
                { key: 'finbert_vote', header: 'FinBERT', render: (row) => row.finbert_vote ?? '—' },
                {
                  key: 'matches_label',
                  header: '',
                  render: (row) =>
                    row.matches_label == null ? (
                      <span className="text-[var(--ink-faint)]">—</span>
                    ) : (
                      <Badge tone={row.matches_label ? 'badge-ghost' : 'badge-alert'}>{row.matches_label ? 'match' : 'miss'}</Badge>
                    ),
                },
              ]}
              rows={data?.rows ?? []}
              rowKey={(row) => row.case_id}
              onRowClick={setActive}
              empty={<Empty title="No cases match those filters" detail="Widen the filters; the table only shows ledger rows." />}
            />
            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[var(--line)] px-4 py-3">
              <span className="text-[.875rem] text-[var(--ink-2)]">
                {data?.total ? `Showing ${offset + 1}–${Math.min(offset + limit, data.total)} of ${int(data.total)}` : 'No rows'}
              </span>
              <div className="flex gap-2">
                <Button size="sm" variant="quiet" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
                  Previous
                </Button>
                <Button size="sm" variant="quiet" disabled={(data?.total ?? 0) <= offset + limit} onClick={() => setOffset(offset + limit)}>
                  Next
                </Button>
              </div>
            </div>
          </>
        )}
      </Panel>

      {active && <TraceDrawer caseId={active.case_id} onClose={() => setActive(null)} />}
    </div>
  );
}
