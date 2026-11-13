import { useState } from 'react';
import { Activity } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatAny, formatDate, formatInstant, humanDuration } from '../lib/dates';
import { int, num, pct, titleize } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { useDashboard } from '../state/DashboardProvider';
import { LineSeriesChart } from '../components/charts';
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
  KeyValue,
  Loading,
  PageHeader,
  Panel,
  ProgressBar,
  Section,
  Segmented,
  SourceTag,
  Stat,
  StatGrid,
  Table,
  Warnings,
} from '../components/ui';
import { Q0AnalyticsTab, Q0CasesTab, Q0LiveTab } from './Q0Tabs';

// Status is the v1 page, unchanged. The three new tabs read telemetry the v1
// dashboard already collected but never plotted.
const TABS = [
  { value: 'status', label: 'Status' },
  { value: 'live', label: 'Live' },
  { value: 'analytics', label: 'Analytics' },
  { value: 'cases', label: 'Cases' },
];

function GateBadge({ gate }) {
  const tone = gate?.verdict === 'PASS' ? 'badge-solid' : gate?.verdict === 'FAIL' ? 'badge-alert' : 'badge-dashed';
  return <Badge tone={tone}>{gate?.verdict ?? 'UNKNOWN'}</Badge>;
}

/**
 * One planned or recorded Q-track phase.
 *
 * `recorded` separates the two cases the API can return. When the state file
 * exists each phase carries its own status; when it does not, the API fills
 * every phase with "pending" because that is the shape the UI expects — so the
 * badge has to say "not recorded" rather than claim a status nobody wrote.
 */
function PhaseRow({ phase, recorded = true }) {
  const tone = !recorded
    ? 'badge-dashed'
    : phase.status === 'complete' || phase.status === 'done'
      ? 'badge-solid'
      : phase.status === 'running'
        ? 'badge-alert'
        : 'badge-dashed';
  return (
    <li className="flex items-start justify-between gap-3 border-t border-[var(--line)] py-3 first:border-t-0 first:pt-0">
      <div className="min-w-0">
        <p className="mono text-[.875rem] font-semibold text-[var(--ink)]">{phase.name}</p>
        {phase.note && <p className="mt-0.5 text-[.875rem] text-[var(--ink-3)]">{phase.note}</p>}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {phase.updated_at && <span className="hidden text-[.8125rem] text-[var(--ink-3)] sm:inline">{formatAny(phase.updated_at)}</span>}
        <Badge tone={tone}>{recorded ? phase.status : 'not recorded'}</Badge>
      </div>
    </li>
  );
}

function ShareBars({ title, data, limit = 6 }) {
  const entries = Object.entries(data ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);
  const total = entries.reduce((sum, [, value]) => sum + value, 0) || 1;
  if (!entries.length) return <p className="text-[.875rem] text-[var(--ink-3)]">No values recorded.</p>;
  return (
    <div>
      {title && <p className="eyebrow mb-2">{title}</p>}
      <ul className="space-y-2.5">
        {entries.map(([key, value]) => (
          <li key={key} className="grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-2">
            <div className="min-w-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="mono truncate text-[.875rem] text-[var(--ink-2)]" title={key}>
                  {titleize(key)}
                </span>
                <span className="mono text-[.8125rem] text-[var(--ink-3)]">{pct(value / total, 1)}</span>
              </div>
              <div className="mt-1 h-1.5 w-full rounded-sm bg-[var(--surface-sunken)]">
                <div className="h-full rounded-sm bg-[var(--ink)]" style={{ width: `${(value / total) * 100}%` }} />
              </div>
            </div>
            <span className="mono text-right text-[.8125rem] text-[var(--ink-3)]">{int(value)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Q0Research() {
  const [tab, setTab] = useState('status');
  const { q0, q0Envelope, q0Error, refreshQ0, q0UpdatedAt } = useDashboard();
  const ledger = useResource(api.q0Ledger, { key: 'q0-ledger', pollMs: POLL_INTERVALS.artifacts });

  const runtime = q0?.runtime;
  const progress = q0?.progress;
  const gate = q0?.gate;
  const thermal = q0?.thermal;
  const supervisor = q0?.supervisor?.available ? q0.supervisor : null;
  const gpu = q0?.gpu;
  const phases = q0?.phases;
  const setup = q0?.setup;
  const typed = ledger.data;

  return (
    <>
      <PageHeader
        eyebrow="Q0 / LLM research"
        title="Zero-shot gate telemetry"
        description="Read live from the runner heartbeat, the append-only rating ledger and the recorded gate verdict. This page never starts, stops or resumes the run."
        actions={
          <>
            <Badge tone={runtime?.status === 'live' ? 'badge-solid' : 'badge-alert'}>{runtime?.status === 'live' ? 'LIVE' : 'STALE'}</Badge>
            <SourceTag envelope={q0Envelope} />
            <Button size="sm" variant="quiet" icon={Activity} onClick={refreshQ0}>
              Poll now
            </Button>
          </>
        }
      />

      <div className="mb-6">
        <Segmented label="Section" options={TABS} value={tab} onChange={setTab} />
      </div>

      {tab === 'live' && <Q0LiveTab />}
      {tab === 'analytics' && <Q0AnalyticsTab />}
      {tab === 'cases' && <Q0CasesTab />}

      {tab === 'status' && q0Error && (
        <div className="mb-5">
          <ErrorNote error={q0Error} label="Q0 telemetry unavailable" onRetry={refreshQ0} />
        </div>
      )}
      {tab === 'status' && q0Envelope?.warnings?.length > 0 && (
        <div className="mb-5">
          <Warnings warnings={q0Envelope.warnings} />
        </div>
      )}
      {tab === 'status' && !q0 && !q0Error && <Loading rows={6} label="Reading Q0 telemetry" />}

      {tab === 'status' && q0 && (
        <>
          <Section eyebrow="Headline" title="How far the run has got">
            <Panel
              title={runtime?.status === 'live' ? 'Runner is active' : 'Run is not active'}
              info={{
                what: 'The runner process behind the Q0 experiment, as its heartbeat last reported.',
                reads: `Polled every ${POLL_INTERVALS.q0 / 1000}s; the reading on screen is ${q0UpdatedAt ? humanDuration((Date.now() - q0UpdatedAt) / 1000) : '—'} old. Heartbeat age is compared against the runner stale limit on the right.`,
                ideal: 'Phase advancing, heartbeat a few seconds old and neither stale flag set.',
                when: 'A stale heartbeat with status live means the process stopped without clearing its state. Check the supervisor panel below.',
              }}
            >
              <div className="grid gap-6 lg:grid-cols-4">
                <KeyValue
                  columns={1}
                  items={[
                    { label: 'Phase', value: runtime?.phase ?? 'not recorded', mono: true },
                    { label: 'Complete', value: runtime?.complete == null ? '—' : runtime.complete ? 'yes' : 'no', mono: true },
                    { label: 'PID', value: runtime?.pid ?? '—', mono: true },
                    { label: 'Started', value: runtime?.run_started_at ? formatInstant(runtime.run_started_at) : '—' },
                  ]}
                />
                <KeyValue
                  columns={1}
                  items={[
                    { label: 'Heartbeat age', value: humanDuration(runtime?.heartbeat_age_seconds), mono: true },
                    { label: 'Heartbeat limit', value: `${runtime?.stale_limits?.runner_seconds ?? '—'}s`, mono: true },
                    { label: 'Runner stale', value: runtime?.runner_stale ? 'yes' : 'no', mono: true },
                    { label: 'Supervisor stale', value: runtime?.supervisor_stale ? 'yes' : 'no', mono: true },
                  ]}
                />
                <div className="lg:col-span-2">
                  <p className="eyebrow">Current case</p>
                  {runtime?.current_case ? (
                    <div className="mt-2 rounded-md border border-[var(--ink)] bg-[var(--surface-sunken)] px-3.5 py-3">
                      <p className="mono text-[.9375rem] font-semibold text-[var(--ink)]">{runtime.current_case.symbol ?? runtime.current_case.raw}</p>
                      <p className="mono mt-1 text-[.8125rem] text-[var(--ink-3)]">
                        {runtime.current_case.decision_date ?? ''} {runtime.current_case.case_hash ? `· ${runtime.current_case.case_hash}` : ''}
                      </p>
                    </div>
                  ) : (
                    <p className="mt-2 text-[.9375rem] text-[var(--ink-2)]">
                      No case in flight. The heartbeat writes <code className="mono">current_case</code> only while a case is being judged.
                    </p>
                  )}
                </div>
              </div>
            </Panel>

            <StatGrid columns={4}>
              <Stat
                label="Rated"
                value={int(progress?.rated)}
                info={{
                  what: 'Unique case ids the runner has written a rating for.',
                  reads: `Of ${int(progress?.pool_total)} cases in the pool. ${progress?.definitions?.rated ?? ''}`.trim(),
                  ideal: 'Rising towards the pool total, with no long plateau.',
                }}
              />
              <Stat
                label="Judged"
                value={int(progress?.judged)}
                info={{
                  what: 'Ratings a downstream consumer can actually use.',
                  reads: `A case is rated as soon as a vote is recorded; it is judged only when that vote parses. ${progress?.definitions?.judged ?? ''}`.trim(),
                  ideal: 'Close to rated. A widening gap means the output harness is rejecting votes it should accept.',
                }}
              />
              <Stat
                label="Harness fail rate"
                value={pct(progress?.harness_fail_rate)}
                info={{
                  what: 'Share of rated cases whose vote the output harness dropped.',
                  reads: 'Counted from the ledger against the harness verdict on each row.',
                  ideal: 'Near zero. A rise means the model output stopped matching the schema the harness expects.',
                }}
              />
              <Stat
                label="ETA"
                value={progress?.eta_hours != null ? `${num(progress.eta_hours, 1)} h` : '—'}
                info={{
                  what: 'Hours left to finish the pool at the observed rate.',
                  reads: `Average ${num(progress?.avg_case_seconds, 1)} s per case, extrapolated over the cases that remain.`,
                  ideal: 'Falling as cases complete. An ETA that climbs means the machine slowed down, not that the work grew.',
                  when: 'Written by the runner itself. The dashboard never estimates it, so a missing ETA means the runner has not written one.',
                }}
              />
            </StatGrid>
          </Section>

          <Section eyebrow="Gate" title="Did Qwen beat FinBERT">
            <div className="grid gap-6 lg:grid-cols-2">
              <Panel
                title="Recorded verdict"
                actions={<GateBadge gate={gate} />}
                info={{
                  what: 'The pre-registered comparison between the two raters, and the verdict the run recorded against it.',
                  reads: 'Δ is the agreement gap, bar (2×SE) is the margin it must clear, and n is how many cases the gate actually saw.',
                  ideal: 'Δ above the 2×SE bar, with n equal to the pool total. PASS then means the gap is larger than sampling noise.',
                  when: 'INCOMPLETE is not a FAIL: it means the gate was never reached. Read the recorded note rather than the badge.',
                }}
              >
                {gate?.available ? (
                  <>
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                      <div>
                        <p className="eyebrow">n</p>
                        <p className="kpi-sm mono mt-1">{int(gate.n)}</p>
                      </div>
                      <div>
                        <p className="eyebrow">pool</p>
                        <p className="kpi-sm mono mt-1">{int(gate.pool_total)}</p>
                      </div>
                      <div>
                        <p className="eyebrow">Δ</p>
                        <p className="kpi-sm mono mt-1">{num(gate.delta, 4)}</p>
                      </div>
                      <div>
                        <p className="eyebrow">bar (2×SE)</p>
                        <p className="kpi-sm mono mt-1">{num(gate.bar_2se, 4)}</p>
                      </div>
                    </div>
                    <div className="mt-5">
                      <KeyValue
                        items={[
                          { label: 'Qwen agreement', value: pct(gate.qwen_agreement), mono: true },
                          { label: 'FinBERT agreement', value: pct(gate.finbert_agreement), mono: true },
                          { label: 'SE of difference', value: num(gate.se_diff, 4), mono: true },
                          { label: 'Recorded pass', value: gate.recorded_pass == null ? '—' : gate.recorded_pass ? 'true' : 'false', mono: true },
                          { label: 'Complete', value: gate.complete == null ? '—' : gate.complete ? 'yes' : 'no', mono: true },
                          { label: 'Recorded at', value: gate.recorded_at ? formatInstant(gate.recorded_at) : '—' },
                        ]}
                      />
                    </div>
                    {gate.note && <p className="mt-5 text-[.875rem] text-[var(--ink-2)]">{gate.note}</p>}
                  </>
                ) : (
                  <Empty title="No gate verdict recorded" detail="results/q0_d1.json has not been written yet." />
                )}
              </Panel>

              <Panel
                title="Q-track progress"
                actions={<Badge tone={phases?.available ? 'badge-ghost' : 'badge-dashed'}>{phases?.available ? 'recorded' : 'plan only'}</Badge>}
                info={{
                  what: 'The research phases this run has to clear, in order, and the status each one recorded.',
                  reads: 'Written to the Q-track state file by the run itself, one row per phase with its status and any note. When that file is absent the order is still shown, but no phase claims a status.',
                  ideal: 'Phases completing in order. The phase marked running is the one currently consuming the GPU.',
                  when: 'Every phase reading "not recorded" means the runner has not written artifacts/state/q_track_state.json yet. The run may still be advanced: the ledger and heartbeat below are the progress that survives a missing phase file.',
                }}
              >
                {phases?.available ? (
                  <ul>
                    {(phases.phases ?? []).map((phase) => (
                      <PhaseRow key={phase.name} phase={phase} />
                    ))}
                  </ul>
                ) : phases?.phases?.length ? (
                  <>
                    <p className="muted mb-4 text-[.875rem]">
                      {phases.note ?? 'The phase state file has not been written.'} The order below is the plan; the ledger
                      coverage underneath is the progress that was actually recorded.
                    </p>
                    <ul>
                      {phases.phases.map((phase) => (
                        <PhaseRow key={phase.name} phase={phase} recorded={false} />
                      ))}
                    </ul>
                  </>
                ) : (
                  <Empty title="No phase plan" detail={phases?.note ?? 'artifacts/state/q_track_state.json is missing.'} />
                )}
              </Panel>
            </div>
          </Section>

          <Section eyebrow="Ledger" title="Rating coverage">
            <Panel
              title="Pool coverage"
              info={{
                what: 'How much of the case pool has a rating, and how many ledger rows were spent getting there.',
                reads: 'Rated counts unique case ids; ledger rows counts attempts. The two differ because a case can be retried after a harness failure.',
                ideal: 'The bar approaching the pool total with ledger rows close to rated.',
                when: 'Rated and judged are deliberately different counts: a case is rated with a malformed vote that no consumer can use.',
              }}
            >
              <ProgressBar
                label="Rated cases"
                value={progress?.rated ?? 0}
                max={progress?.pool_total ?? 0}
                detail={`${int(progress?.ledger_total)} ledger rows · attempted ${int(progress?.attempted)}`}
              />
            </Panel>

            <div className="mt-6 grid gap-6 lg:grid-cols-3">
              <Panel
                className="lg:col-span-2"
                title="Rating distribution"
                info={{
                  what: 'The shape of the ledger: how votes split, which pools they came from, why rows were dropped, and which symbols dominate.',
                  reads: 'Each bar is a share of that group, with the raw count on the right.',
                  ideal: 'Votes spread across up and down rather than collapsing onto one label. Drop reasons should be rare and named.',
                  when: 'A single drop reason taking most of the bar is a harness bug, not a data problem.',
                }}
              >
                {ledger.error ? (
                  <ErrorNote error={ledger.error} label="Ledger unavailable" onRetry={ledger.refresh} />
                ) : typed ? (
                  <div className="grid gap-6 sm:grid-cols-2">
                    <div>
                      <ShareBars title="Votes" data={typed.votes} limit={5} />
                      <div className="mt-6">
                        <ShareBars title="Pools" data={typed.pools} limit={4} />
                      </div>
                    </div>
                    <div>
                      <ShareBars title="Drop reasons" data={typed.drop_reasons} limit={5} />
                      <div className="mt-6">
                        <ShareBars title="Top symbols" data={typed.top_symbols} limit={4} />
                      </div>
                    </div>
                  </div>
                ) : (
                  <Loading rows={5} />
                )}
              </Panel>

              <div className="space-y-6">
                <Panel
                  title="Against the realised label"
                  info={{
                    what: 'How often each rater agrees with what the stock actually did.',
                    reads: 'Matched against the triple-barrier label for the same case. The disagreement rate is FinBERT versus Qwen, not either rater versus the truth.',
                    ideal: 'Both raters well above the majority-class share. Agreement near 50% on a binary label is the same as guessing.',
                  }}
                >
                  {typed ? (
                    <KeyValue
                      columns={1}
                      items={[
                        { label: 'Qwen matches label', value: pct(typed.agreement?.qwen_matches_label), mono: true },
                        { label: 'FinBERT matches label', value: pct(typed.agreement?.finbert_matches_label), mono: true },
                        { label: 'FinBERT ≠ Qwen', value: pct(typed.agreement?.finbert_qwen_disagree_rate), mono: true },
                        { label: 'Judged rows', value: int(typed.agreement?.judged_rows), mono: true },
                        { label: 'Kept for training', value: int(typed.kept_for_training), mono: true },
                      ]}
                    />
                  ) : (
                    <Loading rows={3} />
                  )}
                </Panel>

                <Panel
                  title="Ledger range"
                  info={{
                    what: 'The window the ledger covers and the basic shape of its contents.',
                    reads: 'First and last decision date, row and symbol counts, plus mean confidence, trace length and the GPU temperature at the start and end of each rating.',
                    ideal: 'A window that keeps extending forward. A frozen end date means the runner has not written a new row.',
                  }}
                >
                  {typed ? (
                    <>
                      <p className="mono text-[.9375rem] text-[var(--ink)]">
                        {formatDate(typed.range?.start)} → {formatDate(typed.range?.end)}
                      </p>
                      <p className="mt-1.5 text-[.875rem] text-[var(--ink-2)]">
                        {int(typed.rows)} rows · {int(typed.range?.symbols)} symbols · {int(typed.rated)} rated · {int(typed.judged)} judged
                      </p>
                      <div className="mt-4 space-y-2">
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[.875rem] text-[var(--ink-2)]">Qwen confidence mean</span>
                          <span className="mono text-[.875rem]">{num(typed.confidence?.qwen_conf_mean, 4)}</span>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[.875rem] text-[var(--ink-2)]">Trace length mean</span>
                          <span className="mono text-[.875rem]">{num(typed.trace?.mean_len, 1)}</span>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[.875rem] text-[var(--ink-2)]">GPU temp start → end</span>
                          <span className="mono text-[.875rem]">
                            {num(typed.gpu?.temp_start_mean, 1)} → {num(typed.gpu?.temp_end_mean, 1)} °C
                          </span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <Loading rows={3} />
                  )}
                </Panel>
              </div>
            </div>
          </Section>

          <Section eyebrow="Hardware" title="Thermals, clocks and the watchdog">
            <div className="grid gap-6 lg:grid-cols-3">
              <Panel
                title="Thermals and clocks"
                info={{
                  what: 'What the thermal guard has done to the machine while the run was going.',
                  reads: 'Current and peak temperature, the clock cap it applied, and how many times it rested, tripped, cooled down or aborted.',
                  ideal: 'Temperature well under the peak seen, rests and hard trips at zero, and an applied clock equal to the cap.',
                  when: 'A rising hard-trip count means the run is being throttled; the wall-clock figures below will reflect it.',
                }}
              >
                <KeyValue
                  columns={1}
                  items={[
                    { label: 'State', value: thermal?.state ?? '—', mono: true },
                    { label: 'Temperature', value: thermal?.temperature_c == null ? '—' : `${num(thermal.temperature_c, 1)} °C`, mono: true },
                    { label: 'Max seen', value: thermal?.max_temperature_c == null ? '—' : `${num(thermal.max_temperature_c, 1)} °C`, mono: true },
                    { label: 'Clock cap', value: thermal?.cap_mhz ? `${int(thermal.cap_mhz)} MHz` : '—', mono: true },
                    { label: 'Applied clock', value: thermal?.applied_clock_mhz ? `${int(thermal.applied_clock_mhz)} MHz` : '—', mono: true },
                    { label: 'Rests', value: int(thermal?.rests), mono: true },
                    { label: 'Hard trips', value: int(thermal?.hard_trips), mono: true },
                    { label: 'Cooldowns', value: int(thermal?.cooldowns), mono: true },
                    { label: 'Aborts', value: int(thermal?.aborts), mono: true },
                  ]}
                />
              </Panel>

              <Panel
                className="lg:col-span-2"
                title="GPU temperature samples"
                info={{
                  what: 'Temperature over the run, sampled every five seconds.',
                  reads: `One line in °C against wall-clock time in ${gpu?.timezone ?? 'local'} time. The badges under the chart give the latest, the peak and the sample count.`,
                  ideal: 'A line that plateaus below the thermal limit. Saw-tooth drops are rests triggered by the guard.',
                  when: 'Compare stamps with the heartbeat only after converting: this axis is local, the heartbeat is UTC.',
                }}
              >
                {gpu?.available && gpu.samples?.length ? (
                  <>
                    <LineSeriesChart
                      data={gpu.samples}
                      xKey="at"
                      series={[{ key: 'temp_c', label: 'GPU temp °C', width: 1.5 }]}
                      height={230}
                      yTickFormatter={(value) => num(value, 0)}
                      xTickFormatter={(value) => String(value).slice(11, 16)}
                      unit=" °C"
                    />
                    <div className="mt-4 flex flex-wrap items-center gap-2">
                      <Badge tone="badge-ghost">latest {num(gpu.latest_c, 1)} °C</Badge>
                      <Badge tone="badge-ghost">peak {num(gpu.peak_c, 1)} °C</Badge>
                      <Badge tone="badge-dashed">{gpu.samples.length} samples</Badge>
                    </div>
                  </>
                ) : (
                  <Empty title="No GPU samples" detail="logs/gpu_5s.csv is missing or empty." />
                )}
              </Panel>
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <Panel
                title="Process watchdog"
                info={{
                  what: 'The supervisor that keeps the runner alive, if the run was started under one.',
                  reads: 'Supervisor and child process ids, whether the child is still alive, the last exit code, and the age of the supervisor heartbeat.',
                  ideal: 'Child alive, exit code zero or null, heartbeat seconds old.',
                  when: 'A non-zero exit code with the child dead means the runner crashed. Started directly, there is no supervisor at all.',
                }}
              >
                {supervisor ? (
                  <KeyValue
                    columns={1}
                    items={[
                      { label: 'State', value: supervisor.state ?? '—', mono: true },
                      { label: 'Message', value: supervisor.message ?? '—' },
                      { label: 'Supervisor PID', value: supervisor.pid ?? '—', mono: true },
                      { label: 'Child PID', value: supervisor.child_pid ?? '—', mono: true },
                      { label: 'Child alive', value: supervisor.child_alive == null ? '—' : supervisor.child_alive ? 'yes' : 'no', mono: true },
                      { label: 'Last exit code', value: supervisor.last_exit_code ?? '—', mono: true },
                      { label: 'Heartbeat age', value: humanDuration(supervisor.age_seconds), mono: true },
                    ]}
                  />
                ) : (
                  <Empty title="No supervisor heartbeat" detail="logs/q0_supervisor.json is missing: the run was started directly." />
                )}
              </Panel>

              <Panel
                title="Model and context"
                info={{
                  what: 'What the runner was configured with when it started.',
                  reads: 'Checkpoint and quantization, the pinned context size, the output-token floor, and whether setup had to correct any fields.',
                  ideal: 'The quantized model with a pinned context, so timings between phases stay comparable.',
                }}
              >
                {setup?.available ? (
                  <>
                    <KeyValue
                      columns={1}
                      items={[
                        { label: 'Model', value: setup.model ?? '—' },
                        { label: 'Quantization', value: setup.quantization ?? '—' },
                        { label: 'Context pinned', value: setup.ctx_pinned == null ? '—' : String(setup.ctx_pinned), mono: true },
                        { label: 'num_predict min', value: setup.num_predict_min ?? '—', mono: true },
                        { label: 'Status', value: setup.status ?? '—' },
                      ]}
                    />
                    {setup.corrected_fields && (
                      <p className="mt-4 text-[.875rem] text-[var(--ink-2)]">Fields corrected after setup: {String(setup.corrected_fields)}</p>
                    )}
                  </>
                ) : (
                  <Empty title="Setup state not recorded" />
                )}
              </Panel>
            </div>
          </Section>

          <Section eyebrow="Timing" title="Wall clock">
            <Panel
              title="Measured wall clock"
              info={{
                what: 'How long the run has actually spent working, against the proactive target it was given.',
                reads: 'Work seconds is everything the runner did; model seconds is only the time inside the model call. The last checkpoint is when those counters were written.',
                ideal: 'Work close to model seconds, so little time is lost to overhead, and a checkpoint that keeps moving.',
                when: 'A large gap between work and model seconds points at I/O or harness time, not the GPU.',
              }}
            >
              <KeyValue
                columns={4}
                items={[
                  { label: 'Work seconds', value: humanDuration(q0?.timing?.work_seconds), mono: true },
                  { label: 'Model seconds', value: humanDuration(q0?.timing?.total_model_seconds), mono: true },
                  { label: 'Proactive target', value: humanDuration(q0?.timing?.proactive_target_seconds), mono: true },
                  { label: 'Last checkpoint', value: q0?.timing?.last_checkpoint_at ? formatInstant(q0.timing.last_checkpoint_at) : '—' },
                ]}
              />
            </Panel>
          </Section>

          <Section eyebrow="Provenance" title="Where every field came from">
            <div className="grid gap-6 lg:grid-cols-2">
              <Panel
                title="Sources"
                info={{
                  what: 'Every file the status tab reads, whether it was present, and whether it was actually used.',
                  reads: 'Path, presence, whether it contributed to the payload, the file age and its modified stamp.',
                  ideal: 'Every file the tab depends on present and used. A missing supervisor file only matters if the run was started under one.',
                }}
                flush
              >
                <Table
                  dense
                  columns={[
                    { key: 'name', header: 'Source', strong: true },
                    {
                      key: 'available',
                      header: 'Present',
                      render: (row) => <Badge tone={row.available ? 'badge-ghost' : 'badge-dashed'}>{row.available ? 'yes' : 'no'}</Badge>,
                    },
                    { key: 'used', header: 'Used', render: (row) => (row.used ? 'yes' : 'no') },
                    { key: 'age_seconds', header: 'Age', align: 'right', render: (row) => humanDuration(row.age_seconds) },
                    { key: 'modified_at', header: 'Modified', render: (row) => (row.modified_at ? formatInstant(row.modified_at) : '—') },
                  ]}
                  rows={q0.sources ?? []}
                  rowKey={(row) => row.path}
                />
              </Panel>

              <Panel
                title="Control"
                info={{
                  what: 'Why this page cannot start, stop or resume the run, and how to do it from a terminal.',
                  reads: 'The two resume commands below are the only supported way in. The ledger is append-only, so a restart resumes instead of duplicating cases.',
                  ideal: 'The run resumes from a terminal and the ledger keeps one row per case id. Phase 6 fine-tuning carries no live loss telemetry.',
                }}
              >
                <p className="text-[.9375rem] text-[var(--ink-2)]">This dashboard has no start, stop or reset endpoint. Resume a run from a terminal:</p>
                <code className="mono mt-3 block break-all rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-3 py-2 text-[.8125rem]">
                  {q0.resume?.command}
                </code>
                <code className="mono mt-2 block break-all rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-3 py-2 text-[.8125rem]">
                  {q0.resume?.alternative}
                </code>
                {q0.resume?.note && <p className="mt-3 text-[.875rem] text-[var(--ink-2)]">{q0.resume.note}</p>}
              </Panel>
            </div>
          </Section>
        </>
      )}
    </>
  );
}
