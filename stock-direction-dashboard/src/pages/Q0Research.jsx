import { Activity, Cpu, Database, Gauge, ShieldCheck, Thermometer } from 'lucide-react';
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
  Callout,
  Empty,
  ErrorNote,
  KeyValue,
  Loading,
  PageHeader,
  Panel,
  ProgressBar,
  SourceTag,
  Stat,
  StatGrid,
  Table,
  Warnings,
} from '../components/ui';

function GateBadge({ gate }) {
  const tone = gate?.verdict === 'PASS' ? 'badge-solid' : gate?.verdict === 'FAIL' ? 'badge-alert' : 'badge-dashed';
  return <Badge tone={tone}>{gate?.verdict ?? 'UNKNOWN'}</Badge>;
}

function PhaseRow({ phase }) {
  const tone =
    phase.status === 'complete' || phase.status === 'done' ? 'badge-solid' : phase.status === 'running' ? 'badge-alert' : 'badge-dashed';
  return (
    <li className="flex items-start justify-between gap-3 border-t border-[var(--line)] py-2 first:border-t-0 first:pt-0">
      <div className="min-w-0">
        <p className="mono text-[.75rem] font-semibold text-[var(--ink)]">{phase.name}</p>
        {phase.note && <p className="muted text-[.7rem]">{phase.note}</p>}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {phase.updated_at && <span className="hidden text-[.625rem] text-[var(--ink-faint)] sm:inline">{formatAny(phase.updated_at)}</span>}
        <Badge tone={tone}>{phase.status}</Badge>
      </div>
    </li>
  );
}

function ShareBars({ title, data, limit = 6 }) {
  const entries = Object.entries(data ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);
  const total = entries.reduce((sum, [, value]) => sum + value, 0) || 1;
  if (!entries.length) return <p className="muted text-[.75rem]">No values recorded.</p>;
  return (
    <div>
      {title && <p className="eyebrow mb-1.5">{title}</p>}
      <ul className="space-y-1.5">
        {entries.map(([key, value]) => (
          <li key={key} className="grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-2">
            <div className="min-w-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="mono truncate text-[.7rem] text-[var(--ink-2)]" title={key}>
                  {titleize(key)}
                </span>
                <span className="mono text-[.65rem] text-[var(--ink-3)]">{pct(value / total, 1)}</span>
              </div>
              <div className="mt-0.5 h-1.5 w-full rounded-sm bg-[var(--surface-sunken)]">
                <div className="h-full rounded-sm bg-[var(--ink)]" style={{ width: `${(value / total) * 100}%` }} />
              </div>
            </div>
            <span className="mono text-right text-[.7rem] text-[var(--ink-3)]">{int(value)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Q0Research() {
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

      {q0Error && (
        <div className="mb-4">
          <ErrorNote error={q0Error} label="Q0 telemetry unavailable" onRetry={refreshQ0} />
        </div>
      )}
      {q0Envelope?.warnings?.length > 0 && (
        <div className="mb-4">
          <Warnings warnings={q0Envelope.warnings} />
        </div>
      )}
      {!q0 && !q0Error && <Loading rows={6} label="Reading Q0 telemetry" />}

      {q0 && (
        <>
          <Panel
            eyebrow="Run"
            title={runtime?.status === 'live' ? 'Runner is active' : 'Run is not active'}
            actions={
              <span className="text-[.6875rem] text-[var(--ink-3)]">
                polled every {POLL_INTERVALS.q0 / 1000}s · last {q0UpdatedAt ? humanDuration((Date.now() - q0UpdatedAt) / 1000) : '—'} ago
              </span>
            }
          >
            <div className="grid gap-4 lg:grid-cols-4">
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
                  <div className="mt-1 rounded-md border border-[var(--ink)] bg-[var(--surface-sunken)] px-3 py-2">
                    <p className="mono text-[.8125rem] font-semibold text-[var(--ink)]">{runtime.current_case.symbol ?? runtime.current_case.raw}</p>
                    <p className="mono mt-0.5 text-[.7rem] text-[var(--ink-3)]">
                      {runtime.current_case.decision_date ?? ''} {runtime.current_case.case_hash ? `· ${runtime.current_case.case_hash}` : ''}
                    </p>
                  </div>
                ) : (
                  <p className="muted mt-1">
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
              hint={`Unique case ids · of ${int(progress?.pool_total)} in the pool`}
              footnote={progress?.definitions?.rated}
            />
            <Stat label="Judged" value={int(progress?.judged)} hint="Usable vote, no harness failure" footnote={progress?.definitions?.judged} />
            <Stat label="Harness fail rate" value={pct(progress?.harness_fail_rate)} hint="Dropped by the output harness" />
            <Stat
              label="ETA"
              value={progress?.eta_hours != null ? `${num(progress.eta_hours, 1)} h` : '—'}
              hint={`Avg ${num(progress?.avg_case_seconds, 1)} s / case`}
              footnote="The runner writes this; the dashboard never estimates it"
            />
          </StatGrid>

          <div className="mt-4">
            <Panel eyebrow="Progress" title="Ledger coverage">
              <ProgressBar
                label="Rated cases"
                value={progress?.rated ?? 0}
                max={progress?.pool_total ?? 0}
                detail={`${int(progress?.ledger_total)} ledger rows · attempted ${int(progress?.attempted)}`}
              />
              <p className="mt-2 text-[.6875rem] text-[var(--ink-faint)]">
                Rated and judged are deliberately different counts: a case can be rated with a malformed vote that no downstream consumer can use.
              </p>
            </Panel>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <Panel
              eyebrow="Gate"
              title="Recorded verdict"
              actions={<GateBadge gate={gate} />}
            >
              {gate?.available ? (
                <>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <div>
                      <p className="eyebrow">n</p>
                      <p className="kpi-sm mono mt-0.5">{int(gate.n)}</p>
                    </div>
                    <div>
                      <p className="eyebrow">pool</p>
                      <p className="kpi-sm mono mt-0.5">{int(gate.pool_total)}</p>
                    </div>
                    <div>
                      <p className="eyebrow">Δ</p>
                      <p className="kpi-sm mono mt-0.5">{num(gate.delta, 4)}</p>
                    </div>
                    <div>
                      <p className="eyebrow">bar (2×SE)</p>
                      <p className="kpi-sm mono mt-0.5">{num(gate.bar_2se, 4)}</p>
                    </div>
                  </div>
                  <div className="mt-3">
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
                  {gate.verdict === 'INCOMPLETE' && (
                    <div className="mt-3">
                      <Callout tone="warn" title="No verdict yet">
                        The run recorded <code className="mono">complete: false</code> with n = {int(gate.n)} of {int(gate.pool_total)}. That is not a FAIL
                        — the gate was never reached. {gate.note}
                      </Callout>
                    </div>
                  )}
                  <p className="mt-3 text-[.6875rem] text-[var(--ink-faint)]">{gate.note}</p>
                </>
              ) : (
                <Empty title="No gate verdict recorded" detail="results/q0_d1.json has not been written yet." />
              )}
            </Panel>

            <Panel eyebrow="Phases" title="Q-track progress" actions={<Gauge size={14} className="text-[var(--ink-faint)]" />}>
              {phases?.available ? (
                <ul>
                  {(phases.phases ?? []).map((phase) => (
                    <PhaseRow key={phase.name} phase={phase} />
                  ))}
                </ul>
              ) : (
                <Empty title="Phase state not written" detail={phases?.note ?? 'artifacts/state/q_track_state.json is missing.'} />
              )}
            </Panel>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Panel eyebrow="Thermal" title="Thermals and clocks" actions={<Thermometer size={14} className="text-[var(--ink-faint)]" />}>
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

            <Panel className="lg:col-span-2" eyebrow="GPU" title="Temperature samples" actions={<Cpu size={14} className="text-[var(--ink-faint)]" />}>
              {gpu?.available && gpu.samples?.length ? (
                <>
                  <LineSeriesChart
                    data={gpu.samples}
                    xKey="at"
                    series={[{ key: 'temp_c', label: 'GPU temp °C', width: 1.5 }]}
                    height={200}
                    yTickFormatter={(value) => num(value, 0)}
                    xTickFormatter={(value) => String(value).slice(11, 16)}
                    unit=" °C"
                  />
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <Badge tone="badge-ghost">latest {num(gpu.latest_c, 1)} °C</Badge>
                    <Badge tone="badge-ghost">peak {num(gpu.peak_c, 1)} °C</Badge>
                    <Badge tone="badge-dashed">{gpu.samples.length} samples</Badge>
                  </div>
                  <p className="mt-2 text-[.6875rem] text-[var(--ink-faint)]">
                    Time axis is {gpu.timezone}: do not compare these stamps with UTC heartbeats.
                  </p>
                </>
              ) : (
                <Empty title="No GPU samples" detail="logs/gpu_5s.csv is missing or empty." />
              )}
            </Panel>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Panel eyebrow="Supervisor" title="Process watchdog" actions={<ShieldCheck size={14} className="text-[var(--ink-faint)]" />}>
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

            <Panel eyebrow="Setup" title="Model and context">
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
                    <div className="mt-2">
                      <Callout tone="info">Fields corrected after setup: {String(setup.corrected_fields)}</Callout>
                    </div>
                  )}
                </>
              ) : (
                <Empty title="Setup state not recorded" />
              )}
            </Panel>

            <Panel eyebrow="Timing" title="Measured wall clock">
              <KeyValue
                columns={1}
                items={[
                  { label: 'Work seconds', value: humanDuration(q0?.timing?.work_seconds), mono: true },
                  { label: 'Model seconds', value: humanDuration(q0?.timing?.total_model_seconds), mono: true },
                  { label: 'Proactive target', value: humanDuration(q0?.timing?.proactive_target_seconds), mono: true },
                  { label: 'Last checkpoint', value: q0?.timing?.last_checkpoint_at ? formatInstant(q0.timing.last_checkpoint_at) : '—' },
                ]}
              />
            </Panel>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Panel className="lg:col-span-2" eyebrow="Ledger" title="Rating distribution">
              {ledger.error ? (
                <ErrorNote error={ledger.error} label="Ledger unavailable" onRetry={ledger.refresh} />
              ) : typed ? (
                <div className="grid gap-5 sm:grid-cols-2">
                  <div>
                    <ShareBars title="Votes" data={typed.votes} limit={5} />
                    <div className="mt-4">
                      <ShareBars title="Pools" data={typed.pools} limit={4} />
                    </div>
                  </div>
                  <div>
                    <ShareBars title="Drop reasons" data={typed.drop_reasons} limit={5} />
                    <div className="mt-4">
                      <ShareBars title="Top symbols" data={typed.top_symbols} limit={4} />
                    </div>
                  </div>
                </div>
              ) : (
                <Loading rows={5} />
              )}
            </Panel>

            <div className="space-y-4">
              <Panel eyebrow="Agreement" title="Against the realised label">
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

              <Panel eyebrow="Range" title="Ledger coverage">
                {typed ? (
                  <>
                    <p className="mono text-[.8125rem] text-[var(--ink)]">
                      {formatDate(typed.range?.start)} → {formatDate(typed.range?.end)}
                    </p>
                    <p className="muted mt-1">
                      {int(typed.rows)} rows · {int(typed.range?.symbols)} symbols · {int(typed.rated)} rated · {int(typed.judged)} judged
                    </p>
                    <div className="mt-3 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="muted">Qwen confidence mean</span>
                        <span className="mono text-[.75rem]">{num(typed.confidence?.qwen_conf_mean, 4)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="muted">Trace length mean</span>
                        <span className="mono text-[.75rem]">{num(typed.trace?.mean_len, 1)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="muted">GPU temp start → end</span>
                        <span className="mono text-[.75rem]">
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

          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <Panel eyebrow="Sources" title="Where every field came from" flush actions={<Database size={14} className="text-[var(--ink-faint)]" />}>
              <Table
                dense
                columns={[
                  { key: 'name', header: 'Source', strong: true },
                  { key: 'available', header: 'Present', render: (row) => <Badge tone={row.available ? 'badge-ghost' : 'badge-dashed'}>{row.available ? 'yes' : 'no'}</Badge> },
                  { key: 'used', header: 'Used', render: (row) => (row.used ? 'yes' : 'no') },
                  { key: 'age_seconds', header: 'Age', align: 'right', render: (row) => humanDuration(row.age_seconds) },
                  { key: 'modified_at', header: 'Modified', render: (row) => (row.modified_at ? formatInstant(row.modified_at) : '—') },
                ]}
                rows={q0.sources ?? []}
                rowKey={(row) => row.path}
              />
            </Panel>

            <Panel eyebrow="Control" title="Not available here">
              <div className="space-y-2">
                <Callout tone="info" title="Read-only by design">
                  This dashboard has no start, stop or reset endpoint. A run is resumed from a terminal:
                  <code className="mono mt-1 block break-all text-[.7rem]">{q0.resume?.command}</code>
                  <code className="mono mt-1 block break-all text-[.7rem]">{q0.resume?.alternative}</code>
                </Callout>
                <p className="muted text-[.75rem]">{q0.resume?.note}</p>
                <p className="muted text-[.75rem]">
                  The ledger is append-only, so restarting a run resumes instead of duplicating cases. Phase 6 QLoRA fine-tuning is not instrumented
                  with live loss or step telemetry; this page shows gate and thermal data only.
                </p>
              </div>
            </Panel>
          </div>
        </>
      )}
    </>
  );
}
