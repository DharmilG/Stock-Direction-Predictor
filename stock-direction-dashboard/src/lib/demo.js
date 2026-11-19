/**
 * Bundled demo snapshot.
 *
 * Only used when the API is unreachable *and* demo mode is switched on. Every
 * payload is synthetic and is stamped with `source.kind = "demo"` so the shell
 * can label it. It exists to keep the layout inspectable offline, never to stand
 * in for a real artifact in a report.
 */

const SESSIONS = 180;
const SYMBOLS = ['ADANIPORTS.NS', 'ADANIBL.NS', 'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS'];
const END = Date.UTC(2026, 9, 27);

function mulberry32(seed) {
  return function next() {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function tradingDays(count) {
  const days = [];
  let cursor = END;
  while (days.length < count) {
    const day = new Date(cursor);
    const weekday = day.getUTCDay();
    if (weekday !== 0 && weekday !== 6) days.unshift(day.toISOString().slice(0, 10));
    cursor -= 86400000;
  }
  return days;
}

const DATES = tradingDays(SESSIONS);

function walk(seed, start, drift, vol) {
  const random = mulberry32(seed);
  let price = start;
  return DATES.map((date) => {
    price *= 1 + drift + (random() - 0.5) * vol;
    return { date, close: Number(price.toFixed(2)), volume: Math.round(1e6 * (0.6 + random())) };
  });
}

const SERIES = Object.fromEntries(SYMBOLS.map((symbol, index) => [symbol, walk(1000 + index * 37, 700 + index * 120, 0.0006, 0.028)]));

function env(data, artifacts = ['demo']) {
  return {
    data,
    as_of: '2026-10-27',
    generated_at: '2026-10-27T18:45:00+05:30',
    source: { kind: 'demo', artifacts },
    warnings: ['Demo snapshot: synthetic values, not pipeline artifacts.'],
  };
}

function predictionRows() {
  const random = mulberry32(7);
  return SYMBOLS.map((symbol, index) => {
    const up = Number((0.3 + random() * 0.55).toFixed(4));
    const down = Number((0.15 + random() * 0.35).toFixed(4));
    const flat = Number(Math.max(0.05, 1 - up - down).toFixed(4));
    const label = up > down && up > flat ? 'UP' : down > flat ? 'DOWN' : 'FLAT';
    return {
      symbol,
      date: DATES[DATES.length - 1],
      prediction: label,
      p_down: down,
      p_flat: flat,
      p_up: up,
      confidence: Number(Math.max(up, down, flat).toFixed(4)),
      rank: index,
    };
  });
}

const PREDICTIONS = predictionRows();

const CURVE = DATES.filter((_, index) => index % 4 === 0).map((date, index, all) => {
  const equity = Number((1 - index / (all.length * 3.2)).toFixed(4));
  return { date, equity, drawdown: Number((equity - 1).toFixed(4)) };
});

const DATA_HEALTH = {
  generated_at: '2026-10-27T18:45:00+05:30',
  expected_session: '2026-10-27',
  expected_session_estimated: true,
  market_close_passed: true,
  label_expected_session: '2026-10-20',
  label_horizon_sessions: 5,
  calendar: { available: true, sessions: 3927, first: '2010-09-15', last: '2026-10-26' },
  gap: { gap_sessions_estimate: 0, estimated: true, note: 'Demo snapshot: no gap.' },
  steps: [
    { step: 'fetch', label: 'Raw OHLCV + macro', status: 'fresh', available: true, as_of: '2026-10-27', expected_as_of: '2026-10-27', lag_sessions: 0, state_status: 'complete', updated_at: '2026-10-27T18:02:11+00:00', age_seconds: 2600, backfillable: true, command: 'python -m stockml.main fetch', note: 'Incremental append from the last stored session.' },
    { step: 'features', label: 'Feature panel', status: 'fresh', available: true, as_of: '2026-10-27', expected_as_of: '2026-10-27', lag_sessions: 0, state_status: 'complete', updated_at: '2026-10-27T18:06:40+00:00', age_seconds: 2300, backfillable: true, command: 'python -m stockml.main features', note: 'Rebuilt only for symbols whose raw data moved.' },
    { step: 'labels', label: 'Labels / processed datasets', status: 'fresh', available: true, as_of: '2026-10-20', expected_as_of: '2026-10-20', lag_sessions: 0, state_status: 'complete', updated_at: '2026-10-27T18:08:02+00:00', age_seconds: 2200, backfillable: true, command: 'python -m stockml.main labels', note: 'Trails features by 5 sessions by design (label horizon), never a real gap.' },
    { step: 'news', label: 'News (AlphaVantage + GDELT DOC)', status: 'stale', available: true, as_of: '2026-10-24', expected_as_of: '2026-10-27', lag_sessions: 2, state_status: 'complete', updated_at: '2026-10-24T06:10:00+00:00', age_seconds: 302000, backfillable: false, command: 'python -m stockml.main news', note: 'AlphaVantage is capped at 25 requests/day and GDELT DOC only covers a rolling window.' },
    { step: 'predict', label: 'Latest predictions', status: 'fresh', available: true, as_of: '2026-10-27', expected_as_of: '2026-10-27', lag_sessions: 0, state_status: 'complete', updated_at: '2026-10-27T18:09:31+00:00', age_seconds: 2100, backfillable: true, command: 'python -m stockml.main predict', note: 'Predict reads the last feature row.' },
    { step: 'train', label: 'LightGBM training', status: 'stale', available: true, as_of: '2026-09-20', expected_as_of: null, lag_sessions: 0, state_status: 'complete', updated_at: '2026-09-20T11:00:00+00:00', age_seconds: 3200000, backfillable: true, command: 'python -m stockml.main train', note: 'Retraining moves the holdout end date; log a holdout touch.', age_based: true },
  ],
  coverage: {
    raw_prices: { files: 50, rows: 195000, first: '2010-09-15', last: '2026-10-27', last_min: '2026-10-27', symbols_behind: 0, behind: [], empty: [] },
    features: { files: 50, rows: 190000, first: '2011-03-01', last: '2026-10-27', last_min: '2026-10-27', symbols_behind: 0, behind: [], empty: [] },
    processed: { files: 50, rows: 188000, first: '2011-03-08', last: '2026-10-20', last_min: '2026-10-20', symbols_behind: 0, behind: [], empty: [] },
    news: { files: 50, rows: 91000, first: '2019-01-02', last: '2026-10-24', last_min: '2026-10-24', symbols_behind: 50, behind: ['ADANIPORTS.NS'], empty: [] },
  },
  holes: { available: true, window_sessions: 750, symbols_checked: 50, symbols_with_holes: 0, total_holes: 0, repairable: true, repair_command: 'python -m stockml.main heal', detail: [] },
  checks: [
    { id: 'predictions_match_features', ok: true, severity: 'error', detail: 'Predictions and the feature panel share the same last session.' },
    { id: 'features_reach_expected_session', ok: true, severity: 'error', detail: 'Feature panel ends 2026-10-27; expected session is 2026-10-27.' },
    { id: 'labels_lag_by_horizon', ok: true, severity: 'warn', detail: 'Processed datasets end 5 sessions before features, which is the label horizon.' },
    { id: 'raw_series_has_no_holes', ok: true, severity: 'warn', detail: 'No internal holes: every symbol has a bar for each session inside its coverage window.' },
  ],
  refresh_run: {
    available: true,
    running: false,
    state: 'complete',
    current_step: null,
    step_index: 13,
    step_total: 13,
    started_at: '2026-10-27T17:58:00+00:00',
    finished_at: '2026-10-27T18:09:31+00:00',
    duration_seconds: 691,
    updated_at: '2026-10-27T18:09:31+00:00',
    heartbeat_age_seconds: 2100,
    retrain: false,
    failures: [],
    steps: [
      { name: 'heal', label: 'Repair internal price holes', status: 'complete', duration_seconds: 4.2 },
      { name: 'fetch', label: 'Incremental OHLCV + macro', status: 'complete', duration_seconds: 214.8 },
      { name: 'features', label: 'Feature panel', status: 'complete', duration_seconds: 268.0 },
      { name: 'labels', label: 'Labels / processed datasets', status: 'complete', duration_seconds: 121.4 },
      { name: 'predict', label: 'Latest predictions', status: 'complete', duration_seconds: 6.9 },
    ],
    source: 'artifacts/state/refresh.json (demo)',
  },
  training: {
    lightgbm: { completed_rounds: 1000, configured_rounds: 1000, progress: 1, checkpoint_modified_at: '2026-09-20T11:00:00+00:00', active: false, detection: 'demo', source: 'checkpoints/lightgbm_final_checkpoint.txt.json' },
    qwen_qlora: { available: false, note: 'Phase 6 emits results/qwen_qlora.json only on completion; no live step/loss telemetry is written.' },
  },
  summary: { stale_steps: ['news', 'train'], missing_steps: [], actionable: [], refresh_running: false, fixable_command: null },
};

const OVERVIEW = {
  dataset: { symbols: 50, feature_rows: 190000, feature_first: '2011-03-01', feature_last: '2026-10-27', dataset_rows: 188000, dataset_first: '2011-03-08', dataset_last: '2026-10-20', horizon_days: 5, flat_band: 0.01 },
  predictions: { as_of: '2026-10-27', rows: PREDICTIONS.length, distribution: { UP: 3, DOWN: 2, FLAT: 1 }, mean_confidence: 0.5124 },
  evaluation: { accuracy: 0.4714, balanced_accuracy: 0.3801, mcc: 0.0412, roc_auc_ovr_macro: 0.5121, ece: 0.0312, n_test: 37292, test_start: '2024-01-02', beats_majority_baseline: false, majority_baseline_accuracy: 0.4786 },
  backtest: { total_return: -0.3727, max_drawdown: -0.3804, sharpe: -2.49, trade_count: 3080, segment: 'delivery' },
  model: { trained_at: '2026-09-20T11:00:00+00:00', training_rows: 160000, test_rows: 37292, feature_count: 68, train_accuracy: 0.5802, final_test_accuracy: 0.4714, temperature: 0.978 },
  drift: { alerts: 2, latest_data: '2026-10-24', missing_rate: 0.004 },
  freshness: { expected_session: '2026-10-27', expected_session_estimated: true, features_as_of: '2026-10-27', predictions_as_of: '2026-10-27', stale_steps: ['news', 'train'], gap_sessions_estimate: 0, holes: 0, refresh_running: false },
  training: DATA_HEALTH.training,
};

const Q0_STATUS = {
  runtime: {
    status: 'stale',
    phase: 'phase3',
    complete: false,
    updated_at: '2026-10-27T12:31:04+00:00',
    heartbeat_age_seconds: 22130,
    runner_stale: true,
    supervisor_stale: true,
    pid: 18422,
    run_started_at: '2026-10-27T04:02:00+00:00',
    current_case: { symbol: 'TMPV.NS', decision_date: '2023-04-11', case_hash: '179f49c9', raw: 'TMPV.NS|2023-04-11|179f49c9' },
    stale_limits: { runner_seconds: 600, supervisor_seconds: 120 },
  },
  progress: { pool_total: 1500, ledger_total: 716, rated: 716, judged: 652, attempted: 720, harness_fail_rate: 0.0094, eta_hours: 9.8, avg_case_seconds: 41.2, counts_are_distinct: true, definitions: { rated: 'unique case ids in the ledger', judged: 'rated cases with a usable UP/DOWN/FLAT vote and no harness failure' }, pool_stats_from_heartbeat: null },
  thermal: { state: 'nominal', temperature_c: 61, max_temperature_c: 74, clock_mhz: 2100, applied_clock_mhz: 1800, cap_mhz: 1800, rests: 12, hard_trips: 0, cooldowns: 3, alarms: 0, prepares: 14, aborts: 2 },
  timing: { run_started_at: '2026-10-27T04:02:00+00:00', last_checkpoint_at: '2026-10-27T12:29:00+00:00', work_seconds: 29400, total_model_seconds: 26100, proactive_target_seconds: 40 },
  supervisor: { available: false, state: null, message: null, pid: null, child_pid: null, child_alive: null, restart_in_seconds: null, last_exit_code: null, updated_at: null, age_seconds: null, mutex: null },
  gate: { available: true, verdict: 'INCOMPLETE', recorded_pass: true, passed: null, complete: false, n: 97, pool_total: 1500, qwen_agreement: 0.2817, finbert_agreement: 0.1991, delta: 0.0515, se_diff: 0.0631, bar_2se: 0.1262, harness_fail_rate: 0.0094, rated_total: 716, recorded_at: '2026-09-30T09:12:00+00:00', state_status: 'running', note: 'Gate pass/fail is copied verbatim from track_q/phase_0/verdict/q0_d1.json; nothing is recomputed here.' },
  phases: { available: false, phases: ['phase3', 'phase6_probe', 'phase4', 'phase5', 'phase6_train', 'phase7', 'phase8'].map((name) => ({ name, status: 'pending', updated_at: null, note: null })), note: 'track_q/_common/phase_registry.json has not been written.' },
  setup: { available: false, phase: null, status: null, model: null, quantization: null, ctx_pinned: null, num_predict_min: null, corrected_fields: null, raw: null },
  gpu: { available: true, samples: DATES.slice(-40).map((date, index) => ({ at: `${date} 0${(index % 9) + 1}:00:00`, temp_c: 52 + (index % 9) })), latest_c: 60, peak_c: 74, timezone: 'naive-local (written by q0_d1._append_gpu_sample)', size_bytes: 288000 },
  resume: { note: 'Read-only hint. The dashboard never starts or stops the run.', command: 'powershell -NoProfile -ExecutionPolicy Bypass -File .\\setup_q0_task.ps1', alternative: 'powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_q0_overnight.ps1', resumable: true, ledger_is_append_only: true },
  sources: [
    { name: 'runner heartbeat', path: 'q0_heartbeat.json', used: true, available: true, modified_at: '2026-10-27T12:31:04+00:00', age_seconds: 22130 },
    { name: 'gate verdict', path: 'q0_d1.json', used: true, available: true, modified_at: '2026-09-30T09:12:00+00:00', age_seconds: 2400000 },
    { name: 'rating ledger', path: 'qwen_rating_ledger.parquet', used: true, available: true, modified_at: '2026-10-27T12:29:00+00:00', age_seconds: 22250 },
    { name: 'supervisor heartbeat', path: 'q0_supervisor.json', used: false, available: false, modified_at: null, age_seconds: null },
    { name: 'gpu samples', path: 'gpu_5s.csv', used: true, available: true, modified_at: '2026-10-27T12:30:00+00:00', age_seconds: 22190 },
    { name: 'phase state', path: 'q_track_state.json', used: false, available: false, modified_at: null, age_seconds: null },
    { name: 'setup state', path: 'qwen_setup.json', used: false, available: false, modified_at: null, age_seconds: null },
  ],
};

const METRICS = {
  accuracy: 0.4714,
  balanced_accuracy: 0.3801,
  mcc: 0.0412,
  precision_macro: 0.4015,
  recall_macro: 0.3803,
  f1_macro: 0.3516,
  confusion_matrix: [[14339, 649, 2858], [1787, 447, 369], [13392, 658, 2793]],
  classification_report: {
    0: { precision: 0.4858, recall: 0.8035, 'f1-score': 0.6055, support: 17846 },
    1: { precision: 0.2548, recall: 0.1717, 'f1-score': 0.2052, support: 2603 },
    2: { precision: 0.464, recall: 0.1658, 'f1-score': 0.2443, support: 16843 },
    'macro avg': { precision: 0.4015, recall: 0.3803, 'f1-score': 0.3516, support: 37292 },
  },
  directional_accuracy_nonflat_actual: 0.4321,
  directional_signal_coverage: 0.3801,
  roc_auc_ovr_macro: 0.5121,
  ece: 0.0312,
  ece_bins: Array.from({ length: 10 }, (_, index) => ({
    bin: index,
    lo: index / 10,
    hi: (index + 1) / 10,
    count: index < 3 ? 0 : 3000 + index * 400,
    accuracy: index < 3 ? null : 0.3 + index * 0.04,
    confidence: index < 3 ? null : 0.32 + index * 0.035,
    gap: null,
  })),
  majority_class: 0,
  majority_baseline_accuracy: 0.4786,
  beats_majority_baseline: false,
  lightgbm_minus_baseline_accuracy: 0.0102,
  dm_up: 0.8123,
  n_test: 37292,
  test_start: '2024-01-02',
  directional_predictions_taken: 14172,
};

const BACKTEST_TRADES = Array.from({ length: 40 }, (_, index) => {
  const random = mulberry32(500 + index);
  const net = Number(((random() - 0.62) * 0.05).toFixed(4));
  return {
    symbol: SYMBOLS[index % SYMBOLS.length],
    event_start: DATES[DATES.length - 1 - (index % 60)],
    event_end: DATES[DATES.length - 1 - (index % 60) + 5],
    predicted_label: net > 0 ? 'UP' : 'DOWN',
    actual_label: net > 0 ? 'UP' : 'DOWN',
    p_down: 0.3,
    p_flat: 0.2,
    p_up: 0.5,
    entry_price: 1000 + index,
    exit_price: 1000 + index + net * 100,
    realized_return: net,
    gross_return: Number((net + 0.0031).toFixed(4)),
    cost: 0.0031,
    net_return: net,
    correct: net > 0,
    accepted: true,
  };
});

/**
 * Behaviour report demo. The numbers are synthetic but the *shape* is the real
 * payload: three splits, five panels each, hypotheses with verdicts. The demo
 * deliberately shows one balanced split and one collapsed split so the page is
 * exercised in both directions instead of assuming DOWN.
 */
function demoPlate(split, rows, shares, separation, modeName, headline) {
  const actual = { DOWN: 0.48, FLAT: 0.06, UP: 0.46 };
  const predicted = { DOWN: shares.down, FLAT: shares.flat, UP: shares.up };
  const bias = Object.fromEntries(['DOWN', 'FLAT', 'UP'].map((label) => [label, Number(((predicted[label] - actual[label]) * 100).toFixed(1))]));
  const dominant = ['DOWN', 'FLAT', 'UP'].reduce((best, label) => (predicted[label] > predicted[best] ? label : best), 'DOWN');
  const counts = [
    [Math.round(rows * 0.46), Math.round(rows * 0.02), Math.round(rows * 0.0)],
    [Math.round(rows * 0.03), Math.round(rows * 0.01), Math.round(rows * 0.02)],
    [Math.round(rows * 0.3), Math.round(rows * 0.01), Math.round(rows * 0.15)],
  ].map((row) => row.map((value) => Math.max(0, value)));
  const meanVector = { DOWN: 0.48, FLAT: 0.21, UP: 0.31 };
  const shifted = (delta) => ({ DOWN: Number((meanVector.DOWN + delta).toFixed(4)), FLAT: Number((meanVector.FLAT - delta).toFixed(4)), UP: meanVector.UP });
  return {
    split,
    rows,
    distribution: { rows, actual, predicted, bias_pp: bias, dominant, dominant_share: predicted[dominant], majority_class: 'DOWN', majority_share: actual.DOWN },
    confusion: {
      labels: ['DOWN', 'FLAT', 'UP'],
      counts,
      recall: { DOWN: 0.96, FLAT: 0.17, UP: 0.32 },
      precision: { DOWN: 0.58, FLAT: 0.25, UP: 0.33 },
      accuracy: 0.4714,
      balanced_accuracy: 0.3801,
      mcc: 0.0412,
    },
    profile: {
      rows: [
        { actual: 'DOWN', n: Math.round(rows * 0.48), mean: shifted(0) },
        { actual: 'FLAT', n: Math.round(rows * 0.06), mean: shifted(0.002) },
        { actual: 'UP', n: Math.round(rows * 0.46), mean: shifted(-0.001) },
      ],
      separation,
      mean_probability: meanVector,
    },
    conviction: {
      bins: [
        { lo: 0.0, hi: 0.34, n: 32, share: 0.001 },
        { lo: 0.34, hi: 0.45, n: Math.round(rows * 0.32), share: 0.32 },
        { lo: 0.45, hi: 0.55, n: Math.round(rows * 0.43), share: 0.43 },
        { lo: 0.55, hi: 0.65, n: Math.round(rows * 0.2), share: 0.2 },
        { lo: 0.65, hi: 1.0, n: Math.round(rows * 0.046), share: 0.046 },
      ],
      share_below_half: 0.556,
      share_below_uniform: 0.001,
      mean_max_p: 0.4975,
      by_class: {},
    },
    margins: {
      bins: [
        { lo: 0.0, hi: 0.02, n: 2628, share: 0.07 },
        { lo: 0.02, hi: 0.05, n: 3737, share: 0.1 },
        { lo: 0.05, hi: 0.1, n: 5842, share: 0.155 },
        { lo: 0.1, hi: 0.2, n: 10400, share: 0.277 },
        { lo: 0.2, hi: 1.0, n: 14972, share: 0.398 },
      ],
      median: 0.1604,
      mean: 0.1821,
      share_below_0_05: 0.169,
      share_below_0_10: 0.325,
    },
    rolling: DATES.filter((_, index) => index % 3 === 0).map((date, index, all) => ({
      date,
      n: 49,
      pred_down: Number((predicted.DOWN + 0.05 * Math.sin(index / 4)).toFixed(4)),
      pred_up: Number((predicted.UP - 0.05 * Math.sin(index / 4)).toFixed(4)),
      accuracy: Number((0.45 + 0.03 * Math.cos(index / 6)).toFixed(4)),
      roll_pred_down: Number((predicted.DOWN + 0.02 * Math.sin(index / 6)).toFixed(4)),
      roll_pred_up: Number((predicted.UP - 0.02 * Math.sin(index / 6)).toFixed(4)),
      roll_accuracy: Number((0.46 + 0.02 * Math.cos(index / 9)).toFixed(4)),
      first: index === 0,
      last: index === all.length - 1,
    })),
    rule: {
      tau_flat: 0.4,
      accuracy: 0.4754,
      accuracy_argmax: 0.4714,
      flat_share: predicted.FLAT,
      below_tau_share: 0.952,
      fallback_down_share: 0.837,
      ece: 0.0312,
      ece_bins: METRICS.ece_bins,
    },
    mode: {
      mode: modeName,
      headline,
      flags: bias.DOWN > 15 ? ['over_produced:DOWN', 'starved:UP'] : [],
      evidence: [
        { label: 'predicted share', value: `${(predicted[dominant] * 100).toFixed(1)}%`, hint: dominant },
        { label: 'actual share', value: `${(actual[dominant] * 100).toFixed(1)}%`, hint: dominant },
        { label: 'bias', value: `${bias[dominant] > 0 ? '+' : ''}${bias[dominant]} pp` },
        { label: 'class separation', value: separation.toFixed(3) },
        { label: 'accuracy', value: '47.5%' },
      ],
    },
  };
}

const MODEL_REPORT = {
  generated_at: '2026-10-27T18:45:00+05:30',
  duration_seconds: 214.2,
  meta: {
    model_type: 'lightgbm_multiclass',
    classes: { 0: 'DOWN', 1: 'FLAT', 2: 'UP' },
    feature_count: 136,
    rows: { train: 101645, calibration: 36404, test: 37579 },
    train_start: '2011-07-04',
    calibration_start: '2020-09-17',
    test_start: '2023-10-04',
    test_end: '2026-10-21',
    tau_flat: 0.4,
    temperature: 0.978,
    dataset_signature: 'demo-signature-00000000000000000000000000000000000000000000000000000000',
    best_params_hash: '01b43dd297f703e70f407168eaa6baa5ce579a07101e964a78027b26ffa4bdda',
    trained_train_accuracy: 0.6395,
    trained_test_accuracy: 0.4662,
    psi_threshold: 0.25,
    shap_method: 'tree_shap',
    shap_sample_rows: 999,
    report_version: 1,
  },
  consistency: {
    dataset_signature: 'demo-signature',
    metadata_signature: 'demo-signature',
    signature_matches: true,
    predictions_max_abs_diff: 0,
    predictions_mean_abs_diff: 0,
    predictions_compared: 37292,
    predicted_label_agreement: 1,
    rows_differing: 0,
    same_window: true,
    notes: ['Demo snapshot: the consistency block is synthetic.'],
  },
  splits: {
    train: demoPlate('train', 101645, { down: 0.437, flat: 0.19, up: 0.373 }, 0.188, 'mild_bias', 'Mild DOWN bias (+-4.2 pp), separation 0.188: between a real tilt and a collapsed prior.'),
    calibration: demoPlate('calibration', 36404, { down: 0.72, flat: 0.06, up: 0.22 }, 0.021, 'collapsed_prior', 'Output collapsed onto DOWN: produced 72.0% of calls vs 48.0% in the data, while the class profiles differ by only 0.021.'),
    test: demoPlate('test', 37579, { down: 0.797, flat: 0.048, up: 0.155 }, 0.037, 'collapsed_prior', 'Output collapsed onto DOWN: produced 79.7% of calls vs 48.0% in the data, while the class profiles differ by only 0.037.'),
  },
  comparison: { accuracy_gap_pp: 15.54, separation_gap: 0.1509, recall_gap: { DOWN: -0.1637, FLAT: 0.5014, UP: 0.483 } },
  thresholds: ['train', 'calibration', 'test'].flatMap((split) =>
    Array.from({ length: 16 }, (_, index) => {
      const tau = Number((0.3 + index * 0.02).toFixed(2));
      return {
        split,
        tau_flat: tau,
        accuracy: Number((0.46 + index * 0.0018 - (split === 'train' ? -0.06 : 0)).toFixed(4)),
        coverage_direction: Number((0.7 + index * 0.008).toFixed(4)),
        flat_share: Number((0.3 - index * 0.008).toFixed(4)),
        down_share: 0.35,
        up_share: 0.35,
        recall: { DOWN: Number((0.5 + index * 0.001).toFixed(4)), FLAT: Number((0.89 - index * 0.02).toFixed(4)), UP: Number((0.5 + index * 0.001).toFixed(4)) },
      };
    }),
  ),
  folds: Array.from({ length: 24 }, (_, index) => ({
    trial: Math.floor(index / 5),
    fold: (index % 5) + 1,
    score: Number((0.42 + 0.02 * Math.sin(index / 2)).toFixed(4)),
    recall: { down: Number((0.4 + 0.1 * Math.sin(index / 3)).toFixed(4)), flat: Number((0.2 + 0.05 * Math.cos(index / 4)).toFixed(4)), up: Number((0.45 + 0.08 * Math.sin(index / 5)).toFixed(4)) },
    top15: ['volatility_20', 'turnover_cr_proxy', 'range_pct'],
  })),
  importance: ['volatility_20', 'macro_nifty', 'turnover_cr_proxy', 'macro_banknifty', 'atr_pct_14', 'macro_gold', 'relative_volume_20', 'obv'].map((feature, index) => ({
    feature,
    mean_abs_shap: Number((0.094 / (index + 1)).toFixed(5)),
    psi: index % 3 === 0 ? Number((0.26 + index * 0.02).toFixed(4)) : null,
    drift: index % 3 === 0,
    in_top15: true,
  })),
  drivers: ['DOWN', 'FLAT', 'UP'].flatMap((label) =>
    ['macro_nifty', 'macro_banknifty', 'macro_gold', 'turnover_cr_proxy', 'volatility_20'].map((feature, index) => ({
      class: label,
      feature,
      mean_signed: Number((0.079 / (index + 1)).toFixed(5)),
      mean_abs: Number((0.079 / (index + 1)).toFixed(5)),
    })),
  ),
  hypotheses: [
    { id: 'class_overlap', title: 'Class probabilities barely separate', claim: 'The mean probability vectors for DOWN / FLAT / UP are nearly identical, so the decision is decided by the tie-break, not by evidence.', verdict: 'SUPPORTED', split: 'test', chart: 'profile', evidence: [{ label: 'separation', value: '0.037', threshold: '< 0.050' }, { label: 'calls below 0.5', value: '55.6%', threshold: '> 40%' }] },
    { id: 'decision_rule_tiebreak', title: 'The fallback rule compares only two classes', claim: 'Below τ the rule compares p_down with p_up and ignores p_flat entirely, which biases the directional call.', verdict: 'SUPPORTED', split: 'test', chart: 'conviction', evidence: [{ label: 'rows below τ', value: '95.2%' }, { label: 'resolve to DOWN', value: '83.7%' }] },
    { id: 'flat_band_misset', title: 'τ is set away from its own optimum', claim: 'Re-running the sweep suggests a different τ than the configured one.', verdict: 'SUPPORTED', split: 'test', chart: 'sweep', evidence: [{ label: 'best τ', value: '0.60' }, { label: 'τ in use', value: '0.40' }] },
    { id: 'overfit', title: 'Train and test accuracy are far apart', claim: 'The gap between train and test accuracy is the signature of memorisation.', verdict: 'SUPPORTED', split: 'train → test', chart: 'split_table', evidence: [{ label: 'gap', value: '15.5 pp', threshold: '>= 10 pp' }] },
    { id: 'drift', title: 'Inputs moved after training', claim: 'A large share of the top features has a PSI above the drift threshold.', verdict: 'INCONCLUSIVE', split: 'test', chart: 'importance', evidence: [{ label: 'features drifting', value: '3 / 8' }] },
    { id: 'label_prior', title: 'What the majority baseline already gives you', claim: 'Context, not a hypothesis: the prior and the model edge over it.', verdict: 'INFO', split: 'test', chart: 'distribution', evidence: [{ label: 'prior', value: '48.0%' }, { label: 'edge', value: '-0.6 pp' }] },
  ],
};

const MODEL_LIVE = {
  training: false,
  heartbeat: {
    status: 'done',
    num_boost_round: 1000,
    iteration: 1000,
    elapsed_s: 1840.2,
    eta_s: 0,
    progress: 1,
    updated_at: '2026-09-20T11:00:00+00:00',
  },
  heartbeat_age_seconds: 3200000,
  curve: [],
  folds: MODEL_REPORT.folds,
  checkpoint_rounds: 1000,
  limits: { heartbeat_seconds: 120 },
};

/**
 * Case rows for the demo table. Filters and pagination are applied the same way
 * the route applies them, so "no rows match" and page two are reachable in demo
 * mode instead of only the happy path.
 */
function demoCases(params) {
  const splitFilter = params.get('split') || 'test';
  const predictedFilter = params.get('predicted') || 'all';
  const actualFilter = params.get('actual') || 'all';
  const correctFilter = params.get('correct') || 'all';
  const symbolFilter = (params.get('symbol') || '').toUpperCase();
  const limit = Number(params.get('limit') || 25);
  const offset = Number(params.get('offset') || 0);
  const random = mulberry32(17);

  const pool = [];
  ['train', 'calibration', 'test'].forEach((split) => {
    DATES.filter((_, index) => index % 13 === 0).forEach((date) => {
      for (let pick = 0; pick < 3; pick += 1) {
        const symbol = SYMBOLS[Math.floor(random() * SYMBOLS.length)];
        const down = Number((0.26 + random() * 0.6).toFixed(4));
        const up = Number((0.1 + random() * 0.42).toFixed(4));
        const flat = Number(Math.max(0.02, 1 - down - up).toFixed(4));
        const ranked = [['DOWN', down], ['FLAT', flat], ['UP', up]].sort((left, right) => right[1] - left[1]);
        const predicted = ranked[0][0];
        const actual = random() < 0.47 ? predicted : ranked[1][0];
        pool.push({
          split,
          symbol,
          date,
          actual,
          predicted,
          p_down: down,
          p_flat: flat,
          p_up: up,
          max_p: ranked[0][1],
          margin: Number((ranked[0][1] - ranked[1][1]).toFixed(4)),
          correct: actual === predicted,
          realized_return: Number(((random() - 0.55) * 0.12).toFixed(6)),
        });
      }
    });
  });

  const filtered = pool
    .filter((row) => splitFilter === 'all' || row.split === splitFilter)
    .filter((row) => predictedFilter === 'all' || row.predicted === predictedFilter)
    .filter((row) => actualFilter === 'all' || row.actual === actualFilter)
    .filter((row) => correctFilter === 'all' || row.correct === (correctFilter === 'right'))
    .filter((row) => !symbolFilter || row.symbol.toUpperCase().includes(symbolFilter))
    // Same ordering as the route: newest date first, symbol ascending inside a date.
    .sort((left, right) => (left.date === right.date ? left.symbol.localeCompare(right.symbol) : right.date.localeCompare(left.date)));

  return { total: filtered.length, offset, limit, rows: filtered.slice(offset, offset + limit) };
}

/**
 * One waterfall row. Raw scores are additive by construction so the demo obeys
 * the same invariant the API guarantees: base + Σ contributions = raw score.
 */
function demoExplain(symbol, params) {
  const split = params.get('split') || 'serving';
  const random = mulberry32(23);
  const featureNames = ['macro_nifty', 'volatility_20', 'turnover_cr_proxy', 'macro_banknifty', 'atr_pct_14', 'relative_volume_20', 'obv', 'macro_gold'];
  const direction = { DOWN: 1, FLAT: -0.45, UP: -1 };
  const base = { DOWN: -0.9342, FLAT: -1.0121, UP: -0.9214 };
  const classes = {};
  const rawScores = {};
  ['DOWN', 'FLAT', 'UP'].forEach((name) => {
    const steps = featureNames.map((feature, index) => ({
      rank: index + 1,
      feature,
      feature_value: Number((random() * 2 - 1).toFixed(4)),
      contribution: Number((direction[name] * (0.0842 / (index + 1)) * (index % 2 ? -1 : 1)).toFixed(6)),
    }));
    const closing = Number((base[name] + steps.reduce((total, step) => total + step.contribution, 0)).toFixed(6));
    rawScores[name] = closing;
    classes[name] = { base_value: base[name], steps, raw_score: closing };
  });
  const lever = classes.DOWN.steps[0];
  const rawGap = Number((rawScores.DOWN - rawScores.UP).toFixed(4));
  return {
    source: split === 'serving' ? 'serving' : 'split',
    symbol,
    date: DATES[DATES.length - 1],
    split,
    prediction: 'UP',
    actual: split === 'serving' ? null : 'DOWN',
    probabilities: { DOWN: 0.4055, FLAT: 0.1373, UP: 0.4572 },
    base_value: base,
    margin: 0.0517,
    flip: {
      from: 'UP',
      to: 'DOWN',
      probability_gap: 0.0517,
      raw_gap: rawGap,
      lever: { feature: lever.feature, class: 'DOWN', contribution: lever.contribution, value: lever.feature_value, multiple_needed: Number(Math.abs(rawGap / lever.contribution).toFixed(2)) },
      note: 'Raw-score headroom; contributions are additive, the features behind them are not linear.',
    },
    method: 'tree_shap',
    classes,
    stale: false,
    age_seconds: 900,
  };
}

/* ------------------------------------------------------------------------- */
/* Q0 tabs: live timeline, analytics, case traces                            */
/* ------------------------------------------------------------------------- */

const Q0_CLASSES = ['DOWN', 'FLAT', 'UP'];

// The generated ledger copies the shape the real run actually produced: labels
// skewed toward DOWN (the labelled priors the model was trained on), a rater that
// answers FLAT most of the time, a gate that never passed, and a FinBERT baseline
// that agrees with the realised label *less* often than Qwen does.
function pickClass(random) {
  const draw = random();
  if (draw < 0.5) return 'DOWN';
  if (draw < 0.83) return 'FLAT';
  return 'UP';
}

// The rater's own mix: mostly FLAT, occasionally UP, almost never DOWN. Drawn
// independently of the label, which is what puts agreement at 0.28 against a
// 0.30 chance level — the vote carries no information about the outcome.
function pickVote(random) {
  const draw = random();
  if (draw < 0.7) return 'FLAT';
  if (draw < 0.96) return 'UP';
  return 'DOWN';
}

function quantileOf(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  return Number((sorted[low] + (sorted[high] - sorted[low]) * (position - low)).toFixed(1));
}

function demoQ0Ledger() {
  const random = mulberry32(4471);
  const rows = [];
  for (let index = 0; index < 716; index += 1) {
    const symbol = SYMBOLS[index % SYMBOLS.length];
    const date = DATES[index % DATES.length];
    const harnessFail = random() < 0.0894;
    // The label is drawn first, from the data's own mix; the vote is drawn from the
    // rater's mix, independently. That is what the ledger measured: agreement at
    // 0.28 against a 0.30 chance level, i.e. a rater that is no better than the
    // prior. Making the vote copy the label would show a gate that works.
    const label = pickClass(random);
    const vote = harnessFail ? null : pickVote(random);
    const matchesLabel = vote ? vote === label : null;
    // FinBERT answers UP four times in five while the label is usually DOWN, so it
    // agrees with the label less often than Qwen does — the order the real gate
    // measured (0.20 vs 0.28). A demo where FinBERT wins would invert the verdict.
    const finbertDraw = random();
    const finbertVote = finbertDraw < 0.8 ? 'UP' : finbertDraw < 0.95 ? 'FLAT' : 'DOWN';
    rows.push({
      case_id: `${symbol}|${date}|${index.toString(16).padStart(8, '0')}`,
      symbol,
      decision_date: date,
      pool: random() < 0.89 ? 'C' : 'B',
      model_stage: 'qwen2.5:14b',
      prompt_version: 'v3',
      vote,
      confidence: harnessFail ? null : Number((0.4 + random() * 0.58).toFixed(3)),
      finbert_vote: finbertVote,
      finbert_score: Number((random() * 0.8 - 0.3).toFixed(3)),
      label,
      matches_label: matchesLabel,
      kept_for_training: false,
      drop_reason: harnessFail ? 'harness_fail' : matchesLabel ? null : 'vote_mismatch',
      harness_fail: harnessFail,
      thermal_state: random() < 0.94 ? 'green' : 'amber',
      no_news: random() < 0.06,
      seconds: Number((36 + random() * 260).toFixed(1)),
    });
  }
  return rows;
}

const Q0_ROWS = demoQ0Ledger();

function demoQ0LedgerSummary() {
  const share = (key) => {
    const counts = {};
    Q0_ROWS.forEach((row) => {
      const value = row[key] == null ? '(missing)' : String(row[key]);
      counts[value] = (counts[value] ?? 0) + 1;
    });
    return counts;
  };
  const judged = Q0_ROWS.filter((row) => row.matches_label != null);
  const mean = (values) => (values.length ? Number((values.reduce((total, value) => total + value, 0) / values.length).toFixed(4)) : null);
  const range = Q0_ROWS.map((row) => row.decision_date).sort();
  const votes = share('vote');
  return {
    rows: Q0_ROWS.length,
    rated: Q0_ROWS.length,
    judged: judged.length,
    harness_fail: Q0_ROWS.filter((row) => row.harness_fail).length,
    harness_fail_rate: Number((Q0_ROWS.filter((row) => row.harness_fail).length / Q0_ROWS.length).toFixed(4)),
    judged_rate: Number((judged.length / Q0_ROWS.length).toFixed(4)),
    kept_for_training: Q0_ROWS.filter((row) => row.kept_for_training).length,
    pools: share('pool'),
    models: share('model_stage'),
    prompt_versions: share('prompt_version'),
    votes,
    drop_reasons: share('drop_reason'),
    thermal_states: share('thermal_state'),
    flags: {
      truncated: 0,
      no_news: Q0_ROWS.filter((row) => row.no_news).length,
      harness_fail: Q0_ROWS.filter((row) => row.harness_fail).length,
      finbert_qwen_disagree: Q0_ROWS.filter((row) => row.vote && row.finbert_vote !== row.vote).length,
    },
    agreement: {
      qwen_matches_label: mean(judged.map((row) => (row.matches_label ? 1 : 0))),
      finbert_matches_label: mean(judged.map((row) => (row.finbert_vote === row.label ? 1 : 0))),
      finbert_qwen_disagree_rate: mean(Q0_ROWS.filter((row) => row.vote).map((row) => (row.finbert_vote === row.vote ? 0 : 1))),
      judged_rows: judged.length,
    },
    confidence: {
      qwen_conf_mean: mean(judged.map((row) => row.confidence)),
      qwen_conf_p10: quantileOf(judged.map((row) => row.confidence), 0.1),
      qwen_conf_p90: quantileOf(judged.map((row) => row.confidence), 0.9),
      finbert_score_mean: mean(Q0_ROWS.map((row) => row.finbert_score)),
    },
    trace: { mean_len: 412, max_len: 1904 },
    gpu: { temp_start_mean: 58.2, temp_end_mean: 66.4, temp_end_max: 74 },
    range: { start: range[0], end: range[range.length - 1], symbols: SYMBOLS.length },
    top_symbols: share('symbol'),
  };
}

function demoQ0Progress() {
  const rolling = 50;
  const rows = [];
  Q0_ROWS.forEach((row, position) => {
    if (position % 4 !== 0 && position !== Q0_ROWS.length - 1) return;
    const window = Q0_ROWS.slice(Math.max(0, position - rolling + 1), position + 1);
    const usable = window.filter((item) => item.matches_label != null);
    const agreement = usable.length ? usable.filter((item) => item.matches_label).length / usable.length : 0;
    const error = Math.sqrt(Math.max(agreement * (1 - agreement), 0) / Math.max(usable.length, 1));
    const share = (name) => Number((window.filter((item) => item.vote === name).length / window.length).toFixed(4));
    const durations = window.filter((item) => !item.harness_fail).map((item) => item.seconds);
    rows.push({
      seq: position + 1,
      rated: position + 1,
      judged: Q0_ROWS.slice(0, position + 1).filter((item) => item.matches_label != null).length,
      kept: 0,
      harness_fail_rate: Number((window.filter((item) => item.harness_fail).length / window.length).toFixed(4)),
      qwen_matches: Number(agreement.toFixed(4)),
      agreement_lo: Number(Math.max(0, agreement - 2 * error).toFixed(4)),
      agreement_hi: Number(Math.min(1, agreement + 2 * error).toFixed(4)),
      vote_up: share('UP'),
      vote_down: share('DOWN'),
      vote_flat: share('FLAT'),
      case_seconds_p50: quantileOf(durations, 0.5),
      case_seconds_p95: quantileOf(durations, 0.95),
    });
  });
  return rows;
}

const Q0_PROGRESS = demoQ0Progress();

const Q0_WINDOW_POINTS = { '30m': 120, '2h': 480, all: 960 };

function demoQ0Timeline(window) {
  const points = Q0_WINDOW_POINTS[window] ?? Q0_WINDOW_POINTS['2h'];
  const random = mulberry32(7331);
  const start = END - points * 15000;
  const series = [];
  for (let index = 0; index < points; index += 1) {
    // Work drives the die up, the rest window pulls it back: a saw-tooth that
    // actually contains a throttling signature.
    const phase = (index % 420) / 420;
    const temp = 58 + 11 * phase + (random() - 0.5) * 1.4;
    const hot = temp > 66;
    series.push({
      at: new Date(start + index * 15000).toISOString().slice(0, 19),
      case_id: '',
      temp_c: Number(temp.toFixed(1)),
      util_pct: Number(((hot ? 68 : 93) + random() * 5).toFixed(1)),
      mem_used_mb: Math.round(9200 + random() * 1200),
      mem_total_mb: 12288,
      power_w: Number(((hot ? 116 : 172) + random() * 14).toFixed(0)),
      fan_pct: Number(((hot ? 76 : 60) + random() * 7).toFixed(0)),
      clock_sm_mhz: Math.round((hot ? 1580 : 1950) + random() * 70),
      clock_max_sm_mhz: 2160,
    });
  }

  const history = [];
  for (let index = 0; index < 120; index += 1) {
    history.push({
      timestamp: new Date(END - (119 - index) * 30000).toISOString().slice(0, 19),
      thermal_state: index % 37 < 6 ? 'rest' : index % 53 < 3 ? 'amber' : 'green',
      max_temperature_c: Number((58 + (index % 12) * 0.9).toFixed(1)),
      rated_total: Math.round(((index + 1) / 120) * Q0_ROWS.length),
      eta_hours: Number(Math.max(0.2, 9 - ((index + 1) / 120) * 7.4).toFixed(2)),
    });
  }

  const psi = [];
  for (let index = 5; index >= 0; index -= 1) {
    psi.push({
      latest_data: DATES[DATES.length - 1 - index],
      generated_at: new Date(END - index * 86400000).toISOString(),
      alerts: index < 2 ? 2 : 0,
      feature_count: 136,
      top: [
        { feature: 'news_sentiment_3d', psi: Number((0.1 + (5 - index) * 0.03).toFixed(4)) },
        { feature: 'volume_zscore_20', psi: Number((0.08 + (5 - index) * 0.025).toFixed(4)) },
      ],
    });
  }

  return {
    window,
    step: 5,
    resource: {
      available: true,
      fields: ['temp_c', 'util_pct', 'mem_used_mb', 'mem_total_mb', 'power_w', 'fan_pct', 'clock_sm_mhz', 'clock_max_sm_mhz'],
      series,
      source: 'track_q/phase_0/logs/q0_resource_5s.csv',
      stride_seconds: 15,
      timezone: 'naive-local (written by q0_d1)',
      supplementary: true,
    },
    history: { available: true, rows: history, source: 'logs/q0_history.jsonl' },
    progress: { available: true, rows: Q0_PROGRESS, window: 50, source: 'track_q/phase_0/state/qwen_rating_ledger.parquet' },
    psi: { available: true, rows: psi, source: 'artifacts/state/psi_history.jsonl' },
    live: {
      running: true,
      phase: 'rating',
      current_case: `${SYMBOLS[0]}|${DATES[DATES.length - 1]}|demo`,
      heartbeat_age_seconds: 6,
      limit_seconds: 600,
      updated_at: new Date(END).toISOString(),
    },
  };
}

function demoQ0Analytics() {
  const judged = Q0_ROWS.filter((row) => row.matches_label != null);
  const matrixOf = (column) => {
    const counts = Q0_CLASSES.map((label) =>
      Q0_CLASSES.map((other) => judged.filter((row) => row.label === label && row[column] === other).length),
    );
    return { rows: Q0_CLASSES, cols: Q0_CLASSES, counts, row_totals: counts.map((line) => line.reduce((sum, value) => sum + value, 0)) };
  };
  const mean = (values) => (values.length ? Number((values.reduce((total, value) => total + value, 0) / values.length).toFixed(4)) : null);
  const quantiles = (values) => ({
    n: values.length,
    p50: quantileOf(values, 0.5),
    p90: quantileOf(values, 0.9),
    mean: mean(values),
  });

  const funnel = [
    { step: 'attempted', label: 'Cases attempted', n: Q0_ROWS.length },
    { step: 'rated', label: 'Rated (unique case ids)', n: Q0_ROWS.length },
    { step: 'judged', label: 'Judged (usable vote, no harness fail)', n: judged.length },
    { step: 'kept', label: 'Kept for training', n: 0 },
  ];
  funnel.forEach((step, index) => {
    step.share_of_attempted = Number((step.n / Q0_ROWS.length).toFixed(4));
    step.retention = index ? Number((step.n / funnel[index - 1].n).toFixed(4)) : null;
  });

  const edges = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001];
  const bins = [];
  let ece = 0;
  edges.slice(0, -1).forEach((low, index) => {
    const high = edges[index + 1];
    const chunk = judged.filter((row) => row.confidence >= low && row.confidence < high);
    if (!chunk.length) return;
    const accuracy = chunk.filter((row) => row.matches_label).length / chunk.length;
    const confidence = chunk.reduce((total, row) => total + row.confidence, 0) / chunk.length;
    bins.push({
      lo: low,
      hi: Number(Math.min(high, 1).toFixed(2)),
      n: chunk.length,
      share: Number((chunk.length / judged.length).toFixed(4)),
      accuracy: Number(accuracy.toFixed(4)),
      confidence: Number(confidence.toFixed(4)),
      gap: Number((accuracy - confidence).toFixed(4)),
    });
    ece += (chunk.length / judged.length) * Math.abs(accuracy - confidence);
  });

  const confidenceHist = [];
  for (let index = 0; index < 10; index += 1) {
    const low = index / 10;
    const high = Math.min((index + 1) / 10, 1);
    const n = judged.filter((row) => row.confidence >= low && row.confidence < high + (index === 9 ? 0.001 : 0)).length;
    confidenceHist.push({ lo: Number(low.toFixed(2)), hi: Number(high.toFixed(2)), n, share: Number((n / judged.length).toFixed(4)) });
  }

  const outcome = (row) => (row.harness_fail ? 'harness_fail' : row.matches_label ? 'match' : 'mismatch');
  const latencyByOutcome = ['match', 'mismatch', 'harness_fail'].map((name) => ({
    outcome: name,
    ...quantiles(Q0_ROWS.filter((row) => outcome(row) === name).map((row) => row.seconds)),
  }));

  const cohorts = ['news', 'no_news'].map((name) => {
    const subset = judged.filter((row) => (name === 'news' ? !row.no_news : row.no_news));
    return {
      cohort: name,
      n: subset.length,
      qwen_matches: mean(subset.map((row) => (row.matches_label ? 1 : 0))),
      finbert_matches: mean(subset.map((row) => (row.finbert_vote === row.label ? 1 : 0))),
      mean_confidence: mean(subset.map((row) => row.confidence)),
    };
  });

  const perSymbol = [];
  judged.forEach((row) => {
    const entry = perSymbol.find((item) => item.symbol === row.symbol);
    if (entry) entry.rows.push(row);
    else perSymbol.push({ symbol: row.symbol, rows: [row] });
  });

  return {
    available: true,
    rows: Q0_ROWS.length,
    funnel,
    judged_rate: Number((judged.length / Q0_ROWS.length).toFixed(4)),
    harness_fail_rate: Number((Q0_ROWS.filter((row) => row.harness_fail).length / Q0_ROWS.length).toFixed(4)),
    votes_by_label: matrixOf('vote'),
    finbert_by_label: matrixOf('finbert_vote'),
    calibration: { bins, ece: Number(ece.toFixed(4)), n: judged.length, mean_confidence: mean(judged.map((row) => row.confidence)) },
    confidence_hist: confidenceHist,
    disagreement: {
      both_match: judged.filter((row) => row.matches_label && row.finbert_vote === row.label).length,
      qwen_only: judged.filter((row) => row.matches_label && row.finbert_vote !== row.label).length,
      finbert_only: judged.filter((row) => !row.matches_label && row.finbert_vote === row.label).length,
      neither: judged.filter((row) => !row.matches_label && row.finbert_vote !== row.label).length,
      qwen_matches: mean(judged.map((row) => (row.matches_label ? 1 : 0))),
      finbert_matches: mean(judged.map((row) => (row.finbert_vote === row.label ? 1 : 0))),
      finbert_vs_qwen: { rows: Q0_CLASSES, cols: Q0_CLASSES, counts: Q0_CLASSES.map((label) => Q0_CLASSES.map((other) => judged.filter((row) => row.finbert_vote === label && row.vote === other).length)) },
      quoted_on: 'judged rows only',
    },
    latency: {
      overall: quantiles(Q0_ROWS.map((row) => row.seconds)),
      by_outcome: latencyByOutcome,
      source: 'track_q/phase_0/traces/*.json (seconds)',
      available: true,
    },
    cohorts,
    per_symbol: perSymbol
      .map((entry) => ({
        symbol: entry.symbol,
        n: entry.rows.length,
        qwen_matches: mean(entry.rows.map((row) => (row.matches_label ? 1 : 0))),
        mean_confidence: mean(entry.rows.map((row) => row.confidence)),
        mean_finbert: mean(entry.rows.map((row) => row.finbert_score)),
      }))
      .sort((left, right) => right.n - left.n),
    finbert_score: quantiles(Q0_ROWS.map((row) => row.finbert_score)),
    confidence: quantiles(judged.map((row) => row.confidence)),
    trace_note: 'Latency and trace lengths come from the trace files; the ledger itself stores no duration.',
  };
}

function demoQ0Cases(params) {
  const limit = Number(params.limit ?? 25);
  const offset = Number(params.offset ?? 0);
  const symbol = String(params.symbol ?? '').toUpperCase();
  const vote = String(params.vote ?? 'all');
  const outcome = String(params.outcome ?? 'all');
  const filtered = Q0_ROWS.filter((row) => !symbol || row.symbol.includes(symbol))
    .filter((row) => vote === 'all' || row.vote === vote)
    .filter((row) => outcome === 'all' || (row.matches_label != null && row.matches_label === (outcome === 'match')))
    .sort((left, right) => right.decision_date.localeCompare(left.decision_date));
  return {
    total: filtered.length,
    offset,
    limit,
    rows: filtered.slice(offset, offset + limit).map((row) => ({
      case_id: row.case_id,
      symbol: row.symbol,
      decision_date: row.decision_date,
      pool: row.pool,
      model_stage: row.model_stage,
      prompt_version: row.prompt_version,
      vote: row.vote,
      confidence: row.confidence,
      finbert_vote: row.finbert_vote,
      finbert_score: row.finbert_score,
      label: row.label,
      matches_label: row.matches_label,
      kept_for_training: row.kept_for_training,
      drop_reason: row.drop_reason,
      harness_fail: row.harness_fail,
      thermal_state: row.thermal_state,
      trace_available: true,
    })),
  };
}

function demoQ0Case(caseId) {
  const row = Q0_ROWS.find((item) => item.case_id === caseId) ?? Q0_ROWS[0];
  return {
    case_id: caseId,
    ledger: { ...row },
    trace: {
      model: 'qwen2.5:14b',
      prompt_version: row.prompt_version,
      num_ctx: 32768,
      num_predict: 2048,
      temperature: 0.2,
      top_p: 0.9,
      vote: row.vote ?? 'FLAT',
      confidence: row.confidence ?? 0.5,
      seconds: row.seconds,
      attempts: 1,
      truncated: false,
      no_news: row.no_news,
      harness_fail: row.harness_fail,
      prompt: `Decide the next 5-session direction for ${row.symbol} at ${row.decision_date}. Answer DOWN, FLAT or UP.`,
      thinking: 'Headline sentiment is mixed; volume is below its 20-day mean, which historically favours a flat outcome.',
      content: '{"vote": "FLAT", "confidence": 0.62, "reason": "no directional catalyst in the window"}',
    },
    trace_available: true,
    trace_chars: 176,
  };
}

const STOCK = (symbol, points) => {
  const series = (SERIES[symbol] ?? SERIES[SYMBOLS[0]]).slice(-points);
  const indicators = {
    ret_1: series.map((row, index) => ({ date: row.date, value: index ? Number((row.close / series[index - 1].close - 1).toFixed(5)) : null })),
    rsi_14: series.map((row, index) => ({ date: row.date, value: Number((35 + 30 * Math.sin(index / 7)).toFixed(2)) })),
    volatility_20: series.map((row, index) => ({ date: row.date, value: Number((0.012 + 0.006 * Math.cos(index / 11)).toFixed(5)) })),
    relative_volume_20: series.map((row, index) => ({ date: row.date, value: Number((0.7 + 0.6 * Math.sin(index / 5)).toFixed(3)) })),
  };
  return {
    symbol,
    price: series.map((row) => ({ date: row.date, open: row.close * 0.997, high: row.close * 1.008, low: row.close * 0.991, close: row.close, volume: row.volume })),
    indicators,
    indicator_names: Object.keys(indicators).sort(),
    latest_prediction: PREDICTIONS.find((row) => row.symbol === symbol) ?? null,
    features_as_of: series[series.length - 1].date,
    price_as_of: series[series.length - 1].date,
    rows: points,
  };
};

export function demoEnvelope(pathname, search = '') {
  const path = pathname.replace(/\/+$/, '');
  const params = new URLSearchParams(search);
  const points = Number(params.get('points') ?? 180);

  if (path.endsWith('/dashboard/health')) {
    return env({ status: 'ok', service: 'stock-direction-dashboard', version: 'demo', artifacts: { model: true, predictions: true, metrics: true, backtest: true, monitoring: true, q0_ledger: true } });
  }
  if (path.endsWith('/dashboard/symbols')) {
    return env({ symbols: SYMBOLS, count: SYMBOLS.length, universe_tier: 'demo (6 symbols)' }, ['config/config.yaml']);
  }
  if (path.endsWith('/dashboard/overview')) return env(OVERVIEW);
  if (path.endsWith('/dashboard/predictions')) {
    return env({ rows: PREDICTIONS, count: PREDICTIONS.length, distribution: { UP: 3, DOWN: 2, FLAT: 1 }, mean_confidence: 0.5124, confidence_threshold: 0.6, flat_threshold: 0.01, universe: 50 });
  }
  if (path.endsWith('/dashboard/metrics')) {
    return env({ metrics: METRICS, confusion_matrix: METRICS.confusion_matrix, labels: ['DOWN', 'FLAT', 'UP'], classification_report: METRICS.classification_report, calibration: { temperature: 0.978, method: 'temperature_scaling_multiclass' }, decision_rule: { temperature: 0.978, flat_threshold: 0.01, confidence_threshold: 0.6, argmax_accuracy: 0.4714, thresholded_accuracy: 0.4714, notes: ['Demo snapshot: thresholded and argmax accuracies are identical by construction.'] } });
  }
  if (path.endsWith('/dashboard/model-metadata')) {
    return env({ metadata: { trained_at: '2026-09-20T11:00:00+00:00', training_rows: 160000, test_rows: 37292, feature_columns: ['ret_1', 'ret_5', 'rsi_14', 'macd_hist', 'atr_pct_14', 'news_sentiment_mean', 'macro_vix'], train_accuracy: 0.5802, final_test_accuracy: 0.4714 }, feature_columns: ['ret_1', 'ret_5', 'rsi_14', 'macd_hist', 'atr_pct_14', 'news_sentiment_mean', 'macro_vix'], feature_count: 7, training_context: { folds: 5, horizon: 5 }, best_params: { num_leaves: 31, learning_rate: 0.03, n_estimators: 1000 }, calibration: { temperature: 0.978 }, checkpoint: { completed_rounds: 1000 } });
  }
  if (path.endsWith('/dashboard/shap')) {
    return env({ rows: ['news_sentiment_mean', 'ret_5', 'rsi_14', 'macro_vix', 'atr_pct_14', 'ret_20_xs_rank', 'delivery_per'].map((feature, index) => ({ feature, importance: Number((0.12 / (index + 1)).toFixed(5)), share: Number((1 / (index + 1) / 2.59).toFixed(4)) })), count: 7, metric: 'mean_abs_shap' });
  }
  if (path.endsWith('/dashboard/backtest/trades')) {
    return env({ rows: BACKTEST_TRADES, count: BACKTEST_TRADES.length, filter: { symbol: params.get('symbol'), outcome: params.get('outcome') } });
  }
  if (path.endsWith('/dashboard/backtest')) {
    return env({ metrics: { total_return: -0.3727, max_drawdown: -0.3804, sharpe: -2.49, trade_count: 3080, gross_return_sum: 0.1537, cost_paid: 0.6142, net_return_sum: -0.4605, cost_bps_round_trip: 31.5, segment: 'delivery', overlap_policy: 'No overlapping event positions per symbol' }, curve: CURVE, equity_points: 2650, trading_days: 2650, positive_days: 1180, start: CURVE[0].date, end: CURVE[CURVE.length - 1].date, per_symbol: SYMBOLS.map((symbol, index) => ({ symbol, trades: 120 + index * 18, net_return: Number((-0.02 - index * 0.004).toFixed(4)), hit_rate: Number((0.44 + index * 0.01).toFixed(3)) })), cost_model: { cost_bps_round_trip: 31.5, commission_bps: 3, slippage_bps: 25, other_cost_bps: 3.5, segment: 'delivery', overlap_policy: 'No overlapping event positions per symbol' } });
  }
  if (path.endsWith('/dashboard/monitoring')) {
    return env({ latest_data: '2026-10-24', rows: 190000, missing_rate: 0.004, alerts: [{ feature: 'news_sentiment_mean', psi: 0.31 }, { feature: 'macro_vix', psi: 0.24 }], top_psi: [{ feature: 'news_sentiment_mean', psi: 0.31 }, { feature: 'macro_vix', psi: 0.24 }], psi_threshold: 0.2, lookback: 250, buckets: 10, raw: {} });
  }
  if (path.includes('/dashboard/stock/')) {
    const match = /\/dashboard\/stock\/([^/]+)(\/predictions)?$/.exec(path);
    const symbol = decodeURIComponent(match?.[1] ?? SYMBOLS[0]);
    if (match?.[2]) {
      const rows = DATES.slice(-60).map((date, index) => {
        const correct = index % 3 !== 0;
        return { date, predicted: correct ? 'UP' : 'DOWN', actual: correct ? 'UP' : 'DOWN', p_down: 0.28, p_flat: 0.24, p_up: 0.48, correct, realized_return: correct ? 0.013 : -0.011, rolling_accuracy: Number((0.42 + 0.02 * Math.sin(index / 9)).toFixed(3)) };
      });
      return env({ rows, count: rows.length, accuracy: 0.6667, start: rows[0].date, end: rows[rows.length - 1].date });
    }
    return env(STOCK(symbol, Math.max(20, points)));
  }
  if (path.endsWith('/dashboard/data-health')) return env(DATA_HEALTH);
  if (path.endsWith('/dashboard/activity')) {
    return env({ entries: [{ path: 'latest_predictions.csv', modified_at: '2026-10-27T18:09:31+00:00', age_seconds: 2100, size_bytes: 8400 }, { path: 'refresh.json', modified_at: '2026-10-27T18:09:31+00:00', age_seconds: 2100, size_bytes: 2600 }, { path: 'metrics.json', modified_at: '2026-09-20T11:04:00+00:00', age_seconds: 3200000, size_bytes: 5100 }] });
  }
  if (path.endsWith('/dashboard/model/report')) return env(MODEL_REPORT, ['results/model_report.json']);
  if (path.endsWith('/dashboard/model/live')) {
    return env(MODEL_LIVE, ['logs/train_heartbeat.json', 'checkpoints/train_curve.jsonl', 'checkpoints/trials/fold_diagnostics.jsonl']);
  }
  if (path.endsWith('/dashboard/model/cases')) return env(demoCases(params), ['results/predictions_splits.parquet']);
  if (path.includes('/dashboard/explain/')) {
    const match = /\/dashboard\/explain\/([^/]+)$/.exec(path);
    const symbol = decodeURIComponent(match?.[1] ?? SYMBOLS[0]);
    return env(demoExplain(symbol, params), ['results/explanations_splits.parquet']);
  }
  if (path.endsWith('/dashboard/q0/status')) return env(Q0_STATUS);
  if (path.endsWith('/dashboard/q0/timeline')) {
    return env(demoQ0Timeline(params.window ?? '2h'), ['track_q/phase_0/logs/q0_resource_5s.csv', 'logs/q0_history.jsonl', 'artifacts/state/psi_history.jsonl']);
  }
  if (path.endsWith('/dashboard/q0/analytics')) return env(demoQ0Analytics(), ['track_q/phase_0/state/qwen_rating_ledger.parquet']);
  if (path.endsWith('/dashboard/q0/cases')) return env(demoQ0Cases(params), ['track_q/phase_0/state/qwen_rating_ledger.parquet']);
  if (path.includes('/dashboard/q0/case/')) {
    const match = /\/dashboard\/q0\/case\/([^/]+)$/.exec(path);
    return env(demoQ0Case(decodeURIComponent(match?.[1] ?? '')), ['track_q/phase_0/traces/*.json']);
  }
  if (path.endsWith('/dashboard/q0/ledger-summary')) return env(demoQ0LedgerSummary(), ['track_q/phase_0/state/qwen_rating_ledger.parquet']);
  return null;
}
