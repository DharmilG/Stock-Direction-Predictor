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
  gate: { available: true, verdict: 'INCOMPLETE', recorded_pass: true, passed: null, complete: false, n: 97, pool_total: 1500, qwen_agreement: 0.2817, finbert_agreement: 0.1991, delta: 0.0515, se_diff: 0.0631, bar_2se: 0.1262, harness_fail_rate: 0.0094, rated_total: 716, recorded_at: '2026-09-30T09:12:00+00:00', state_status: 'running', note: 'Gate pass/fail is copied verbatim from results/q0_d1.json; nothing is recomputed here.' },
  phases: { available: false, phases: ['phase3', 'phase6_probe', 'phase4', 'phase5', 'phase6_train', 'phase7', 'phase8'].map((name) => ({ name, status: 'pending', updated_at: null, note: null })), note: 'artifacts/state/q_track_state.json has not been written.' },
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
  if (path.endsWith('/dashboard/q0/status')) return env(Q0_STATUS);
  if (path.endsWith('/dashboard/q0/ledger-summary')) {
    return env({ rows: 716, rated: 716, judged: 652, harness_fail: 7, harness_fail_rate: 0.0098, judged_rate: 0.9106, kept_for_training: 611, pools: { C: 640, B: 76 }, models: { qwen: 716 }, prompt_versions: { v3: 716 }, votes: { UP: 281, DOWN: 244, FLAT: 127, '(missing)': 64 }, drop_reasons: { '(missing)': 640, harness_fail: 7, truncated: 12 }, thermal_states: { nominal: 690, warm: 26 }, flags: { truncated: 12, no_news: 41, harness_fail: 7, finbert_qwen_disagree: 388 }, agreement: { qwen_matches_label: 0.2817, finbert_matches_label: 0.1991, finbert_qwen_disagree_rate: 0.5419, judged_rows: 652 }, confidence: { qwen_conf_mean: 0.7412, qwen_conf_p10: 0.51, qwen_conf_p90: 0.93, finbert_score_mean: 0.0612 }, trace: { mean_len: 412, max_len: 1904 }, gpu: { temp_start_mean: 58.2, temp_end_mean: 66.4, temp_end_max: 74 }, range: { start: '2023-01-04', end: '2026-10-27', symbols: 48 }, top_symbols: { 'TMPV.NS': 26, 'ADANIPORTS.NS': 21, 'BEL.NS': 19 } });
  }
  return null;
}
