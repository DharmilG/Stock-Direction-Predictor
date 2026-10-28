import { useCallback, useEffect, useMemo, useState } from 'react';
import { FlaskConical } from 'lucide-react';
import { AppShell } from './components/AppShell';
import { DashboardProvider, useDashboard } from './state/DashboardProvider';
import { NAV_SECTIONS, ROUTES } from './lib/constants';
import Overview from './pages/Overview';
import Predictions from './pages/Predictions';
import StockAnalysis from './pages/StockAnalysis';
import ModelEvaluation from './pages/ModelEvaluation';
import Backtesting from './pages/Backtesting';
import Q0Research from './pages/Q0Research';
import DataIntegration from './pages/DataIntegration';

const PAGES = {
  overview: Overview,
  predictions: Predictions,
  stock: StockAnalysis,
  evaluation: ModelEvaluation,
  backtesting: Backtesting,
  q0: Q0Research,
  data: DataIntegration,
};

const TITLES = Object.fromEntries(NAV_SECTIONS.flatMap((section) => section.items.map((item) => [item.id, item.label])));

function readRoute() {
  const raw = window.location.hash.replace(/^#\/?/, '').trim();
  return ROUTES.includes(raw) ? raw : 'overview';
}

function DemoBanner() {
  const { demoMode } = useDashboard();
  if (!demoMode) return null;
  return (
    <div className="mb-4 flex items-start gap-2 border-l-[5px] border-[var(--ink)] bg-[var(--surface-sunken)] px-3 py-2">
      <FlaskConical size={14} className="mt-0.5 shrink-0" />
      <p className="text-[.8125rem] leading-relaxed text-[var(--ink-2)]">
        <strong className="font-semibold text-[var(--ink)]">Demo fallback enabled.</strong> While the API is unreachable, panels are filled from a
        bundled synthetic snapshot stamped <code className="mono">source.kind = "demo"</code>. A reachable API always wins; nothing here is a measured
        result.
      </p>
    </div>
  );
}

function Router() {
  const [route, setRoute] = useState(readRoute);

  useEffect(() => {
    const onHashChange = () => setRoute(readRoute());
    window.addEventListener('hashchange', onHashChange);
    if (!window.location.hash) window.location.replace('#overview');
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = useCallback((next) => {
    if (!ROUTES.includes(next)) return;
    if (readRoute() === next) {
      setRoute(next);
      return;
    }
    window.location.hash = next;
  }, []);

  useEffect(() => {
    document.title = `${TITLES[route] ?? 'Overview'} · Stock Direction Intelligence`;
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [route]);

  const Page = useMemo(() => PAGES[route] ?? Overview, [route]);

  return (
    <AppShell route={route} onNavigate={navigate}>
      <DemoBanner />
      <Page onNavigate={navigate} />
    </AppShell>
  );
}

export default function App() {
  return (
    <DashboardProvider>
      <Router />
    </DashboardProvider>
  );
}
