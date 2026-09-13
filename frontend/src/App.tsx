import { BrowserRouter, Navigate, Outlet, Route, Routes } from 'react-router-dom'
import { AppProviders } from './context/AppContext.tsx'
import { AppShell } from './components/layout/AppShell.tsx'
import Dashboard from './pages/Dashboard.tsx'
import PredictStudent from './pages/PredictStudent.tsx'
import PredictionHistory from './pages/PredictionHistory.tsx'
import StudentAnalytics from './pages/StudentAnalytics.tsx'
import ModelPerformance from './pages/ModelPerformance.tsx'
import AboutProject from './pages/AboutProject.tsx'
import Settings from './pages/Settings.tsx'

function AppRoutes() {
  return <Routes>
    <Route element={<AppShell />}>
      <Route path="/" element={<Dashboard />} />
      <Route path="/predict" element={<PredictStudent />} />
      <Route path="/predict/result" element={<PredictStudent />} />
      <Route path="/history" element={<PredictionHistory />} />
      <Route path="/analytics" element={<StudentAnalytics />} />
      <Route path="/model-performance" element={<ModelPerformance />} />
      <Route path="/about" element={<AboutProject />} />
      <Route path="/settings" element={<Settings />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Route>
  </Routes>
}

export default function App() {
  return <BrowserRouter><AppProviders><AppRoutes /></AppProviders></BrowserRouter>
}

export function AppOutlet() { return <Outlet /> }
