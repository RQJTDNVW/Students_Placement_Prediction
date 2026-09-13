import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import type { PredictionResponse, StudentInput } from '../types/prediction.ts'

export type ThemeMode = 'dark' | 'light'
export type MotionMode = 'standard' | 'reduced'
export type DataMode = 'production' | 'demo'

interface PreferencesContextValue {
  theme: ThemeMode
  motion: MotionMode
  dataMode: DataMode
  setTheme: (value: ThemeMode) => void
  setMotion: (value: MotionMode) => void
  setDataMode: (value: DataMode) => void
}

interface ToastItem { id: number; title: string; message: string; tone: 'success' | 'error' | 'info' }
interface ToastContextValue { toasts: ToastItem[]; pushToast: (toast: Omit<ToastItem, 'id'>) => void; dismissToast: (id: number) => void }
interface SessionPrediction { input: StudentInput; result: PredictionResponse; saved: boolean }
interface SessionContextValue { currentPrediction: SessionPrediction | null; setCurrentPrediction: (value: SessionPrediction) => void; markSaved: () => void; clearPrediction: () => void }
interface ActivityContextValue { isApiLoading: boolean; startRequest: () => void; endRequest: () => void }

const PreferencesContext = createContext<PreferencesContextValue | null>(null)
const ToastContext = createContext<ToastContextValue | null>(null)
const SessionContext = createContext<SessionContextValue | null>(null)
const ActivityContext = createContext<ActivityContextValue | null>(null)

function PreferencesProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<ThemeMode>(() => (localStorage.getItem('spi-theme') as ThemeMode) || 'dark')
  const [motion, setMotion] = useState<MotionMode>(() => (localStorage.getItem('spi-motion') as MotionMode) || 'standard')
  const [dataMode, setDataMode] = useState<DataMode>(() => (localStorage.getItem('spi-data-mode') as DataMode) || 'demo')
  useEffect(() => { document.documentElement.classList.toggle('light', theme === 'light'); localStorage.setItem('spi-theme', theme) }, [theme])
  useEffect(() => { document.documentElement.dataset.motion = motion; localStorage.setItem('spi-motion', motion) }, [motion])
  useEffect(() => { localStorage.setItem('spi-data-mode', dataMode) }, [dataMode])
  const value = useMemo(() => ({ theme, motion, dataMode, setTheme, setMotion, setDataMode }), [theme, motion, dataMode])
  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>
}

function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const pushToast = (toast: Omit<ToastItem, 'id'>) => { const id = Date.now(); setToasts((current) => [...current, { ...toast, id }]); window.setTimeout(() => setToasts((current) => current.filter((item) => item.id !== id)), 5000) }
  const dismissToast = (id: number) => setToasts((current) => current.filter((item) => item.id !== id))
  return <ToastContext.Provider value={{ toasts, pushToast, dismissToast }}>{children}</ToastContext.Provider>
}

function SessionProvider({ children }: { children: ReactNode }) {
  const [currentPrediction, setCurrentPrediction] = useState<SessionPrediction | null>(null)
  const markSaved = () => setCurrentPrediction((current) => current ? { ...current, saved: true } : current)
  return <SessionContext.Provider value={{ currentPrediction, setCurrentPrediction, markSaved, clearPrediction: () => setCurrentPrediction(null) }}>{children}</SessionContext.Provider>
}

function ActivityProvider({ children }: { children: ReactNode }) {
  const [requests, setRequests] = useState(0)
  const value = useMemo(() => ({ isApiLoading: requests > 0, startRequest: () => setRequests((count) => count + 1), endRequest: () => setRequests((count) => Math.max(0, count - 1)) }), [requests])
  return <ActivityContext.Provider value={value}>{children}</ActivityContext.Provider>
}

export function AppProviders({ children }: { children: ReactNode }) {
  return <PreferencesProvider><ToastProvider><SessionProvider><ActivityProvider>{children}</ActivityProvider></SessionProvider></ToastProvider></PreferencesProvider>
}

export function usePreferences() { const context = useContext(PreferencesContext); if (!context) throw new Error('usePreferences must be used within AppProviders'); return context }
export function useToasts() { const context = useContext(ToastContext); if (!context) throw new Error('useToasts must be used within AppProviders'); return context }
export function useSessionPrediction() { const context = useContext(SessionContext); if (!context) throw new Error('useSessionPrediction must be used within AppProviders'); return context }
export function useApiActivity() { const context = useContext(ActivityContext); if (!context) throw new Error('useApiActivity must be used within AppProviders'); return context }
