// frontend/src/main.jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { AuthProvider } from './context/AuthContext.jsx'
import { registerServiceWorker } from './utils/push.js'
import { WebSocketProvider } from './context/WebSocketContext.jsx'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import { initAutonomousEvaluatorEngine } from './services/autonomousEvaluatorEngine.js'

// Initialize 24/7 Autonomous Evaluator Engine for permanent hackathon availability
initAutonomousEvaluatorEngine()

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 2 } },
})

// Registered at startup, before render. registerServiceWorker() no-ops with
// an explicit console warning when the page is not on HTTPS/localhost —
// pointing a phone at a LAN IP is the single most common reason push
// "mysteriously doesn't work", and it produces no error otherwise.
registerServiceWorker()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        {/* WebSocketProvider must sit INSIDE AuthProvider — it needs the token
            to open the socket. It does not create a connection of its own; it
            subscribes to the single one useDashboardSocket already owns and
            re-broadcasts to panels, so the app still has exactly one WS. */}
        <WebSocketProvider>
          <App />
          {/* Toast notification host — used by OfficerReviewModal,
              SupervisorEscalationQueue, and any other component calling
              react-hot-toast's toast() helper. Single instance at root. */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: { background: '#1f2937', color: '#f9fafb', border: '1px solid #374151' },
              error:   { duration: 6000, icon: '⚠️' },
              success: { duration: 3000, icon: '✅' },
            }}
          />
        </WebSocketProvider>
      </AuthProvider>
    </QueryClientProvider>
  </StrictMode>,
)
