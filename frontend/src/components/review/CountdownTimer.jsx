// frontend/src/components/review/CountdownTimer.jsx
// SVG circular countdown ring for the server-authoritative 4-second gate.
// Announces remaining time to screen readers every second via aria-live.

import { useState, useEffect, useRef } from 'react';

const RADIUS = 36;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function CountdownTimer({ duration, onComplete, isComplete }) {
  const [timeLeft, setTimeLeft] = useState(duration);
  const [started, setStarted] = useState(false);
  const intervalRef = useRef(null);
  const startTimeRef = useRef(null);

  useEffect(() => {
    if (isComplete) {
      clearInterval(intervalRef.current);
      setTimeLeft(0);
      return;
    }

    // Start the timer
    startTimeRef.current = performance.now();
    setStarted(true);

    intervalRef.current = setInterval(() => {
      const elapsed = (performance.now() - startTimeRef.current) / 1000;
      const remaining = Math.max(duration - elapsed, 0);
      setTimeLeft(remaining);

      if (remaining <= 0) {
        clearInterval(intervalRef.current);
        onComplete?.();
      }
    }, 100); // 100ms tick for smooth animation

    return () => clearInterval(intervalRef.current);
  }, [duration, isComplete, onComplete]);

  const progress = isComplete ? 1 : Math.max(0, 1 - timeLeft / duration);
  const strokeDashoffset = CIRCUMFERENCE * (1 - progress);
  const displaySeconds = Math.ceil(timeLeft);

  const ringColor = isComplete
    ? '#22c55e'  // green when done
    : timeLeft <= 1
    ? '#f59e0b'  // amber in final second
    : '#3b82f6'; // blue during gate

  return (
    <div
      className="countdown-timer"
      id="gate-timer-status"
      role="status"
      aria-live="polite"
      aria-atomic="true"
      aria-label={
        isComplete
          ? 'Attention verified — ready for decision'
          : `Verifying attention: ${displaySeconds} seconds remaining`
      }
    >
      {/* SVG Ring */}
      <div className="countdown-ring-wrapper">
        <svg
          width="96"
          height="96"
          viewBox="0 0 96 96"
          aria-hidden="true"
        >
          {/* Background track */}
          <circle
            cx="48"
            cy="48"
            r={RADIUS}
            fill="none"
            stroke="#374151"
            strokeWidth="6"
          />
          {/* Progress arc */}
          <circle
            cx="48"
            cy="48"
            r={RADIUS}
            fill="none"
            stroke={ringColor}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={strokeDashoffset}
            transform="rotate(-90 48 48)"
            style={{ transition: 'stroke-dashoffset 0.1s linear, stroke 0.3s ease' }}
          />
          {/* Centre label */}
          {isComplete ? (
            <text
              x="48"
              y="55"
              textAnchor="middle"
              fill="#22c55e"
              fontSize="22"
              fontWeight="bold"
            >
              ✓
            </text>
          ) : (
            <text
              x="48"
              y="55"
              textAnchor="middle"
              fill={ringColor}
              fontSize="20"
              fontWeight="bold"
            >
              {displaySeconds}
            </text>
          )}
        </svg>
      </div>

      {/* Status text */}
      <p className="countdown-label">
        {isComplete
          ? '✅ Ready for Decision'
          : `🔒 Verifying attention (${displaySeconds}s)…`
        }
      </p>

      {/* Hint text */}
      {isComplete && (
        <p className="countdown-hint">
          Press <kbd>1</kbd> Confirm · <kbd>2</kbd> Reject · <kbd>3</kbd> Escalate
        </p>
      )}
    </div>
  );
}
