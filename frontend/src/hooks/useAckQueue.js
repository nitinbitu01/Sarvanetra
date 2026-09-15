// frontend/src/hooks/useAckQueue.js — React glue for utils/ackQueue.js.
//
// Kept separate from ackQueue.js on purpose: the queue module is framework-
// agnostic (plain functions + a pub/sub set), which is what makes it
// testable without mounting a component and reusable from any surface that
// needs it. This file is the only place that touches React.
import { useEffect, useState } from 'react';
import { getItemForAlert, subscribe } from '../utils/ackQueue';

/**
 * Live queue state for one alert's acknowledgement, or undefined if it was
 * never queued (the common case — most acks succeed on the first try and
 * never touch this module at all).
 */
export function useAckQueueItem(alertId) {
  const [item, setItem] = useState(() => getItemForAlert(alertId));

  useEffect(() => {
    setItem(getItemForAlert(alertId));
    return subscribe(() => setItem(getItemForAlert(alertId)));
  }, [alertId]);

  return item;
}
