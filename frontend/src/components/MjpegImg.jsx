// frontend/src/components/MjpegImg.jsx
//
// <img> for a live MJPEG (multipart/x-mixed-replace) stream that actually
// closes its connection when it goes away.
//
// Removing an <img> from the page does not reliably stop a multipart stream
// in Firefox: the response keeps downloading on the detached element. Every
// camera view closed and every tile paused leaked one open connection, and a
// browser allows only 6 per server — measured 2026-09-11, Firefox held 12 and
// then 20 connections to :8000 while the dashboard showed five tiles, after
// which new views sat black with no error (a queued request fires no
// onerror). Pointing the element at an empty data URL before it is removed
// makes the browser cancel the stream.
import { useEffect, useRef } from 'react';

export default function MjpegImg({ src, ...rest }) {
  const ref = useRef(null);
  const srcRef = useRef(src);
  srcRef.current = src;

  useEffect(() => {
    const el = ref.current;
    // Re-apply after a StrictMode mount/unmount/mount cycle, whose cleanup
    // below would otherwise leave the element blank.
    if (el && srcRef.current && el.getAttribute('src') !== srcRef.current) {
      el.src = srcRef.current;
    }
    return () => {
      if (!el) return;
      el.onerror = null;
      el.src = 'data:,';
    };
  }, []);

  return <img ref={ref} src={src} {...rest} />;
}
