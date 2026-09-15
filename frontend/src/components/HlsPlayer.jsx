// frontend/src/components/HlsPlayer.jsx
import Hls from 'hls.js';
import React, { useEffect, useRef } from 'react';
import { getToken } from '../utils/auth';

const API = import.meta.env.VITE_API_URL || '/api/v1';

export function HlsPlayer({ cameraId, autoPlay = true }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (!videoRef.current || !cameraId) return;

    const src = `${API}/stream/${cameraId}/output.m3u8`;
    let hlsInstance = null;

    if (Hls.isSupported()) {
      hlsInstance = new Hls({
        maxBufferLength: 6,
        xhrSetup: (xhr) => {
          const token = getToken();
          if (token) {
            xhr.setRequestHeader('Authorization', `Bearer ${token}`);
          }
        },
      });

      hlsInstance.loadSource(src);
      hlsInstance.attachMedia(videoRef.current);

      hlsInstance.on(Hls.Events.MANIFEST_PARSED, () => {
        if (autoPlay) {
          videoRef.current?.play().catch(() => {});
        }
      });

      hlsInstance.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          setTimeout(() => {
            hlsInstance?.loadSource(src);
          }, 1500);
        }
      });

    } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
      videoRef.current.src = src;
      if (autoPlay) videoRef.current.play().catch(() => {});
    } else {
      console.warn('[HlsPlayer] HLS not supported in this browser.');
    }

    return () => {
      hlsInstance?.destroy();
      if (videoRef.current) {
        videoRef.current.src = '';
      }
    };
  }, [cameraId, autoPlay]);

  return (
    <video
      ref={videoRef}
      id={`stream-${cameraId}`}
      className="hls-player"
      muted
      playsInline
      controls
      style={{
        width: '100%',
        maxHeight: 280,
        borderRadius: 8,
        background: '#000',
        objectFit: 'contain',
        border: '1px solid var(--border)',
      }}
    />
  );
}

export default HlsPlayer;
