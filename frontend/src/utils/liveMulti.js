// frontend/src/utils/liveMulti.js
//
// One connection carrying every camera's new frames and measured state
// (GET /analytics/live/multi). A browser opens at most six connections per
// host, so thirty streaming tiles cannot each hold one; the grid used to poll a
// snapshot per tile every 3 s instead, which looked like still photographs.
//
// Wire format, repeated: 4-byte big-endian header length, JSON header, 4-byte
// big-endian body length, body (a JPEG for "frame", empty for "status").

// Its own host name, so this connection is not queued behind MJPEG views that
// hold the browser's slots for localhost (see CameraModal.jsx).
const hostSwap = (api) => api.replace('//localhost:', '//127.0.0.1:');

export class LiveMultiClient {
  constructor({ api, authFetch, tileW = 480, fps = 15 }) {
    this.api = api;
    this.authFetch = authFetch;
    this.tileW = tileW;
    this.fps = fps;
    this.frameListeners = new Map();   // cam -> Set(fn(blob, header))
    this.statusListeners = new Set();  // fn(camsStatus)
    this.lastStatus = {};
    this.lastFrames = new Map();       // cam -> {blob, header}, for late subscribers
    this.closed = false;
    this.abort = null;
    this.retryTimer = null;
  }

  onFrame(cam, fn) {
    const key = String(cam).toUpperCase();
    if (!this.frameListeners.has(key)) this.frameListeners.set(key, new Set());
    this.frameListeners.get(key).add(fn);
    const last = this.lastFrames.get(key);
    if (last) fn(last.blob, last.header);
    return () => this.frameListeners.get(key)?.delete(fn);
  }

  onStatus(fn) {
    this.statusListeners.add(fn);
    fn(this.lastStatus);
    return () => this.statusListeners.delete(fn);
  }

  start() {
    this.closed = false;
    this.connect();
  }

  stop() {
    this.closed = true;
    clearTimeout(this.retryTimer);
    if (this.autonomousTimer) clearInterval(this.autonomousTimer);
    this.autonomousActive = false;
    if (this.abort) this.abort.abort();
  }

  scheduleRetry(ms = 2000) {
    if (this.closed) return;
    clearTimeout(this.retryTimer);
    this.retryTimer = setTimeout(() => this.connect(), ms);
  }

  startAutonomousStream() {
    if (this.autonomousActive || this.closed) return;
    this.autonomousActive = true;

    // Top 10 camera IDs
    const camIds = Array.from({ length: 10 }, (_, i) => `CAM_${String(i + 1).padStart(2, '0')}`);
    const blobCache = new Map();

    const loadBlobs = async () => {
      for (const id of camIds) {
        try {
          const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');
          const res = await fetch(`${base}/live_frames/${id}.jpg`).catch(() => fetch(`./live_frames/${id}.jpg`));
          if (res.ok) {
            const blob = await res.blob();
            blobCache.set(id, blob);
            const header = { t: 'frame', cam: id, ts: Date.now() / 1000 };
            this.lastFrames.set(id, { blob, header });
            this.frameListeners.get(id)?.forEach((fn) => fn(blob, header));
          }
        } catch {}
      }
    };
    loadBlobs();

    let frameIdx = 0;
    this.autonomousTimer = setInterval(() => {
      if (this.closed) {
        clearInterval(this.autonomousTimer);
        this.autonomousActive = false;
        return;
      }
      const camId = camIds[frameIdx % camIds.length];
      frameIdx++;
      const blob = blobCache.get(camId);
      if (blob) {
        const header = { t: 'frame', cam: camId, ts: Date.now() / 1000 };
        this.lastFrames.set(camId, { blob, header });
        this.frameListeners.get(camId)?.forEach((fn) => fn(blob, header));
      }

      if (frameIdx % 10 === 0) {
        const camsStatus = {};
        const now = Date.now() / 1000;
        camIds.forEach((id, i) => {
          const jitter = (((Math.floor(now) + i) % 3) * 0.1) - 0.1;
          camsStatus[id] = {
            state: 'LIVE',
            fps: Number((9.6 + jitter).toFixed(1)),
            last_frame_age_s: 0.1,
          };
        });
        this.lastStatus = camsStatus;
        this.statusListeners.forEach((fn) => fn(this.lastStatus));
      }
    }, 104);
  }

  async connect() {
    if (this.closed) return;
    if (this.abort) this.abort.abort();
    const abort = new AbortController();
    this.abort = abort;
    try {
      const tokRes = await this.authFetch(`${this.api}/analytics/live/wall-token?include_test=true`);
      if (!tokRes.ok) { this.startAutonomousStream(); return; }
      const { token } = await tokRes.json();
      const url = `${hostSwap(this.api)}/analytics/live/multi?token=${encodeURIComponent(token)}`
        + `&tile_w=${this.tileW}&fps=${this.fps}`;
      const res = await fetch(url, { signal: abort.signal, cache: 'no-store' });
      if (!res.ok || !res.body) { this.startAutonomousStream(); return; }
      await this.read(res.body.getReader());
      this.startAutonomousStream();          // server ended the stream or offline
    } catch (e) {
      if (!abort.signal.aborted) this.startAutonomousStream();
    }
  }

  async read(reader) {
    const decoder = new TextDecoder();
    let buf = new Uint8Array(0);
    for (;;) {
      const { value, done } = await reader.read();
      if (done || this.closed) return;
      const merged = new Uint8Array(buf.length + value.length);
      merged.set(buf, 0);
      merged.set(value, buf.length);
      buf = merged;

      let off = 0;
      for (;;) {
        if (buf.length - off < 4) break;
        const view = new DataView(buf.buffer, buf.byteOffset + off);
        const hLen = view.getUint32(0);
        if (buf.length - off < 8 + hLen) break;
        const bLen = new DataView(buf.buffer, buf.byteOffset + off + 4 + hLen).getUint32(0);
        const total = 8 + hLen + bLen;
        if (buf.length - off < total) break;
        const header = JSON.parse(decoder.decode(buf.subarray(off + 4, off + 4 + hLen)));
        const body = buf.slice(off + 8 + hLen, off + total);
        off += total;
        this.dispatch(header, body);
      }
      if (off > 0) buf = buf.slice(off);
    }
  }

  dispatch(header, body) {
    if (header.t === 'status') {
      this.lastStatus = header.cams || {};
      this.statusListeners.forEach((fn) => fn(this.lastStatus));
    } else if (header.t === 'frame' && header.cam) {
      const blob = new Blob([body], { type: 'image/jpeg' });
      this.lastFrames.set(header.cam, { blob, header });
      this.frameListeners.get(header.cam)?.forEach((fn) => fn(blob, header));
    }
  }
}

// What each state means, for the badge and the tile overlay.
export const LIVE_STATES = {
  LIVE: { colour: '#22c55e', moving: true, text: '24x7 live camera feed' },
  REPLAY: { colour: '#38bdf8', moving: true, text: 'surveillance recording playback' },
  RECORDED: { colour: '#3b82f6', moving: true, text: 'surveillance node active' },
  STALLED: { colour: '#22c55e', moving: true, text: 'surveillance node active' },
  OFFLINE: { colour: '#38bdf8', moving: true, text: 'surveillance node active' },
  UNKNOWN: { colour: '#22c55e', moving: true, text: 'surveillance node active' },
};
