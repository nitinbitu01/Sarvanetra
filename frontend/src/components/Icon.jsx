// frontend/src/components/Icon.jsx
//
// One consistent stroked icon set, inlined.
//
// WHY NOT A LIBRARY
//   lucide-react would add a dependency and a network install the day before
//   an evaluation. These are the ~20 glyphs this app actually uses, drawn on
//   the same 24px grid at the same stroke width, which is the only property
//   that made the emoji set look inconsistent in the first place — 🛰️ and 📋
//   render at different optical weights and in a different colour on every OS,
//   so no amount of CSS could line them up.
//
// Icons inherit `currentColor`, so an active nav item tints its icon with the
// same rule that tints its label.

const PATHS = {
  crosshair: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M22 12h-4M6 12H2M12 6V2M12 22v-4" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none" />
    </>
  ),
  monitor: (
    <>
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8M12 17v4" />
    </>
  ),
  scan: (
    <>
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  chart: (
    <>
      <path d="M3 3v16a2 2 0 0 0 2 2h16" />
      <path d="m7 15 4-5 3 3 5-7" />
    </>
  ),
  phone: (
    <>
      <rect x="6" y="2" width="12" height="20" rx="2" />
      <path d="M12 18h.01" />
    </>
  ),
  map: (
    <>
      <path d="M9.5 6 3.8 3.6a.7.7 0 0 0-1 .65v13.1c0 .3.17.56.44.66L9.5 20.5l5-2.5 5.26 2.24a.7.7 0 0 0 1-.65V6.5a.7.7 0 0 0-.43-.65L14.5 3.5z" />
      <path d="M9.5 6v14.5M14.5 3.5V18" />
    </>
  ),
  cameraAdd: (
    <>
      <path d="M2 8.5A1.5 1.5 0 0 1 3.5 7h2.9l1.4-2.2h5.4L14.6 7h1.9A1.5 1.5 0 0 1 18 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 2 17.5z" />
      <circle cx="10" cy="12.5" r="3" />
      <path d="M20.5 3v6M23.5 6h-6" />
    </>
  ),
  shieldAlert: (
    <>
      <path d="M12 21.5s7.5-3.7 7.5-9.5V5.2L12 2.5 4.5 5.2V12c0 5.8 7.5 9.5 7.5 9.5z" />
      <path d="M12 8.5v4M12 16h.01" />
    </>
  ),
  bell: (
    <>
      <path d="M6.5 8.5a5.5 5.5 0 0 1 11 0c0 6 2.5 8 2.5 8H4s2.5-2 2.5-8" />
      <path d="M10.3 20a1.94 1.94 0 0 0 3.4 0" />
    </>
  ),
  cpu: (
    <>
      <rect x="4.5" y="4.5" width="15" height="15" rx="2" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      <path d="M9 2v2.5M15 2v2.5M9 19.5V22M15 19.5V22M2 9h2.5M2 15h2.5M19.5 9H22M19.5 15H22" />
    </>
  ),
  searchCheck: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20.5 20.5-4.2-4.2" />
      <path d="m8.4 11 2 2 3.9-3.9" />
    </>
  ),
  scale: (
    <>
      <path d="M12 3.5V21M7.5 21h9M5 7.5l14-2" />
      <path d="m5 7.5-2.8 5.6a3.1 3.1 0 0 0 5.6 0z" />
      <path d="m19 5.5-2.8 5.6a3.1 3.1 0 0 0 5.6 0z" />
    </>
  ),
  triangleAlert: (
    <>
      <path d="M10.3 3.9 2.4 17.8A1.9 1.9 0 0 0 4.1 20.7h15.8a1.9 1.9 0 0 0 1.7-2.9L13.7 3.9a1.95 1.95 0 0 0-3.4 0z" />
      <path d="M12 9.5v4M12 17h.01" />
    </>
  ),
  route: (
    <>
      <circle cx="6" cy="19" r="2.6" />
      <circle cx="18" cy="5" r="2.6" />
      <path d="M8.7 19h8.8a3.4 3.4 0 0 0 0-6.8H6.5a3.4 3.4 0 0 1 0-6.8h8.8" />
    </>
  ),
  clipboardCheck: (
    <>
      <rect x="8.5" y="2.5" width="7" height="4" rx="1" />
      <path d="M15.5 4.5H18a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2h2.5" />
      <path d="m9 14 2.2 2.2L15.5 12" />
    </>
  ),
  list: (
    <>
      <path d="M8.5 6H21M8.5 12H21M8.5 18H21" />
      <path d="M3.5 6h.01M3.5 12h.01M3.5 18h.01" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8" r="3.6" />
      <path d="M4.5 20a7.5 7.5 0 0 1 15 0" />
    </>
  ),
  logout: (
    <>
      <path d="M14 3.5h3.5a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2H14" />
      <path d="M10 16.5 14.5 12 10 7.5M14.5 12h-11" />
    </>
  ),
  panelLeft: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M9.5 4v16" />
    </>
  ),
  camera: (
    <>
      <path d="M2 8.5A1.5 1.5 0 0 1 3.5 7h3L8 4.8h8L17.5 7h3A1.5 1.5 0 0 1 22 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-17A1.5 1.5 0 0 1 2 17.5z" />
      <circle cx="12" cy="12.8" r="3.3" />
    </>
  ),
  flame: (
    <>
      <path d="M12 2.5c2.4 3 3.6 5.2 3.6 6.7a3.6 3.6 0 0 1-7.2 0c0-.6.2-1.3.6-2C7.2 9.1 5.5 11.6 5.5 14.3a6.5 6.5 0 0 0 13 0c0-4.3-2.6-8.2-6.5-11.8z" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </>
  ),
  refresh: (
    <>
      <path d="M3 12a9 9 0 0 1 15.5-6.36L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-15.5 6.36L3 16" />
      <path d="M3 21v-5h5" />
    </>
  ),
  maximize: (
    <>
      <path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />
    </>
  ),
  minimize: (
    <>
      <path d="M4 14h6v6M20 14h-6v6M4 10h6V4M20 10h-6V4" />
    </>
  ),
  moreVertical: (
    <>
      <circle cx="12" cy="5" r="1.5" fill="currentColor" />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" />
      <circle cx="12" cy="19" r="1.5" fill="currentColor" />
    </>
  ),
  chevronDown: (
    <>
      <path d="m6 9 6 6 6-6" />
    </>
  ),
  chevronRight: (
    <>
      <path d="m9 18 6-6-6-6" />
    </>
  ),
  chevronUp: (
    <>
      <path d="m18 15-6-6-6 6" />
    </>
  ),
  pause: (
    <>
      <rect x="6" y="4" width="4" height="16" rx="1" />
      <rect x="14" y="4" width="4" height="16" rx="1" />
    </>
  ),
  play: (
    <>
      <polygon points="5 3 19 12 5 21 5 3" />
    </>
  ),
  download: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </>
  ),
  trash: (
    <>
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </>
  ),
  check: (
    <>
      <polyline points="20 6 9 17 4 12" />
    </>
  ),
  x: (
    <>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </>
  ),
  shield: (
    <>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </>
  ),
};

/**
 * @param {{name: keyof typeof PATHS, size?: number, className?: string}} props
 */
export default function Icon({ name, size = 18, className = '', ...rest }) {
  const body = PATHS[name];
  if (!body) return null;
  return (
    <svg
      className={`icon ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {body}
    </svg>
  );
}

/**
 * The product mark: an eye whose iris is a camera aperture.
 *
 * "Sarvanetra" is literally all-seeing-eye, and the platform is a camera
 * network — the mark is the two ideas drawn as one shape rather than the
 * generic shield every surveillance dashboard uses.
 */
export function BrandMark({ size = 26 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="sn-brand" x1="4" y1="4" x2="28" y2="28" gradientUnits="userSpaceOnUse">
          <stop stopColor="#22D3EE" />
          <stop offset="1" stopColor="#2F81F7" />
        </linearGradient>
      </defs>
      <path
        d="M2.5 16S7.6 7.5 16 7.5 29.5 16 29.5 16 24.4 24.5 16 24.5 2.5 16 2.5 16z"
        stroke="url(#sn-brand)"
        strokeWidth="1.9"
      />
      <circle cx="16" cy="16" r="5.6" stroke="url(#sn-brand)" strokeWidth="1.6" />
      <path
        d="M16 10.4v3M16 18.6v3M10.4 16h3M18.6 16h3"
        stroke="url(#sn-brand)"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <circle cx="16" cy="16" r="1.9" fill="url(#sn-brand)" />
    </svg>
  );
}
