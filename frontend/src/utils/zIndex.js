// frontend/src/utils/zIndex.js
//
// Single source of truth for stacking order. Import this instead of writing a
// literal z-index anywhere.
//
// The problem this solves is already present: AlertDetailCard was written
// with z-index 9999 and UnroutedBanner with 50, chosen independently on
// different days with no shared scale. That happens to work today and stops
// working the moment a third overlay is added between them — silently, and
// only in the layout where they overlap.
export const Z = {
  PANEL: 1,             // ordinary grid panels
  CAMERA_BANNER: 100,   // per-camera offline notices
  OFFLINE_HEADER: 200,  // network-level offline bar
  UNROUTED_BANNER: 300, // "no officer available"
  SYSTEM_LEARNING: 400, // feedback / learning notice
  DEMO_CONTROLS: 500,   // demo trigger buttons
  MODAL: 1000,          // camera modal
  ALERT_FULLSCREEN: 1050, // critical alert card — must sit ABOVE the camera
                          // modal: a CRITICAL alert arriving while an
                          // operator is inspecting a camera feed has to be
                          // the thing they see.
  TOAST: 1100,
};

export default Z;
