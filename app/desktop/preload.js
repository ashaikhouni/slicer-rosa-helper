// Preload: expose a minimal, safe native bridge to the renderer. The existing
// web frontend feature-detects `window.rosaNative` (see desktop-shim.js) to
// replace browser uploads with real absolute paths, and to write JS-ML outputs
// to disk for the sidecar to consume. contextIsolation + sandbox are on, so this
// is the only surface the renderer gets to Node/OS.

"use strict";

const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("rosaNative", {
  // Native pickers → { path, name } (or null if cancelled).
  openFile: (opts) => ipcRenderer.invoke("rosa:openFile", opts),
  openDirectory: (opts) => ipcRenderer.invoke("rosa:openDirectory", opts),
  saveFile: (opts) => ipcRenderer.invoke("rosa:saveFile", opts),

  // Absolute path of a drag-dropped File (File.path was removed in Electron ≥32).
  getDroppedPath: (file) => {
    try { return webUtils.getPathForFile(file); } catch (_e) { return null; }
  },

  // Persist a renderer-produced artifact (e.g. a JS-ML brain mask) so the sidecar
  // can read it by path. `bytes` is a Uint8Array/ArrayBuffer.
  writeArtifact: (destPath, bytes) => ipcRenderer.invoke("rosa:writeArtifact", destPath, bytes),

  platform: process.platform,
});
