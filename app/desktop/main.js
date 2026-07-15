// ROSA desktop shell — Electron main process.
//
// Wraps the existing server-side app WITHOUT forking it: spawn the local
// `rosa_service` sidecar (which prints one JSON line `{url, port}` on stdout,
// then serves the UI on 127.0.0.1), read that line, health-check it, and point
// a BrowserWindow at the localhost URL. The renderer loads the *served* URL
// (never file://) so the frontend's same-origin fetch / SSE / viewer-iframe
// postMessage all work byte-for-byte unchanged.
//
// Dev mode (no frozen binary present): spawn `<python> -m rosa_service` with the
// repo's app/ on PYTHONPATH. Packaged mode: spawn the PyInstaller `rosa-sidecar`
// binary as `rosa-sidecar serve` from process.resourcesPath. Selection is by
// binary presence; override with ROSA_SIDECAR_MODE=dev|packaged.

"use strict";

const { app, BrowserWindow, dialog, ipcMain, shell, Menu } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");
const fs = require("fs");

const REPO_ROOT = path.resolve(__dirname, "..", ".."); // app/desktop -> app -> repo
const HEALTH_TIMEOUT_MS = 40000;
const HEALTH_POLL_MS = 400;

let sidecar = null; // the spawned child process
let sidecarInfo = null; // { url, port }
let mainWindow = null;
let quitting = false;
let logStream = null;

// ---------------------------------------------------------------------------
// Sidecar command resolution
// ---------------------------------------------------------------------------

function frozenSidecarPath() {
  // Packaged: electron-builder places the PyInstaller one-dir under
  // <resources>/rosa-sidecar/ with the launcher binary `rosa-sidecar`.
  const p = path.join(process.resourcesPath || "", "rosa-sidecar", "rosa-sidecar");
  return fs.existsSync(p) ? p : null;
}

function resolveSidecar() {
  // Explicit override — point at a PyInstaller `rosa-sidecar` binary to test the
  // real packaged compute path before full packaging (ROSA_SIDECAR_BIN=/path/...).
  const explicit = process.env.ROSA_SIDECAR_BIN;
  if (explicit && fs.existsSync(explicit)) {
    return { cmd: explicit, args: ["serve"], env: {} };
  }
  const mode = process.env.ROSA_SIDECAR_MODE || "";
  const frozen = frozenSidecarPath();
  if (mode !== "dev" && frozen) {
    return { cmd: frozen, args: ["serve"], env: {} };
  }
  // Dev: run the service module from the repo. Requires a python with the
  // rosa_service + rosa_agent packages importable (repo app/ + CommonLib/cli).
  const py = process.env.ROSA_SIDECAR_PYTHON || "python3";
  return {
    cmd: py,
    args: ["-m", "rosa_service"],
    env: {
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: [path.join(REPO_ROOT, "app"), process.env.PYTHONPATH || ""]
        .filter(Boolean)
        .join(path.delimiter),
    },
  };
}

function workRoot() {
  // Cases live in the user's app-data dir by default; override with
  // ROSA_APP_WORKDIR (used in dev to point at an existing case store).
  if (process.env.ROSA_APP_WORKDIR) return process.env.ROSA_APP_WORKDIR;
  return path.join(app.getPath("userData"), "cases");
}

// ---------------------------------------------------------------------------
// Spawn + JSON port handshake + health check
// ---------------------------------------------------------------------------

function openLog() {
  try {
    const dir = app.getPath("logs");
    fs.mkdirSync(dir, { recursive: true });
    logStream = fs.createWriteStream(path.join(dir, "sidecar.log"), { flags: "a" });
  } catch (_e) {
    logStream = null;
  }
}

function logLine(s) {
  if (logStream) logStream.write(s.endsWith("\n") ? s : s + "\n");
}

function startSidecar() {
  return new Promise((resolve, reject) => {
    const { cmd, args, env } = resolveSidecar();
    const wr = workRoot();
    fs.mkdirSync(wr, { recursive: true });
    logLine(`[shell] spawning sidecar: ${cmd} ${args.join(" ")} (workdir=${wr})`);

    sidecar = spawn(cmd, args, {
      cwd: REPO_ROOT,
      env: { ...process.env, ...env, ROSA_APP_WORKDIR: wr },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let settled = false;
    let stdoutBuf = "";

    sidecar.on("error", (err) => {
      if (!settled) { settled = true; reject(new Error(`could not launch sidecar (${cmd}): ${err.message}`)); }
    });
    sidecar.on("exit", (code, signal) => {
      logLine(`[shell] sidecar exited code=${code} signal=${signal}`);
      if (!settled) { settled = true; reject(new Error(`sidecar exited before ready (code=${code})`)); }
      else if (!quitting) onSidecarCrash(code, signal);
    });

    // The service prints exactly one JSON line {"url","port"} before uvicorn.
    sidecar.stdout.on("data", (chunk) => {
      stdoutBuf += chunk.toString();
      let nl;
      while ((nl = stdoutBuf.indexOf("\n")) >= 0) {
        const line = stdoutBuf.slice(0, nl).trim();
        stdoutBuf = stdoutBuf.slice(nl + 1);
        logLine(`[sidecar] ${line}`);
        if (!sidecarInfo && line.startsWith("{")) {
          try {
            const j = JSON.parse(line);
            if (j && j.url) {
              sidecarInfo = { url: j.url, port: j.port };
              waitForHealth(j.url).then(
                () => { if (!settled) { settled = true; resolve(sidecarInfo); } },
                (e) => { if (!settled) { settled = true; reject(e); } }
              );
            }
          } catch (_e) { /* not the port line */ }
        }
      }
    });
    sidecar.stderr.on("data", (chunk) => logLine(`[sidecar:err] ${chunk.toString().trimEnd()}`));
  });
}

function waitForHealth(url) {
  const healthUrl = new URL("healthz", url);
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(healthUrl, (res) => {
        res.resume();
        if (res.statusCode === 200) resolve();
        else retry();
      });
      req.on("error", retry);
      req.setTimeout(2000, () => req.destroy(new Error("health timeout")));
    };
    const retry = () => {
      if (Date.now() > deadline) reject(new Error("sidecar did not become healthy in time"));
      else setTimeout(tick, HEALTH_POLL_MS);
    };
    tick();
  });
}

function onSidecarCrash(code) {
  if (quitting) return;
  dialog
    .showMessageBox(mainWindow, {
      type: "error",
      title: "ROSA engine stopped",
      message: "The local ROSA engine stopped unexpectedly.",
      detail: `Exit code ${code}. Relaunch the engine, or quit.`,
      buttons: ["Relaunch", "Quit"],
      defaultId: 0,
      cancelId: 1,
    })
    .then(({ response }) => {
      if (response === 0) relaunchSidecar();
      else app.quit();
    });
}

async function relaunchSidecar() {
  sidecarInfo = null;
  try {
    const info = await startSidecar();
    if (mainWindow) mainWindow.loadURL(info.url);
  } catch (e) {
    dialog.showErrorBox("ROSA could not restart", String(e.message || e));
    app.quit();
  }
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

function createWindow(url) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 680,
    title: "ROSA",
    backgroundColor: "#0b1017",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });

  // Keep navigation on the localhost origin; open external links in the OS browser.
  const origin = new URL(url).origin;
  mainWindow.webContents.on("will-navigate", (e, target) => {
    if (new URL(target).origin !== origin) { e.preventDefault(); shell.openExternal(target); }
  });
  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    shell.openExternal(target);
    return { action: "deny" };
  });

  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => { mainWindow = null; });
  mainWindow.loadURL(url);
}

// ---------------------------------------------------------------------------
// Native-dialog IPC (consumed by preload's rosaNative bridge)
// ---------------------------------------------------------------------------

const NII_FILTERS = [
  { name: "NIfTI / NRRD", extensions: ["nii", "nii.gz", "gz", "nrrd"] },
  { name: "All files", extensions: ["*"] },
];

function registerIpc() {
  ipcMain.handle("rosa:openFile", async (_e, opts = {}) => {
    const r = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile"],
      filters: opts.filters || NII_FILTERS,
      title: opts.title || "Choose a file",
    });
    if (r.canceled || !r.filePaths.length) return null;
    const p = r.filePaths[0];
    return { path: p, name: path.basename(p) };
  });

  ipcMain.handle("rosa:openDirectory", async (_e, opts = {}) => {
    const r = await dialog.showOpenDialog(mainWindow, {
      properties: ["openDirectory"],
      title: opts.title || "Choose a folder",
    });
    if (r.canceled || !r.filePaths.length) return null;
    const p = r.filePaths[0];
    return { path: p, name: path.basename(p) };
  });

  ipcMain.handle("rosa:saveFile", async (_e, opts = {}) => {
    const r = await dialog.showSaveDialog(mainWindow, {
      title: opts.title || "Save",
      defaultPath: opts.defaultPath || undefined,
      filters: opts.filters || undefined,
    });
    if (r.canceled || !r.filePath) return null;
    return { path: r.filePath };
  });

  // Write a renderer-produced artifact (e.g. a JS-ML brain mask) to disk so the
  // sidecar can consume it by path. bytes is a transferred ArrayBuffer/Uint8Array.
  ipcMain.handle("rosa:writeArtifact", async (_e, destPath, bytes) => {
    await fs.promises.mkdir(path.dirname(destPath), { recursive: true });
    await fs.promises.writeFile(destPath, Buffer.from(bytes));
    return { path: destPath };
  });
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

function stopSidecar() {
  if (sidecar && !sidecar.killed) {
    try { sidecar.kill("SIGTERM"); } catch (_e) { /* ignore */ }
  }
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) { if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.focus(); }
  });

  app.whenReady().then(async () => {
    openLog();
    registerIpc();
    if (process.platform === "darwin") Menu.setApplicationMenu(Menu.buildFromTemplate(macMenu()));
    try {
      const info = await startSidecar();
      createWindow(info.url);
    } catch (e) {
      logLine(`[shell] startup failed: ${e.message}`);
      dialog.showErrorBox("ROSA could not start", String(e.message || e));
      app.quit();
    }
  });

  app.on("activate", () => {
    if (mainWindow === null && sidecarInfo) createWindow(sidecarInfo.url);
  });

  app.on("window-all-closed", () => app.quit());
  app.on("before-quit", () => { quitting = true; stopSidecar(); });
  app.on("will-quit", stopSidecar);
}

function macMenu() {
  return [
    { role: "appMenu" },
    { role: "editMenu" },
    { role: "viewMenu" },
    { role: "windowMenu" },
  ];
}
