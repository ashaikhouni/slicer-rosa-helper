// electron-builder afterPack hook: deep ad-hoc sign the whole .app.
//
// Research/internal build has no Apple Developer ID (identity: null), so
// electron-builder skips signing and the bundle keeps Electron's linker
// signature — inconsistent once we add the frozen sidecar as extraResources,
// which macOS on Apple Silicon can refuse to launch. A deep ad-hoc sign gives
// the whole bundle (app + Electron framework + the PyInstaller sidecar) one
// consistent signature so it runs (Gatekeeper still asks for right-click → Open
// on first launch of a downloaded copy).

"use strict";

const { execFileSync } = require("child_process");
const path = require("path");

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== "darwin") return;
  const appName = `${context.packager.appInfo.productFilename}.app`;
  const appPath = path.join(context.appOutDir, appName);
  console.log(`[after-pack] deep ad-hoc signing ${appName}`);
  execFileSync("codesign", ["--force", "--deep", "--sign", "-", appPath], { stdio: "inherit" });
};
