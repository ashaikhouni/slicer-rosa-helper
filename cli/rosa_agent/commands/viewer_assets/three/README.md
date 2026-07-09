# Vendored three.js (viewer runtime)

These files are copied verbatim from **three.js r0.158.0** (and `es-module-shims`
v1.10.0) so the exported viewer renders with **no network access** — the app,
Electron builds, and offline/air-gapped machines all work, and an ad/privacy
extension that blocks the `unpkg.com` CDN can no longer leave the 3D canvas blank.

`export_view.py` copies this `three/` tree next to a served/export `index.html`
and points the import map at `./three/…` (see `_IMPORTMAP_LOCAL`). The `picker`
(GitHub Pages) build keeps the CDN import map.

Files:
- `three.module.js` — three.js core
- `addons/controls/OrbitControls.js`
- `addons/loaders/GLTFLoader.js`
- `addons/utils/BufferGeometryUtils.js` (GLTFLoader dependency)
- `es-module-shims.js` — import-map polyfill for older browsers

## License

three.js is **MIT licensed** — Copyright © 2010–2023 three.js authors
(SPDX-License-Identifier: MIT). `es-module-shims` is MIT licensed
(Copyright © Guy Bedford). Both permit verbatim redistribution; the MIT notices
are retained in the file headers.

To update: re-download the matching version from unpkg
(`https://unpkg.com/three@<ver>/build/three.module.js`, the three addons under
`examples/jsm/…`, and `es-module-shims`), and bump the version in
`export_view.py`'s `_IMPORTMAP_CDN`.
