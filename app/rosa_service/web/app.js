"use strict";
// ROSA web UI — a single-page wizard over the local service contract.
// Drop CT → run pipeline job → review/edit contacts + 3D viewer → export.
// Talks ONLY to /api/v1 + /healthz (never the engine directly).

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const state = { ct: null, jobId: null, es: null, poll: null,
                mri: null, labelJobId: null, labelPoll: null, qc: null,
                creationMri: null,     // optional MRI (T1) picked at case creation
                probePref: "ortho" };  // remembered Ortho/Probe slice-rail choice

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
async function jsend(url, method, body) {
  const r = await fetch(url, {
    method, headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

// Desktop (Electron) hands us real absolute paths via window.rosaNative, so we
// skip the browser upload round-trip entirely — the local sidecar reads the file
// in place (no multi-hundred-MB copy). In a browser, rosaNative is absent and we
// POST to /uploads exactly as before.
const IS_DESKTOP = !!(window.rosaNative && window.rosaNative.pathForFile);

// Resolve a picked/dropped File to { path, name, bytes }. Desktop → the real
// path (instant); browser → upload the bytes and use the saved path.
async function resolveFile(file) {
  if (IS_DESKTOP) {
    const path = window.rosaNative.pathForFile(file);
    if (path) return { path, name: file.name, bytes: file.size };
  }
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/uploads`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// THOMAS is a folder pick. Desktop → a native directory dialog (the label
// endpoint takes the folder path directly, no upload). Browser → the
// webkitdirectory input that uploads the masks (uploadThomasDir).
async function pickThomasDir() {
  if (IS_DESKTOP && window.rosaNative.openDirectory) {
    const r = await window.rosaNative.openDirectory({ title: "Choose a THOMAS output folder (has left/ + right/)" });
    if (r && r.path && state.jobId) { state.thomasDir = r.path; runLabel(); }
    return;
  }
  $("thomasdir-file").click();
}

function showStep(name) {
  document.body.classList.toggle("results-active", name === "results");
  // The step bar tracks the new-case wizard (drop → run → review); it's noise on
  // the case list, the import form, and the workspace (which has its own flow).
  document.body.classList.toggle("nosteps", ["cases", "import", "results", "cohort", "rosa"].includes(name));
  for (const p of document.querySelectorAll(".panel")) {
    p.classList.toggle("active", p.id === `panel-${name}`);
  }
  const order = ["drop", "run", "results"];
  document.querySelectorAll("#stepbar li").forEach((li) => {
    const s = li.dataset.step;
    li.classList.toggle("active", s === name);
    li.classList.toggle("done", order.indexOf(s) < order.indexOf(name));
  });
}

// ---- step 1: pick a CT ------------------------------------------------

function setCt(ct) {
  state.ct = ct;
  $("ctinfo").textContent = ct
    ? `CT: ${ct.name}${ct.bytes ? ` (${(ct.bytes / 1e6).toFixed(1)} MB)` : ""}`
    : "";
  $("runbtn").disabled = !ct;
}

async function uploadFile(file) {
  $("ctinfo").textContent = IS_DESKTOP ? `Loading ${file.name}…` : `Uploading ${file.name}…`;
  try { setCt(await resolveFile(file)); }
  catch (e) { $("ctinfo").textContent = `Failed: ${e.message}`; }
}

function wireDrop() {
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((e) =>
    dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("hover"); }));
  ["dragleave", "drop"].forEach((e) =>
    dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("hover"); }));
  dz.addEventListener("drop", (ev) => {
    const f = ev.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
  $("fileinput").addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (f) uploadFile(f);
  });
  // The path field takes a NIfTI file OR a DICOM folder (auto-detected).
  $("ctpath").addEventListener("input", (ev) => typedPathInput("ct", ev.target.value));
  $("ctpath").addEventListener("keydown", (e) => { if (e.key === "Enter") typedPathEnter("ct", e.target.value); });
  // Optional MRI (T1) at case creation — upload or point at a path.
  $("mrifileinput").addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (f) uploadCreationMri(f);
  });
  $("mripath").addEventListener("input", (ev) => typedPathInput("mri", ev.target.value));
  $("mripath").addEventListener("keydown", (e) => { if (e.key === "Enter") typedPathEnter("mri", e.target.value); });
}

function setCreationMri(mri) {
  state.creationMri = mri;
  $("mricreateinfo").textContent = mri
    ? `MRI: ${mri.name}${mri.bytes ? ` (${(mri.bytes / 1e6).toFixed(1)} MB)` : ""}`
    : "";
}

async function uploadCreationMri(file) {
  $("mricreateinfo").textContent = IS_DESKTOP ? `Loading ${file.name}…` : `Uploading ${file.name}…`;
  try { setCreationMri(await resolveFile(file)); }
  catch (e) { $("mricreateinfo").textContent = `Failed: ${e.message}`; }
}

// ---- image loading: one picker per role, NIfTI file OR DICOM folder ----------
// The app decides what was given: a NIfTI file loads directly; a directory is a
// DICOM series → converted to a de-identified NIfTI first. Desktop uses one
// native dialog that accepts either; typed/pasted paths + the browser fall back
// to the file extension (NIfTI) vs "looks like a folder" (DICOM).
const NIFTI_RE = /\.(nii(\.gz)?|nrrd|gz)$/i;
const _role = (t) => (t === "mri" ? "MRI" : "CT");
const _infoEl = (t) => $(t === "mri" ? "mricreateinfo" : "ctinfo");
const _setImage = (t, picked) => (t === "mri" ? setCreationMri(picked) : setCt(picked));

// target = "ct" | "mri". Native picker on desktop; browser falls back to the
// hidden file input (NIfTI only — folder upload isn't a browser thing).
async function pickImage(target) {
  if (!(IS_DESKTOP && window.rosaNative.openImage)) {
    $(target === "mri" ? "mrifileinput" : "fileinput").click();
    return;
  }
  const r = await window.rosaNative.openImage({
    title: `Choose the ${_role(target)} — NIfTI file or DICOM folder` });
  if (!r || !r.path) return;
  if (r.isDirectory) { importDicom(target, r.path); return; }
  _setImage(target, { path: r.path, name: r.path.split("/").pop() });
}

// A typed/pasted path: NIfTI extension → load live; otherwise it's a folder →
// convert on Enter (never per-keystroke — the conversion is a round-trip).
function typedPathInput(target, raw) {
  const p = (raw || "").trim();
  if (!p) { _setImage(target, null); return; }
  if (NIFTI_RE.test(p)) _setImage(target, { path: p, name: p.split("/").pop() });
  // non-NIfTI: hold until Enter (typedPathEnter) so we don't convert mid-type.
}
function typedPathEnter(target, raw) {
  const p = (raw || "").trim();
  if (p && !NIFTI_RE.test(p)) importDicom(target, p);
}

// Convert a DICOM series folder → de-identified NIfTI, then load it as CT/MRI.
async function importDicom(target, dir) {
  const info = _infoEl(target), role = _role(target);
  info.textContent = "Converting DICOM → NIfTI (de-identifying)…";
  try {
    const r = await jsend(`${API}/dicom-to-nifti`, "POST", { dicom_dir: dir });
    _setImage(target, { path: r.path, name: r.path.split("/").pop() });   // enables run
    info.innerHTML = `${role}: <b>de-identified from DICOM</b> ✓ — <span class="muted">${r.path.split("/").pop()}</span>`;
  } catch (e) {
    info.textContent = `DICOM import failed: ${e.message}`;
  }
}

// ---- import a ROSA robot case (de-identify → bake volumes → confirm roles) ----
let ROSA = null;
async function importRosa() {
  if (!(IS_DESKTOP && window.rosaNative && window.rosaNative.openDirectory)) {
    window.alert("Importing a ROSA case needs the desktop app."); return;
  }
  const r = await window.rosaNative.openDirectory({ title: "Choose the ROSA case folder (contains a .ros file)" });
  if (!r || !r.path) return;
  showStep("rosa");
  $("rosa-sub").textContent = "· " + r.path.split("/").pop();
  $("rosa-displays").innerHTML = ""; $("rosa-foot").hidden = true;
  $("rosa-hint").textContent = "De-identifying + baking volumes… (this can take a moment)";
  try { ROSA = await jsend(`${API}/rosa-import/prepare`, "POST", { rosa_dir: r.path }); }
  catch (e) { $("rosa-hint").innerHTML = `<span style="color:var(--bad)">Prepare failed: ${e.message}</span>`; return; }
  renderRosaConfirm();
}
function renderRosaConfirm() {
  $("rosa-label").value = ROSA.label || "";
  const host = $("rosa-displays"); host.innerHTML = "";
  for (const d of ROSA.displays) {
    const card = el("div", { class: "rosa-card" });
    const img = el("img", { class: "rosa-thumb", alt: d.name });
    if (d.png) img.src = d.png;
    const sel = el("select", { class: "rosa-role" }); sel.dataset.path = d.path;
    for (const [v, t] of [["", "— ignore —"], ["ct", "Post-op CT"], ["t1", "MRI T1 (non-contrast)"]]) {
      const o = el("option", { value: v }, t); if (v === d.guess) o.selected = true; sel.append(o);
    }
    sel.onchange = updateRosaCreate;
    card.append(img, el("div", { class: "rosa-name" }, d.name), sel);
    host.append(card);
  }
  $("rosa-foot").hidden = false;
  updateRosaCreate();
}
function _rosaRoles() {
  const roles = {};
  document.querySelectorAll(".rosa-role").forEach((s) => {
    if (s.value) (roles[s.value] = roles[s.value] || []).push(s.dataset.path);
  });
  return roles;
}
function updateRosaCreate() {
  const roles = _rosaRoles(), nCt = (roles.ct || []).length, nT1 = (roles.t1 || []).length;
  $("rosa-create").disabled = !(nCt === 1 && nT1 <= 1);
  $("rosa-hint").textContent = nCt === 1
    ? "Confirm the roles, then create the case." + (nT1 ? "" : " (T1 optional but recommended.)")
    : (nCt === 0 ? "Select the post-op CT." : "Only one post-op CT — set the others to ignore.");
}
async function createRosaCase() {
  const roles = _rosaRoles(), ct = (roles.ct || [])[0];
  if (!ct) return;
  const t1 = (roles.t1 || [])[0] || null;
  $("rosa-create").disabled = true; $("rosa-hint").textContent = "Creating case + running detection…";
  try {
    const job = await jsend(`${API}/rosa-import/create`, "POST", {
      ct_path: ct, t1_path: t1, label: ($("rosa-label").value || "").trim(),
      seeds_path: ROSA.seeds || "" });
    openCase(job.id);
  } catch (e) {
    $("rosa-hint").innerHTML = `<span style="color:var(--bad)">Create failed: ${e.message}</span>`;
    $("rosa-create").disabled = false;
  }
}

// ---- electrode library (constrain detection to the site's electrode types) ---

// id → "Manufacturer Reference" (e.g. "DIXI D08-05AM"), for the review list.
function modelLabel(id) {
  return (id && state.emodelLabels && state.emodelLabels[id]) || id || "—";
}

async function loadElectrodeTypes() {
  const list = $("etypeslist");
  try {
    const { types, labels } = await jget(`${API}/electrode-models`);
    state.emodelLabels = labels || {};
    list.innerHTML = "";
    for (const t of types) {
      const cc = t.contact_counts && t.contact_counts.length ? ` · ${t.contact_counts.join("/")} contacts` : "";
      const cb = el("input", { type: "checkbox", value: t.type });
      cb.checked = true;
      cb.addEventListener("change", updateEtypesSummary);
      // "DIXI AM" / "Medtronic ring_4" / "NeuroPace depth_4_3.5mm" / "PMT"
      const name = t.manufacturer && t.manufacturer.toUpperCase() !== t.type.toUpperCase()
        ? `${t.manufacturer} ${t.type}` : t.type;
      const lbl = el("label", { class: "etype" });
      lbl.append(cb, el("span", { class: "etype-name" }, name),
                 el("span", { class: "etype-meta" }, `${t.count} model${t.count > 1 ? "s" : ""}${cc}`));
      list.append(lbl);
    }
    updateEtypesSummary();
  } catch (_e) {
    list.innerHTML = '<span class="muted">electrode library unavailable</span>';
  }
}

function selectedEtypes() {
  return [...$("etypeslist").querySelectorAll('input[type="checkbox"]')]
    .filter((c) => c.checked).map((c) => c.value);
}

function setAllEtypes(on) {
  $("etypeslist").querySelectorAll('input[type="checkbox"]').forEach((c) => { c.checked = on; });
  updateEtypesSummary();
}

function updateEtypesSummary() {
  const total = $("etypeslist").querySelectorAll('input[type="checkbox"]').length;
  const sel = selectedEtypes();
  const s = $("etypessummary");
  if (!total) { s.textContent = ""; return; }
  if (sel.length === total) { s.textContent = "all types"; s.className = "muted"; }
  else if (sel.length === 0) { s.textContent = "none selected → all types used"; s.className = "muted"; }
  else { s.textContent = sel.join(", "); s.className = "accent"; }
}

// ---- step 2: run ------------------------------------------------------

async function run(force = false) {
  const label = $("label").value.trim() || "case";
  const params = { ct: state.ct.path, label };
  if (force) params.force = true;
  if (state.creationMri) {
    params.t1 = state.creationMri.path;                        // MRI from the start
    params.surface = $("surfacesel").value;                    // brain-surface backend
  }
  // Constrain detection to the site's electrode types — only when the user
  // narrowed it to a proper subset (all/none → full library, no constraint).
  const etypes = selectedEtypes();
  const etypesTotal = $("etypeslist").querySelectorAll('input[type="checkbox"]').length;
  if (etypes.length && etypes.length < etypesTotal) params.electrode_types = etypes.join(",");
  // POST first — a duplicate CT (409) is caught here, before we leave the form.
  let r, body;
  try {
    r = await fetch(`${API}/jobs`, { method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind: "pipeline", params }) });
    body = await r.json().catch(() => ({}));
  } catch (e) { $("ctinfo").textContent = `Could not start: ${e.message}`; return; }
  if (r.status === 409 && body.detail && body.detail.existing)
    return promptDuplicate(body.detail, () => run(true));
  if (!r.ok) { $("ctinfo").textContent = `Could not start: ${body.detail || r.status}`; return; }
  state.jobId = body.id;
  showStep("run");
  $("log").textContent = "";
  $("runstate").textContent = "Starting…";
  // The MRI adds brain-strip + surface reconstruction to the run.
  $("runsub").textContent = state.creationMri
    ? "detect → place contacts → strip MRI + build surface → viewer"
    : "detect → place contacts → build viewer (~2–3 min)";
  $("spinner").classList.remove("stopped");
  streamLogs(body.id);
  pollStatus(body.id);
}

// A New-case / Import hit an existing case for the same CT. Offer to open it, or
// run anyway (force). Keeps accidental re-runs of the same scan out of the list.
function promptDuplicate(detail, onForce) {
  const ex = detail.existing || {};
  const when = ex.created_at ? new Date(ex.created_at * 1000).toLocaleDateString() : "";
  const msg = `${detail.message}${when ? ` Created ${when}.` : ""}\n\n` +
    "OK → open the existing case.\nCancel → create another anyway.";
  if (window.confirm(msg)) openCase(ex.id);
  else onForce();
}

function appendLog(line, cls) {
  const log = $("log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  const span = document.createElement("span");
  if (cls) span.className = cls;
  else if (line.startsWith("[step")) span.className = "step";
  span.textContent = line + "\n";
  log.appendChild(span);
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function streamLogs(id) {
  if (state.es) state.es.close();
  const es = new EventSource(`${API}/jobs/${id}/logs`);
  state.es = es;
  es.onmessage = (ev) => { if (ev.data) appendLog(ev.data); };
  es.addEventListener("end", () => es.close());
  es.onerror = () => es.close();
}

function pollStatus(id) {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let st;
    try { st = await jget(`${API}/jobs/${id}`); } catch { return; }
    $("runstate").textContent = `State: ${st.state}`;
    if (["succeeded", "failed", "cancelled"].includes(st.state)) {
      clearInterval(state.poll);
      $("spinner").classList.add("stopped");
      if (st.state === "succeeded") loadResults(id);
      else $("runstate").textContent =
        st.state === "cancelled" ? "Cancelled" : `Failed (exit ${st.exit_code}${st.error ? ": " + st.error : ""})`;
    }
  }, 1000);
}

// Cancel a running detection: stop watching, terminate the job (DELETE →
// SIGTERM), and go back to the load screen. The CT (and any MRI) stay selected
// so re-running is one click — `run()` resets the run screen itself.
async function cancel() {
  clearInterval(state.poll);
  if (state.es) { state.es.close(); state.es = null; }
  const id = state.jobId;
  state.jobId = null;
  showStep("drop");
  if (id) { try { await fetch(`${API}/jobs/${id}`, { method: "DELETE" }); } catch {} }
}

// ---- step 3: review + viewer -----------------------------------------

async function loadResults(id) {
  const frame = $("viewerframe");
  // The embedded viewer (export-view) has its OWN trajectories/contacts sidebar
  // (#side) — redundant with our editable list. Hide it (same-origin iframe) and
  // let the 3D scene + slices reflow to fill. Cosmetic; degrades gracefully.
  frame.onload = () => {
    try {
      const d = frame.contentDocument;
      const s = d.createElement("style");
      s.textContent = "#side{display:none!important} #app{grid-template-columns:1fr 340px!important}";
      d.head.appendChild(s);
    } catch (_e) { /* ignore */ }
    syncVisibility(state.doc);   // apply reject-hiding once the viewer is ready
  };
  frame.src = `${API}/jobs/${id}/viewer/`;
  state.jobId = id;
  showStep("results");
  resetLabelCard();
  prefillLabelMriFromCase(id);   // MRI provided at creation → no re-upload to label
  try { renderReview(await jget(`${API}/jobs/${id}/review`)); }
  catch (e) { $("reviewlist").textContent = `Could not load review: ${e.message}`; }
  setCaseSlots();
  showWs("review");   // land in Review; the Edit view lazy-loads the editor on demand
}

// ---- workspace: Edit / Review views + the case-input control ----------

function showWs(v) {
  document.querySelectorAll("#wsflow button").forEach((b) => b.setAttribute("aria-selected", b.dataset.ws === v));
  $("ws-edit").classList.toggle("on", v === "edit");
  $("ws-review").classList.toggle("on", v === "review");
  if (v === "edit" && state.jobId) {
    const f = $("editframe"), want = `${API}/jobs/${state.jobId}/editor/`;
    if (f.getAttribute("src") !== want) f.src = want;   // load the editor once, on first open
  }
}

// The embedded editor saved a geometry edit (postMessage) → a viewer rebuild is
// running. Poll it, then refresh Review's contacts + 3D and land back on Review.
async function onEdited(rebuildJob) {
  const refresh = async () => {
    try { renderReview(await jget(`${API}/jobs/${state.jobId}/review`)); } catch (_e) {}
    $("viewerframe").src = `${API}/jobs/${state.jobId}/viewer/?t=${Date.now()}`;
    $("editframe").removeAttribute("src");   // reload the editor's plan next open (geometry changed)
    showWs("review");
  };
  const id = rebuildJob && rebuildJob.id;
  if (!id) return refresh();
  clearInterval(state.rebuildPoll);
  state.rebuildPoll = setInterval(async () => {
    let st; try { st = await jget(`${API}/jobs/${id}`); } catch { return; }
    if (["succeeded", "failed", "cancelled"].includes(st.state)) { clearInterval(state.rebuildPoll); refresh(); }
  }, 1000);
}

function setCaseSlots() {
  const mriOn = !!state.mri;
  const m = $("slot-mri");
  m.className = "slot " + (mriOn ? "on" : "off");
  m.querySelector(".fn").textContent = mriOn ? "loaded" : "add";
  $("slot-reg").hidden = !mriOn;
  $("atlasctl").hidden = !mriOn;        // atlas picker appears once an MRI is in
  $("labelbarMri").hidden = mriOn;      // MRI upload row shows only when there's no MRI
}

// If the case was created with an MRI, the pipeline job carries its t1: pre-fill
// the label card so labeling needs only an atlas pick (no second upload).
async function prefillLabelMriFromCase(id) {
  try {
    const st = await jget(`${API}/jobs/${id}`);
    if (st && st.t1) {
      state.mri = { path: st.t1, name: "(MRI from case creation)" };
      $("labelstatus").textContent = "· MRI from case creation";
      $("labelbtn").disabled = false;
      setCaseSlots();
    }
  } catch (_e) { /* ignore */ }
}

// The label card is per-job: a fresh run starts with no MRI / no proposal.
function resetLabelCard() {
  clearInterval(state.labelPoll);
  state.mri = null; state.labelJobId = null; state.labeledOnce = false;
  $("mriinput").value = "";
  $("labelbtn").disabled = true;
  $("approvebtn").hidden = true;
  $("labelstatus").textContent = "";
  $("labelmsg").textContent = "";
  const ll = $("labellog"); ll.hidden = true; ll.textContent = "";
  $("qcplanes").innerHTML = ""; state.qc = null;
  $("atlasctl").hidden = true;
  $("atlascheckbtn").hidden = true;
  $("regtab").hidden = true;              // no registration to check until an atlas runs
  state.thomasDir = null;
  setViewerTab("electrodes");
}

// The review list is a per-electrode NAVIGATOR: one collapsible row per shank
// (12, not 176), expand to see its contacts + labels. Editing geometry lives in
// the Edit view — so no keep/reject checkboxes here; Review is read + label +
// accept. Contact rows keep their data-* hooks + .region input so the labeling
// preview, 3D selection, and relabel all still bind to them.
function renderReview(doc) {
  state.doc = doc;
  const nShanks = doc.shanks.length;
  const nContacts = doc.shanks.reduce((a, s) => a + s.contacts.length, 0);
  $("summary").textContent = `· ${nShanks} electrodes · ${nContacts} contacts`;

  const list = $("reviewlist");
  list.innerHTML = "";
  for (const shank of doc.shanks) {
    const box = el("div", { class: "shank" });
    box.dataset.shank = shank.name;
    const labeled = shank.contacts.some((c) => c.region);

    const head = el("div", { class: "shank-head" });
    const dot = el("span", { class: "shank-dot" + (labeled ? " on" : "") });
    dot.title = labeled ? "labeled" : "not labeled";
    head.append(el("span", { class: "caret" }, "▸"), el("strong", {}, shank.name),
      el("span", { class: "muted shank-meta" }, `${modelLabel(shank.model)} · ${shank.contacts.length}`), dot);
    head.onclick = () => toggleShank(box, shank);
    box.append(head);

    const cs = el("div", { class: "contacts" });
    for (const c of shank.contacts) {
      const row = el("div", { class: "contact" });
      row.dataset.label = c.name; row.dataset.shank = shank.name; row.dataset.cindex = c.index;
      row.onclick = (ev) => { if (!ev.target.closest("input")) selectInViewer(c.name, shank.name); };
      const region = el("input", { type: "text", class: "region", value: c.region || "", placeholder: "—" });
      region.onchange = () => {
        if (region.value.trim())
          patch([{ op: "relabel_contact", shank: shank.name, index: c.index, region: region.value.trim() }]);
      };
      row.append(el("span", { class: "cname", title: "show in 3D" }, c.name), region);
      cs.append(row);
    }
    box.append(cs);
    list.append(box);
  }
  // Keep the open shank open across re-renders (labeling/relabel re-render the list).
  if (state.openShank) {
    const b = list.querySelector(`.shank[data-shank="${CSS.escape(state.openShank)}"]`);
    if (b) b.classList.add("open");
  }
  syncVisibility(doc);
}

// Accordion: one shank open at a time; opening one snaps the 3D view to it.
function toggleShank(box, shank) {
  const open = !box.classList.contains("open");
  box.parentElement.querySelectorAll(".shank.open").forEach((b) => b.classList.remove("open"));
  box.classList.toggle("open", open);
  state.openShank = open ? shank.name : null;
  if (open && shank.contacts[0]) selectInViewer(shank.contacts[0].name, shank.name);
  else if (!open) showAllInViewer();   // collapsing the electrode → un-isolate the 3D
}

function _postViewer(msg) {
  try { $("viewerframe").contentWindow.postMessage(msg, "*"); } catch (_e) { /* not loaded */ }
}

// The embedded viewer reports which slice-display controls apply for this case
// (has a warped atlas? has an MRI to fade to?). Host them in the top row and
// reset to the viewer's defaults (atlas on, fade at CT) on each (re)load.
function _onSliceCaps({ atlas, fade }) {
  state.sliceCaps = { atlas: !!atlas, fade: !!fade };
  $("app-atlas-ov").checked = true;
  $("app-slice-fade").value = 0;
  _syncModeControls();
}

// Contextual top bar: the 3-D-view controls (Ortho/Probe, slice-fade, atlas-on-
// slices) and the registration-check controls (space + blend + slider) never show
// at the same time — the bar SWAPS with the view, so there's one CT—MRI slider at
// a time. Availability (probe present, slice caps) still gates the 3-D controls.
function _syncModeControls() {
  const is3d = (state.viewMode || "electrodes") !== "qc";
  const caps = state.sliceCaps || {};
  const probe = state.probeCaps || {};
  $("probemodectl").hidden = !(is3d && probe.present);
  $("slicefade-item").hidden = !(is3d && caps.fade);
  $("atlasov-item").hidden = !(is3d && caps.atlas);
  $("sliceovctl").hidden = !(is3d && (caps.fade || caps.atlas));
  $("qctools").hidden = is3d;
}

function _setAppSliceMode(mode) {
  for (const b of $("app-slice-mode").querySelectorAll("button"))
    b.classList.toggle("active", b.dataset.smode === mode);
}
// The Ortho|Probe toggle is ALWAYS shown for a case with electrodes; Probe is
// greyed until an electrode (≥2 contacts) is selected — so the switch is
// predictable and the disabled state explains why it's on Ortho. We also
// remember the user's choice: re-selecting an electrode returns to Probe if
// that's what they were using (no surprise flip back to Ortho).
function _onProbeAvail({ present, available }) {
  state.probeCaps = { present: !!present, available: !!available };
  const probeBtn = $("app-slice-mode").querySelector('[data-smode="probe"]');
  probeBtn.disabled = !available;
  probeBtn.title = available ? "Along the selected electrode + probe's-eye"
                             : "Select an electrode to enable the probe view";
  if (available && state.probePref === "probe") {
    _setAppSliceMode("probe"); _postViewer({ type: "rosa:slice-mode", mode: "probe" });
  } else if (!available) {
    _setAppSliceMode("ortho");
  }
  _syncModeControls();
}
// The viewer selected a contact (3D click, or arrow-stepping the probe) → mirror
// it in the left navigator: open the shank, highlight + scroll to the row.
function _onViewerSelected({ label, shank }) {
  if (shank) {
    const box = document.querySelector(`#reviewlist .shank[data-shank="${CSS.escape(shank)}"]`);
    if (box && !box.classList.contains("open")) {
      document.querySelectorAll("#reviewlist .shank.open").forEach((b) => b.classList.remove("open"));
      box.classList.add("open"); state.openShank = shank;
    }
  }
  document.querySelectorAll("#reviewlist .contact.selected").forEach((r) => r.classList.remove("selected"));
  const sel = window.CSS && CSS.escape ? CSS.escape(label) : label;
  const row = document.querySelector(`#reviewlist .contact[data-label="${sel}"]`);
  if (row) { row.classList.add("selected"); row.scrollIntoView({ block: "nearest" }); }
  $("showallbtn").hidden = false;
}

// Un-isolate the 3D view: show every electrode again + clear the selection.
// Send BOTH messages: newer viewers handle rosa:showall (full reset); any viewer
// (incl. ones rendered before this change) handles rosa:visibility with empty
// hide-lists, which un-hides every shank/contact — so "Show all" works without
// waiting for the case's viewer to be re-rendered.
function showAllInViewer() {
  const w = $("viewerframe").contentWindow;
  try {
    w.postMessage({ type: "rosa:showall" }, "*");
    w.postMessage({ type: "rosa:visibility", hideShanks: [], hideContacts: [] }, "*");
  } catch (_e) { /* viewer not loaded yet */ }
  document.querySelectorAll("#reviewlist .contact.selected").forEach((r) => r.classList.remove("selected"));
  $("showallbtn").hidden = true;
}

// Hide rejected shanks/contacts in the 3D viewer; re-accepting brings them back.
function syncVisibility(doc) {
  if (!doc) return;
  const hideShanks = [], hideContacts = [];
  for (const s of doc.shanks) {
    if (!s.accepted) { hideShanks.push(s.name); continue; }
    for (const c of s.contacts) if (!c.accepted) hideContacts.push({ label: c.name, shank: s.name });
  }
  try { $("viewerframe").contentWindow.postMessage({ type: "rosa:visibility", hideShanks, hideContacts }, "*"); }
  catch (_e) { /* viewer not ready */ }
}

async function patch(ops) {
  try { renderReview(await jsend(`${API}/jobs/${state.jobId}/review`, "PATCH", { ops })); }
  catch (e) { $("exportmsg").textContent = `Edit failed: ${e.message}`; }
}

// Click a contact in the list → highlight it in the embedded 3D viewer
// (postMessage) and mark the row selected. The viewer snaps camera + slices.
function selectInViewer(label, shank) {
  try { $("viewerframe").contentWindow.postMessage({ type: "rosa:select", label, shank }, "*"); }
  catch (_e) { /* viewer not loaded yet */ }
  // Selecting isolates the shank in 3D — surface the "Show all" way back.
  $("showallbtn").hidden = false;
  document.querySelectorAll("#reviewlist .contact.selected").forEach((r) => r.classList.remove("selected"));
  const sel = window.CSS && CSS.escape ? CSS.escape(label) : label;
  const row = document.querySelector(`#reviewlist .contact[data-label="${sel}"]`);
  if (row) row.classList.add("selected");
}

async function doExport() {
  try {
    const r = await jsend(`${API}/jobs/${state.jobId}/review/export`, "POST");
    const url = `${API}/jobs/${state.jobId}/files/${r.rel_path}`;
    $("exportmsg").innerHTML =
      `Exported ${r.n_contacts} contacts — <a href="${url}" download>download ${r.rel_path}</a>`;
  } catch (e) { $("exportmsg").textContent = `Export failed: ${e.message}`; }
}

function restart() {
  if (state.es) state.es.close();
  clearInterval(state.poll);
  resetLabelCard();
  state.ct = null; state.jobId = null;
  $("ctpath").value = ""; $("fileinput").value = ""; $("exportmsg").textContent = "";
  $("mripath").value = ""; $("mrifileinput").value = ""; $("mrioptional").open = false;
  setCreationMri(null);
  $("viewerframe").src = "about:blank";
  setCt(null);
  showStep("drop");
}

// ---- anatomical labeling (MRI → register → propose → approve) --------

async function loadAtlases() {
  try {
    const { atlases, default: def } = await jget(`${API}/atlases`);
    const sel = $("atlassel");
    sel.innerHTML = "";
    for (const a of atlases) {
      const short = (a.name || a.id).split(":")[0].trim();
      const nc = a.license_tier === "noncommercial" ? " · non-commercial" : "";
      const o = el("option", { value: a.id }, `${short}${nc}`);
      o.title = a.license || "";                     // full license on hover
      if (!a.available) { o.disabled = true; o.textContent += " (not installed)"; }
      if (a.id === def) o.selected = true;
      sel.append(o);
    }
    _syncThomasLoadBtn();
  } catch (_e) { $("labelstatus").textContent = "· atlas list unavailable"; }
}

// THOMAS is a BYO atlas, so it needs a "load folder" trigger EVERY time it's the
// selection — the dropdown's `change` doesn't fire when THOMAS is already picked
// (case reopened, or re-importing a new folder). This chip, shown whenever THOMAS
// is selected, is that reliable trigger.
function _syncThomasLoadBtn() {
  $("thomasload").hidden = !String($("atlassel").value || "").startsWith("thomas");
}

async function uploadMri(file) {
  $("labelstatus").textContent = IS_DESKTOP ? `· loading ${file.name}…` : `· uploading ${file.name}…`;
  try {
    state.mri = await resolveFile(file);
    $("labelstatus").textContent = `· MRI ${state.mri.name}`;
    $("labelbtn").disabled = false;
    setCaseSlots();
  } catch (_e) { $("labelstatus").textContent = "· MRI failed"; }
}

// Busy state during registration/labeling: a spinner beside the atlas picker.
// The picker stays ENABLED so you can switch atlas mid-run (see runLabel — it
// cancels the in-flight job so the new atlas takes over instead of queuing).
function setLabelBusy(on) {
  $("atlasbusy").hidden = !on;
}

async function runLabel() {
  if (!state.jobId) return;
  const atlasId = $("atlassel").value;
  const isThomas = atlasId.startsWith("thomas");
  // THOMAS brings its own reference MRI, so it labels without a patient MRI — but
  // it needs the uploaded folder. Every other atlas needs the patient MRI.
  if (isThomas) { if (!state.thomasDir) return; }
  else if (!state.mri) return;
  // Cancel any in-flight label for this case first — otherwise max_concurrent=1
  // would queue the new atlas behind the old (slow) one and it'd look stuck.
  if (state.labelJobId) {
    try { await fetch(`${API}/jobs/${state.labelJobId}`, { method: "DELETE" }); } catch (_e) {}
  }
  clearInterval(state.labelPoll);
  // First label of a case registers CT↔MRI (verify once); later atlas switches
  // reuse that registration and only recompute labels.
  const firstLabel = !state.labeledOnce;
  setLabelBusy(true);
  $("labelbtn").disabled = true;
  $("approvebtn").hidden = true;
  const ll = $("labellog"); ll.hidden = false; ll.textContent = "";
  const atlas = $("atlassel").options[$("atlassel").selectedIndex]?.textContent || atlasId;
  $("labelmsg").innerHTML = isThomas
    ? `Registering THOMAS MRI → CT and warping the nuclei (~30–60 s) — the 3D + slices update when done…`
    : firstLabel
      ? `Registering MRI → CT and warping <b>${atlas}</b> (~30 s) — the 3D recolors when done…`
      : `Relabeling with <b>${atlas}</b> — CT↔MRI registration is cached…`;
  try {
    const body = { atlas: atlasId };
    if (state.mri) body.t1 = state.mri.path;
    if (isThomas) body.thomas_dir = state.thomasDir;
    const job = await jsend(`${API}/jobs/${state.jobId}/label`, "POST", body);
    state.labelJobId = job.id;
    streamInto(job.id, "labellog");     // shows the registration metrics (QC)
    pollLabel(job.id, firstLabel);
  } catch (e) {
    setLabelBusy(false);
    $("labelmsg").textContent = `Failed to start: ${e.message}`;
    $("labelbtn").disabled = false;
  }
}

// The newest label job for the current case (or null). /jobs is newest-first.
async function latestLabelJob() {
  try {
    const jobs = await jget(`${API}/jobs`);
    return jobs.find((j) => j.kind === "label" && j.parent === state.jobId) || null;
  } catch { return null; }
}

function pollLabel(id, firstLabel) {
  clearInterval(state.labelPoll);
  state.labelPoll = setInterval(async () => {
    let st;
    try { st = await jget(`${API}/jobs/${id}`); } catch { return; }
    if (!["succeeded", "failed", "cancelled"].includes(st.state)) return;
    clearInterval(state.labelPoll);
    $("labelbtn").disabled = false;
    // Follow the LATEST label job for the case, not this fixed id — so a
    // failed→retry→succeeded sequence advances the UI instead of sticking on
    // the failed message (the retry is often a newer job).
    const latest = await latestLabelJob();
    if (latest && latest.id !== id) {
      state.labelJobId = latest.id;
      if (latest.state === "succeeded") {
        state.labeledOnce = true;
        // Reload the viewer on every label — the brain surface is now recolored
        // by the active atlas, so the 3D changes on an atlas switch, not just the
        // first label. Only the first label jumps to the registration screen.
        showProposed(latest.id, { reloadViewer: true, jumpToQc: firstLabel });
      } else if (["failed", "cancelled"].includes(latest.state)) {
        _showLabelError(latest);
      } else { pollLabel(latest.id, firstLabel); }   // a newer one is still running
      return;
    }
    if (st.state === "succeeded") {
      state.labeledOnce = true;
      showProposed(id, { reloadViewer: true, jumpToQc: firstLabel });
    } else { _showLabelError(st); }
  }, 1000);
}

function _showLabelError(st) {
  setLabelBusy(false);
  $("labelmsg").innerHTML =
    `Labeling <strong>${st.state}</strong>${st.error ? ": " + st.error : ` (exit ${st.exit_code})`}. ` +
    `Pick the atlas again to retry.`;
  $("labelbtn").disabled = false;
}

// ---- THOMAS: a BYO atlas (load a folder) --------------------------------
// THOMAS is picked from the atlas dropdown like any other atlas; the only
// difference is it prompts for a THOMAS output folder. We upload the masks, then
// run the NORMAL label flow against them (register its MRI → CT, warp the nuclei,
// tint the slices + render the 3-D meshes + label contacts, with check-reg +
// approve). Only the files build_thomas_labelmap reads are uploaded.
async function uploadThomasDir(e) {
  const all = Array.from(e.target.files || []);
  e.target.value = "";                      // allow re-picking the same folder
  if (!all.length) return;
  // Prefer the 2 combined per-hemisphere labelmaps (thomasfull_{L,R}) — the whole
  // segmentation in 2 files instead of ~40 per-nucleus masks. Fall back to the
  // individual masks when THOMAS didn't emit the combined files.
  const rel = (f) => f.webkitRelativePath || f.name;
  const isFull = (f) => /thomasfull_[lr]\.nii(\.gz)?$/i.test(rel(f));
  const isRootNifti = (f) => { const s = rel(f).split("/"); return s.length === 2 && /\.nii(\.gz)?$/i.test(s[1]); };
  const hasFull = all.some(isFull);
  const files = all.filter((f) =>
    isRootNifti(f) || (hasFull ? isFull(f)
                               : rel(f).split("/").some((p) => p === "left" || p === "right")));
  if (!files.length) { $("labelmsg").textContent = "That folder isn't a THOMAS output (needs thomasfull_{L,R} or left/ + right/)."; return; }
  setLabelBusy(true);
  $("labelmsg").textContent = `Uploading ${files.length} THOMAS files…`;
  try {
    const fd = new FormData();
    // Carry each file's relative path AS its filename so the server rebuilds the
    // tree — one clean `files` list, no parallel path array.
    for (const f of files) fd.append("files", f, f.webkitRelativePath || f.name);
    const r = await fetch(`${API}/uploads/dir`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`);
    state.thomasDir = (await r.json()).path;
    runLabel();                             // label against THOMAS, like any atlas
  } catch (err) {
    setLabelBusy(false);
    $("labelmsg").textContent = `THOMAS upload failed: ${err.message}`;
  }
}

async function showProposed(id, { reloadViewer = true, jumpToQc = true } = {}) {
  setLabelBusy(false);
  try {
    const p = await jget(`${API}/jobs/${id}/labels`);
    // Already applied? Approval commits the proposed regions into the parent
    // ReviewDoc (state.doc), so if every proposed region is already committed
    // there, this atlas's labels were approved (survives reload). A different
    // atlas → regions differ → still needs applying.
    const proposed = (p.contacts || []).filter((c) => c.region);
    const approved = proposed.length > 0 && proposed.every((c) => {
      const sh = state.doc && state.doc.shanks.find((s) => s.name === c.shank);
      const ct = sh && sh.contacts.find((x) => x.index === c.index);
      return ct && ct.region === c.region;
    });
    $("labelmsg").innerHTML = approved
      ? `Labels from <strong>${p.atlas}</strong> applied ✓ ` +
        `(<strong>${p.n_labeled}/${p.n_contacts}</strong>). Pick another atlas to relabel.`
      : jumpToQc
        ? `Proposed <strong>${p.n_labeled}/${p.n_contacts}</strong> labels from ` +
          `<strong>${p.atlas}</strong>. Verify the <em>Registration</em> tab, then Apply.`
        : `Labels from <strong>${p.atlas}</strong> ` +
          `(<strong>${p.n_labeled}/${p.n_contacts}</strong>) — Apply to commit.`;
    $("approvebtn").hidden = approved;
    $("atlasctl").hidden = false;        // atlas picker + approve live on the 3D toolbar
    if (p.atlas) $("atlassel").value = p.atlas;   // reflect which atlas is shown
    _syncThomasLoadBtn();
    $("atlascheckbtn").hidden = true;    // the [Registration] tab is the entry now
    previewProposed(p.contacts);        // show the proposed regions per contact
    // showQc reveals the [Registration] tab + auto-jumps on the first label so the
    // registration gets verified; later atlas switches just refresh it in place.
    if (p.has_mri_qc) showQc(p.has_mni_qc, jumpToQc);
    // The 3D viewer is (re)built only on the first label; reload the iframe then
    // so the brain surface shows. Atlas switches leave the viewer untouched.
    const f = $("viewerframe");
    if (reloadViewer && f && state.jobId) f.src = `${API}/jobs/${state.jobId}/viewer/?t=${Date.now()}`;
  } catch (e) { $("labelmsg").textContent = `Could not read labels: ${e.message}`; }
}

// Fill the review list's region fields with a label job's PROPOSED regions
// (styled as proposed, not yet committed) so switching atlases visibly changes
// the labels. Approve commits them into the ReviewDoc; a review edit re-renders
// from the doc and clears the preview. Maps by shank+contact-index.
function previewProposed(contacts) {
  const list = $("reviewlist");
  list.querySelectorAll(".region.proposed").forEach((r) => r.classList.remove("proposed"));
  for (const c of contacts || []) {
    if (!c.region) continue;
    const row = list.querySelector(
      `.contact[data-shank="${CSS.escape(c.shank)}"][data-cindex="${c.index}"]`);
    const region = row && row.querySelector(".region");
    if (region) { region.value = c.region; region.classList.add("proposed"); }
  }
}

// Registration QC lives in the big viewer pane (a tab beside the 3D electrode
// view). Three orthogonal planes; each is a SINGLE <img> whose URL carries the
// full composite request (mode + value + slice). The SERVER composites CT+MRI
// (opacity / wipe / color), so there is no fragile browser overlay — a plane is
// exactly one image that we re-fetch when something changes.
const QC_PLANES = [[2, "Axial"], [1, "Coronal"], [0, "Sagittal"]];

function qcSrc(p) {
  const { mode, value, dir, space } = state.qc;
  // CT↔MRI (Native + AC-PC) is a per-CASE registration → the case job; the atlas
  // check (template↔MRI) is per-LABEL → the label job.
  const jid = space === "atlas" ? state.labelJobId : state.jobId;
  return `${API}/jobs/${jid}/qc?axis=${p.axis}&mode=${mode}` +
    `&value=${value.toFixed(3)}&dir=${dir}&frac=${p.frac.toFixed(3)}&space=${space}`;
}

// Load a FRESH <img> each time and swap it in on load. Mutating one <img>'s
// src in place proved unreliable (Safari would fetch the new image — visible in
// the server log — but not repaint the element). A new element always paints,
// and swapping on `load` avoids flicker (the old slice stays until the new one
// is ready).
function refreshPane(p) {
  if (!state.qc) return;
  const img = new Image();
  img.className = "qc-img";
  img.alt = p.label;
  img.onload = () => { if (state.qc && p.holder.isConnected) p.holder.replaceChildren(img); };
  img.src = qcSrc(p);
}

let _qcRaf = 0;
function refreshAllPanes() {
  if (!state.qc) return;
  cancelAnimationFrame(_qcRaf);   // coalesce rapid slider ticks into one frame
  _qcRaf = requestAnimationFrame(() => { for (const p of state.qc.panes) refreshPane(p); });
}

// Show/hide the space buttons per what's available (tracked on state.qc):
// Native (CT↔MRI) is always there once there's an MRI; AC-PC appears once its
// rigid reorientation is built; Atlas appears when an MNI atlas was labeled.
function _applyQcSpaces() {
  const avail = { ct: true, acpc: !!state.qc._hasAcpc, atlas: !!state.qc._hasAtlas };
  for (const b of $("qcspace").querySelectorAll("button")) b.hidden = !avail[b.dataset.space];
  if (!avail[state.qc.space]) state.qc.space = avail.acpc ? "acpc" : "ct";
  const cur = $("qcspace").querySelector(`[data-space="${state.qc.space}"]`);
  if (cur) setActive("qcspace", cur);
}

// Build (once, cached) the rigid AC-PC reorientation for the CASE. ~10s the
// first time; returns whether the AC-PC space is ready.
async function ensureAcpc() {
  if (!state.jobId || !state.mri) return false;
  try {
    const r = await jsend(`${API}/jobs/${state.jobId}/acpc`, "POST");
    return !!(r && r.ready);
  } catch (_e) { return false; }
}

// Reveal the AC-PC button once its reorientation is ready (keeps current space).
async function _buildAcpcInBg() {
  const ready = await ensureAcpc();
  if (!ready || !state.qc) return;
  state.qc._hasAcpc = true;
  _applyQcSpaces();
}

// Build the 3 QC panes once per case (registration is per-case). Atlas switches
// + the Reg chip re-enter but keep the panes + the user's mode/slice settings.
function _ensureQcPanes() {
  if (state.qc) return;
  state.qc = { mode: "color", value: 0.5, dir: "h", space: "ct", panes: [],
               _hasAcpc: false, _hasAtlas: false };
  const wrap = $("qcplanes");
  wrap.innerHTML = "";
  for (const [axis, name] of QC_PLANES) {
    const holder = el("div", { class: "qc-holder" });
    const slice = el("input", { type: "range", min: "2", max: "98", value: "50", class: "qc-slice" });
    const p = { axis, label: name, holder, frac: 0.5 };
    slice.addEventListener("input", () => { p.frac = Number(slice.value) / 100; refreshPane(p); });
    const pane = el("div", { class: "qc-pane" });
    pane.append(el("div", { class: "muted qc-plane-label" }, name), holder, slice);
    wrap.append(pane);
    state.qc.panes.push(p);
  }
  setActive("qcmodes", $("qcmodes").querySelector('[data-mode="color"]'));
  $("qcvaluewrap").style.visibility = "hidden";   // color needs no value slider
  $("qcdir").hidden = true;
}

// The label QC: verify the atlas registration that placed the labels, with the
// case-level CT↔MRI checks (Native + AC-PC) alongside. Defaults to the Atlas
// check (the labeling concern); AC-PC builds in the background.
function showQc(hasMni, jumpToQc = true) {
  const firstTime = !state.qc;
  _ensureQcPanes();
  state.qc._hasAtlas = hasMni;
  if (firstTime) state.qc.space = hasMni ? "atlas" : "ct";
  _applyQcSpaces();
  $("regtab").hidden = false;        // the [3D][Registration] toggle is now usable
  if (jumpToQc) setViewerTab("qc");
  else if (!$("viewerqc").hidden) refreshAllPanes();
  if (state.mri) _buildAcpcInBg();   // make the AC-PC button live
}

// The Reg chip: the CT↔MRI check straight from the CASE's own registration, so
// it works WITHOUT labeling. Shows Native immediately, then defaults to the
// rigid AC-PC reorientation (upright standard planes) once built (~10s, cached).
async function openRegCheck() {
  if (!state.jobId) return;
  if (!state.mri) { $("labelmsg").textContent = "Add a patient MRI to enable the CT↔MRI check."; return; }
  _ensureQcPanes();
  state.qc._hasAtlas = false;
  state.qc.space = "ct";             // Native immediately — always available
  _applyQcSpaces();
  $("regtab").hidden = false;
  setViewerTab("qc");
  $("labelmsg").textContent = "Aligning CT↔MRI to AC-PC planes…";
  const ready = await ensureAcpc();
  if (!state.qc) return;             // case reset while building — abandon
  if (ready) {
    state.qc._hasAcpc = true;
    state.qc.space = "acpc";         // default to the upright view
    _applyQcSpaces();
    if (!$("viewerqc").hidden) refreshAllPanes();
    $("labelmsg").textContent = "";
  } else {
    $("labelmsg").textContent = "AC-PC unavailable — showing native.";
  }
}

// Switch the big pane between the 3D electrode view and the registration QC, and
// SWAP the top-bar controls to match (via _syncModeControls) so only one set shows.
function setViewerTab(tab) {
  const qc = tab === "qc";
  state.viewMode = qc ? "qc" : "electrodes";
  $("viewerframe").hidden = qc;
  $("viewerqc").hidden = !qc;
  for (const b of $("viewertabs").querySelectorAll("button[data-tab]"))
    b.classList.toggle("active", b.dataset.tab === tab);
  _syncModeControls();
  if (qc) refreshAllPanes();      // (re)load images every time the QC is shown
}

function wireQc() {
  $("viewertabs").addEventListener("click", (ev) => {
    const b = ev.target.closest("button[data-tab]");
    if (b && !b.disabled) setViewerTab(b.dataset.tab);
  });
  $("qcspace").addEventListener("click", (ev) => {
    const b = ev.target.closest("button"); if (!b || b.disabled || b.hidden || !state.qc) return;
    state.qc.space = b.dataset.space;
    setActive("qcspace", b);
    refreshAllPanes();
  });
  $("qcmodes").addEventListener("click", (ev) => {
    const b = ev.target.closest("button"); if (!b || !state.qc) return;
    state.qc.mode = b.dataset.mode;
    setActive("qcmodes", b);
    const showVal = b.dataset.mode === "opacity" || b.dataset.mode === "wipe";
    $("qcvaluewrap").style.visibility = showVal ? "visible" : "hidden";
    $("qcdir").hidden = b.dataset.mode !== "wipe";
    refreshAllPanes();
  });
  $("qcvalue").addEventListener("input", (ev) => {
    if (!state.qc) return;
    state.qc.value = Number(ev.target.value) / 100;
    refreshAllPanes();
  });
  $("qcdir").addEventListener("click", () => {
    if (!state.qc) return;
    state.qc.dir = state.qc.dir === "h" ? "v" : "h";
    $("qcdir").textContent = state.qc.dir === "h" ? "⇄" : "⇅";
    refreshAllPanes();
  });
}

function setActive(groupId, btn) {
  for (const b of $(groupId).querySelectorAll("button")) b.classList.toggle("active", b === btn);
}

async function approveLabels() {
  if (!state.labelJobId) {
    $("labelmsg").textContent = "No proposed labels to apply — pick an atlas first.";
    return;
  }
  $("approvebtn").disabled = true;
  try {
    const doc = await jsend(`${API}/jobs/${state.labelJobId}/labels/approve`, "POST");
    renderReview(doc);   // regions now populated → visible per contact + in export
    const n = doc.shanks.reduce((a, s) => a + s.contacts.filter((c) => c.region).length, 0);
    $("labelmsg").textContent = `✓ Applied ${n} labels — shown per contact and included in the export.`;
    $("approvebtn").hidden = true;
    setViewerTab("electrodes");   // back to the 3-D view so the labeled contacts are visible
  } catch (e) {
    $("labelmsg").textContent = `Approve failed: ${e.message}`;
  } finally {
    $("approvebtn").disabled = false;
  }
}

// Stream a job's logs into a <pre> by id (used for the label job's reg metrics).
function streamInto(id, elId) {
  const es = new EventSource(`${API}/jobs/${id}/logs`);
  es.onmessage = (ev) => {
    if (!ev.data) return;
    const e = $(elId);
    e.textContent += ev.data + "\n";
    e.scrollTop = e.scrollHeight;
  };
  es.addEventListener("end", () => es.close());
  es.onerror = () => es.close();
}

// tiny element helper
function el(tag, attrs, text) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}

// ---- home: the case list ---------------------------------------------

async function showCases() {
  showStep("cases");
  try { state.cases = await jget(`${API}/cases`); }   // enriched: counts, MRI, labels
  catch (e) { $("caseslist").textContent = `Could not load cases: ${e.message}`; return; }
  renderCases();
}

// Cohort registry table. Columns are sortable client-side over the fetched
// cases (instant, no refetch). `num` columns default to descending on first
// click (biggest/newest first); text columns ascending.
const CASE_COLS = [
  { key: "label",        label: "Subject",    cls: "",        num: false },
  { key: "kind",         label: "Kind",       cls: "",        num: false },
  { key: "has_mri",      label: "MRI",        cls: "c-center", num: false },
  { key: "n_shanks",     label: "Electrodes", cls: "c-num",    num: true  },
  { key: "n_contacts",   label: "Contacts",   cls: "c-num",    num: true  },
  { key: "labeled",      label: "Labeled",    cls: "c-center", num: false },
  { key: "atlas",        label: "Atlas",      cls: "",        num: false },
  { key: "mni_eligible", label: "MNI",        cls: "c-center", num: false },
  { key: "created_at",   label: "Date",       cls: "",        num: true  },
];

function renderCases() {
  const all = state.cases || [];
  const q = (state.caseSearch || "").trim().toLowerCase();
  const f = state.caseFilter || "all";
  const shown = sortCases(all.filter((c) =>
    (f === "all" || c.kind === f) &&
    (!q || (c.label || "").toLowerCase().includes(q)
        || c.id.toLowerCase().includes(q)
        || (c.ct_hash || "").toLowerCase().includes(q))));   // scan hash is searchable too
  $("casescount").textContent = all.length ? `· ${all.length}` : "";
  $("casesempty").hidden = all.length > 0;
  $("casesnoresults").hidden = !(all.length > 0 && shown.length === 0);
  const list = $("caseslist");
  list.innerHTML = "";
  if (shown.length) list.append(caseTable(shown));
}

function sortCases(rows) {
  const sort = state.caseSort || { key: "created_at", dir: -1 };
  const v = (c) => {
    switch (sort.key) {
      case "label":        return (c.label || c.id || "").toLowerCase();
      case "kind":         return c.kind || "";
      case "has_mri":      return c.has_mri ? 1 : 0;
      case "n_shanks":     return c.n_shanks || 0;
      case "n_contacts":   return c.n_contacts || 0;
      case "labeled":      return c.labeled ? 1 : 0;
      case "atlas":        return (c.atlas || "").toLowerCase();
      case "mni_eligible": return c.mni_eligible ? 1 : 0;
      default:             return c.created_at || 0;
    }
  };
  return rows.slice().sort((a, b) => {
    const va = v(a), vb = v(b);
    if (va < vb) return -sort.dir;
    if (va > vb) return sort.dir;
    return (b.created_at || 0) - (a.created_at || 0);   // tiebreak: newest first
  });
}

function caseTable(rows) {
  const sort = state.caseSort || { key: "created_at", dir: -1 };
  const table = el("table", { class: "casetable" });
  const htr = el("tr");
  for (const col of CASE_COLS) {
    const on = sort.key === col.key;
    const th = el("th", { class: (col.cls + (on ? " sorted" : "")).trim() },
      col.label + (on ? (sort.dir < 0 ? " ▾" : " ▴") : ""));
    th.onclick = () => {
      const cur = state.caseSort || { key: "created_at", dir: -1 };
      state.caseSort = cur.key === col.key
        ? { key: col.key, dir: -cur.dir }
        : { key: col.key, dir: col.num ? -1 : 1 };
      renderCases();
    };
    htr.append(th);
  }
  htr.append(el("th", { class: "c-center" }));   // delete-action column
  const thead = el("thead"); thead.append(htr);
  const tbody = el("tbody");
  for (const c of rows) tbody.append(caseRow(c));
  table.append(thead, tbody);
  return table;
}

function ynCell(v) {
  const td = el("td", { class: "c-center" });
  td.append(el("span", { class: v ? "yes" : "no" }, v ? "✓" : "–"));
  return td;
}

function atlasName(a) {
  if (!a) return "";
  if (a.startsWith("thomas")) return "THOMAS";
  return { cerebra: "CerebrA", fastsurfer: "FastSurfer", deepmriprep: "deepmriprep" }[a] || a;
}

function caseRow(c) {
  const tr = el("tr", { class: "caserow", tabindex: "0", role: "button" });
  tr.onclick = () => openCase(c.id);
  tr.onkeydown = (e) => { if (e.key === "Enter") openCase(c.id); };
  const when = c.created_at ? new Date(c.created_at * 1000).toLocaleDateString(
    undefined, { month: "short", day: "numeric", year: "numeric" }) : "";
  const scan = (c.ct_hash || c.id || "").slice(0, 8);

  const subj = el("td");
  subj.append(el("div", { class: "ct-name" }, c.label || scan),
              el("div", { class: "ct-scan" }, scan));
  tr.append(subj);

  const kindTd = el("td");
  kindTd.append(el("span", { class: `cc-badge ${c.kind}` }, c.kind === "import" ? "imported" : "detected"));
  tr.append(kindTd);

  tr.append(ynCell(c.has_mri));
  tr.append(el("td", { class: "c-num" }, String(c.n_shanks || 0)));
  tr.append(el("td", { class: "c-num" }, String(c.n_contacts || 0)));
  tr.append(ynCell(c.labeled));

  const atlasTd = el("td");
  const an = atlasName(c.atlas);
  atlasTd.append(an ? el("span", {}, an) : el("span", { class: "no" }, "–"));
  tr.append(atlasTd);

  const mniTd = el("td", { class: "c-center" });
  mniTd.append(c.mni_eligible
    ? el("span", { class: "mni-dot", title: "Contacts poolable in MNI (CerebrA-registered)" })
    : el("span", { class: "no", title: "Label with CerebrA (through MRI) to pool this subject in MNI" }, "–"));
  tr.append(mniTd);

  tr.append(el("td", { class: "ct-date" }, when));

  const delTd = el("td", { class: "c-center" });
  const del = el("button", { class: "ct-del", type: "button", title: "Delete case" }, "×");
  del.onclick = (ev) => { ev.stopPropagation(); deleteCase(c); };
  delTd.append(del);
  tr.append(delTd);
  return tr;
}

async function deleteCase(c) {
  if (!window.confirm(`Delete case "${c.label || c.id.slice(0, 8)}"? This removes its files — it can't be undone.`))
    return;
  try {
    const r = await fetch(`${API}/cases/${c.id}`, { method: "DELETE" });
    if (!r.ok && r.status !== 204) throw new Error(await r.text());
  } catch (e) { window.alert(`Delete failed: ${e.message}`); return; }
  state.cases = (state.cases || []).filter((x) => x.id !== c.id);
  renderCases();
}

// Open an existing case: load its results, then restore its newest label job
// (labels/QC) so the case comes back exactly as it was left.
async function openCase(id) {
  $("showallbtn").hidden = true;   // fresh case: viewer loads with all electrodes shown
  $("sliceovctl").hidden = true;   // re-shown when the viewer reports its slice caps
  $("probemodectl").hidden = true; // re-shown (Ortho) once the viewer loads
  state.probePref = "ortho";
  await loadResults(id);
  try {
    const jobs = await jget(`${API}/jobs`);
    const labelJobs = jobs.filter((j) => j.kind === "label" && j.parent === id);
    const newest = labelJobs[0];   // /jobs is newest-first
    if (!newest) return;
    state.labeledOnce = labelJobs.some((j) => j.state === "succeeded");
    state.labelJobId = newest.id;
    if (newest.t1) { state.mri = { path: newest.t1, name: "(uploaded MRI)" }; $("labelbtn").disabled = false; }
    if (newest.atlas) { $("atlassel").value = newest.atlas; _syncThomasLoadBtn(); }
    if (newest.state === "succeeded") showProposed(newest.id, { reloadViewer: false, jumpToQc: false });
    else if (["failed", "cancelled"].includes(newest.state)) _showLabelError(newest);
    else pollLabel(newest.id, !state.labeledOnce);
  } catch (_e) { /* a case with no label job is fine */ }
}

// ---- import a localization computed elsewhere ------------------------

// browse → real path (desktop) or upload (browser) → fill the paired path field.
async function importBrowse(fileInputId, pathId) {
  const f = $(fileInputId).files[0];
  if (!f) return;
  $("imp-msg").textContent = IS_DESKTOP ? `Loading ${f.name}…` : `Uploading ${f.name}…`;
  try { $(pathId).value = (await resolveFile(f)).path; $("imp-msg").textContent = ""; }
  catch (e) { $("imp-msg").textContent = `Failed: ${e.message}`; }
}

function showImportCheck(check, isError, message) {
  const c = $("imp-check");
  c.hidden = false;
  const v = (check && check.verdict) || (isError ? "red" : "");
  c.className = "check " + v;
  c.innerHTML = "";
  c.append(el("span", { class: "check-pill" }, v || "error"),
           el("span", { class: "check-reason" }, message || (check && check.reason) || ""));
  if (check) c.append(el("span", { class: "check-stat" },
    `${check.n_on_metal}/${check.n} on metal · ${check.n_in_bounds}/${check.n} in bounds`));
}

async function runImport(force = false) {
  const ct = $("imp-ct").value.trim(), contacts = $("imp-contacts").value.trim(),
        traj = $("imp-traj").value.trim(), t1 = $("imp-t1").value.trim(),
        label = $("imp-label").value.trim() || "imported";
  if (!ct || !contacts || !traj) {
    $("imp-msg").textContent = "CT, contacts TSV, and trajectories TSV are all required.";
    return;
  }
  $("imp-msg").textContent = "Checking the CT ↔ TSV match…";
  $("imp-check").hidden = true;
  let r, body;
  try {
    r = await fetch(`${API}/jobs/import`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ct, contacts, trajectories: traj, label, force, ...(t1 ? { t1 } : {}) }),
    });
    body = await r.json().catch(() => ({}));
  } catch (e) { $("imp-msg").textContent = `Import failed: ${e.message}`; return; }
  $("imp-msg").textContent = "";
  if (!r.ok) {
    const d = body.detail;
    if (d && typeof d === "object" && d.existing)          // duplicate CT
      return promptDuplicate(d, () => runImport(true));
    if (d && typeof d === "object" && d.check) showImportCheck(d.check, true, d.message);
    else showImportCheck(null, true, typeof d === "string" ? d : "Import failed.");
    return;
  }
  // green/yellow: show the match, then run the (view-results) job like a pipeline.
  showImportCheck(body.check, false);
  const job = body.job;
  state.jobId = job.id;
  showStep("run");
  $("log").textContent = "";
  $("runstate").textContent = "Importing…";
  $("runsub").textContent = "build viewer" + (t1 ? " + strip MRI + surface" : "") + " from imported localization";
  $("spinner").classList.remove("stopped");
  streamLogs(job.id);
  pollStatus(job.id);
}

// ---- boot -------------------------------------------------------------

async function boot() {
  wireDrop();
  $("runbtn").onclick = run;
  $("etypesall").onclick = () => setAllEtypes(true);
  $("etypesnone").onclick = () => setAllEtypes(false);
  $("ctpickbtn").onclick = () => pickImage("ct");
  $("mripickbtn").onclick = () => pickImage("mri");
  $("cancelbtn").onclick = cancel;
  $("exportbtn").onclick = doExport;
  document.querySelectorAll("#wsflow button").forEach((b) => { b.onclick = () => showWs(b.dataset.ws); });
  window.addEventListener("message", (ev) => {   // viewer/editor iframe → app
    const d = ev.data; if (!d) return;
    if (d.type === "rosa:edited") onEdited(d.rebuild);
    else if (d.type === "rosa:slice-caps") _onSliceCaps(d);
    else if (d.type === "rosa:probe-avail") _onProbeAvail(d);
    else if (d.type === "rosa:selected") _onViewerSelected(d);
  });
  $("slot-mri").onclick = () => {
    if (!state.mri) { showWs("review"); $("labelbar").scrollIntoView({ behavior: "smooth", block: "nearest" }); $("mriinput").focus(); }
  };
  $("slot-reg").onclick = () => { showWs("review"); openRegCheck(); };  // case CT↔MRI QC (pre-label)
  $("restartbtn").onclick = showCases;          // workspace → back to the case list
  $("newcasebtn").onclick = restart;            // case list → fresh new-case (drop) form
  $("importbtn").onclick = () => showStep("import");
  // Cohort MNI viewer (Data Explorer): reload the iframe each time so newly
  // labeled cases get pooled in.
  $("cohortbtn").onclick = () => { $("cohortframe").src = "/cohort/?t=" + Date.now(); showStep("cohort"); };
  $("cohortback").onclick = showCases;
  $("rosaimportbtn").onclick = importRosa;      // import a ROSA robot case
  $("rosa-cancel").onclick = showCases;
  $("rosa-create").onclick = createRosaCase;
  $("importback").onclick = showCases;
  $("casesearch").addEventListener("input", (ev) => { state.caseSearch = ev.target.value; renderCases(); });
  $("casesfilter").addEventListener("click", (ev) => {
    const b = ev.target.closest("button"); if (!b) return;
    state.caseFilter = b.dataset.f; setActive("casesfilter", b); renderCases();
  });
  $("imp-run").onclick = runImport;
  [["imp-ct-file", "imp-ct"], ["imp-contacts-file", "imp-contacts"],
   ["imp-traj-file", "imp-traj"], ["imp-t1-file", "imp-t1"]].forEach(([f, p]) =>
    $(f).addEventListener("change", () => importBrowse(f, p)));
  $("mriinput").addEventListener("change", (ev) => {
    const f = ev.target.files[0]; if (f) uploadMri(f);
  });
  $("labelbtn").onclick = runLabel;
  $("approvebtn").onclick = approveLabels;
  $("showallbtn").onclick = showAllInViewer;
  // THOMAS is a BYO atlas: selecting it in the dropdown opens the folder picker
  // (below), which uploads the masks and labels against them via runLabel.
  $("thomasdir-file").addEventListener("change", uploadThomasDir);
  // Slice-display controls hosted in the top row → drive the embedded viewer.
  $("app-slice-fade").addEventListener("input", (e) =>
    _postViewer({ type: "rosa:slice-fade", value: parseFloat(e.target.value) }));
  $("app-atlas-ov").addEventListener("change", (e) =>
    _postViewer({ type: "rosa:atlas-overlay", on: e.target.checked }));
  $("app-slice-mode").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-smode]"); if (!b || b.disabled) return;
    state.probePref = b.dataset.smode;   // remember the choice
    _setAppSliceMode(b.dataset.smode);
    _postViewer({ type: "rosa:slice-mode", mode: b.dataset.smode });
  });
  // ↑/↓ step contacts within the selected shank from anywhere in the app (the
  // viewer handles them when it has focus; this covers the left navigator).
  window.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    const tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    const cur = document.querySelector("#reviewlist .contact.selected");
    if (!cur) return;
    const rows = [...(cur.closest(".shank")?.querySelectorAll(".contact") || [])];
    const i = rows.indexOf(cur);
    const j = Math.max(0, Math.min(rows.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)));
    if (i < 0 || j === i) return;
    e.preventDefault();
    selectInViewer(rows[j].dataset.label, rows[j].dataset.shank);
  });
  // Selecting a different atlas re-labels immediately (registration is cached,
  // so only the atlas warp + sampling re-runs) — so the labels track the atlas.
  $("atlassel").addEventListener("change", () => {
    _syncThomasLoadBtn();                 // show/hide the "Load folder…" chip
    if (!state.jobId) return;
    // THOMAS: prompt for the folder first; the picker's change handler uploads
    // then labels. Every other atlas labels immediately (needs the patient MRI).
    if ($("atlassel").value.startsWith("thomas")) { pickThomasDir(); return; }
    if (state.mri) runLabel();
  });
  // The "Load folder…" chip: works even when THOMAS is already the selected atlas
  // (re-open / re-import), where the dropdown's change never fires.
  $("thomasload").onclick = pickThomasDir;
  wireQc();
  await loadAtlases();   // populate the picker before a resume sets its value
  loadElectrodeTypes();  // new-case electrode-type checkboxes (default all)
  try {
    const h = await jget("/healthz");
    $("engine").textContent = `engine ${h.engine_version} · ${h.engine_import_ok ? "ready" : "NOT LINKED"}`;
  } catch { $("engine").textContent = "service unreachable"; }
  // Land on the case list (the front door). Opening a case restores its viewer,
  // reviewed contacts, and newest label job — see openCase().
  await showCases();
  // Deep-link: /#case=<id> opens that case directly (bookmark / Electron link).
  const m = location.hash.match(/case=([a-z0-9]+)/i);
  if (m && (state.cases || []).some((c) => c.id === m[1])) openCase(m[1]);
}
boot();
