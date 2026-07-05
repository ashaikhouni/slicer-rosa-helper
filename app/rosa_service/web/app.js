"use strict";
// ROSA web UI — a single-page wizard over the local service contract.
// Drop CT → run pipeline job → review/edit contacts + 3D viewer → export.
// Talks ONLY to /api/v1 + /healthz (never the engine directly).

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const state = { ct: null, jobId: null, es: null, poll: null,
                mri: null, labelJobId: null, labelPoll: null, qc: null };

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

function showStep(name) {
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
  $("ctinfo").textContent = `Uploading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/uploads`, { method: "POST", body: fd });
  if (!r.ok) { $("ctinfo").textContent = `Upload failed: ${await r.text()}`; return; }
  setCt(await r.json());
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
  $("ctpath").addEventListener("input", (ev) => {
    const p = ev.target.value.trim();
    setCt(p ? { path: p, name: p.split("/").pop() } : null);
  });
}

// ---- step 2: run ------------------------------------------------------

async function run() {
  const label = $("label").value.trim() || "case";
  showStep("run");
  $("log").textContent = "";
  $("runstate").textContent = "Starting…";
  $("spinner").classList.remove("stopped");
  try {
    const job = await jsend(`${API}/jobs`, "POST",
      { kind: "pipeline", params: { ct: state.ct.path, label } });
    state.jobId = job.id;
    streamLogs(job.id);
    pollStatus(job.id);
  } catch (e) {
    appendLog(`error: ${e.message}`, "err");
    $("runstate").textContent = "Failed to start";
    $("spinner").classList.add("stopped");
  }
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

async function cancel() {
  if (state.jobId) { try { await fetch(`${API}/jobs/${state.jobId}`, { method: "DELETE" }); } catch {} }
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
  showStep("results");
  resetLabelCard();
  try { renderReview(await jget(`${API}/jobs/${id}/review`)); }
  catch (e) { $("reviewlist").textContent = `Could not load review: ${e.message}`; }
}

// The label card is per-job: a fresh run starts with no MRI / no proposal.
function resetLabelCard() {
  clearInterval(state.labelPoll);
  state.mri = null; state.labelJobId = null;
  $("mriinput").value = "";
  $("labelbtn").disabled = true;
  $("approvebtn").hidden = true;
  $("labelstatus").textContent = "";
  $("labelmsg").textContent = "";
  const ll = $("labellog"); ll.hidden = true; ll.textContent = "";
  $("qcbox").hidden = true; $("qcplanes").innerHTML = ""; state.qc = null;
}

function renderReview(doc) {
  state.doc = doc;
  const nShanks = doc.shanks.length;
  const nContacts = doc.shanks.reduce((a, s) => a + s.contacts.length, 0);
  const nKept = doc.shanks.filter((s) => s.accepted)
    .reduce((a, s) => a + s.contacts.filter((c) => c.accepted).length, 0);
  $("summary").textContent = `— ${nShanks} shanks, ${nKept}/${nContacts} kept`;

  const list = $("reviewlist");
  list.innerHTML = "";
  for (const shank of doc.shanks) {
    const box = document.createElement("div");
    box.className = "shank" + (shank.accepted ? "" : " rejected");

    const head = document.createElement("div");
    head.className = "shank-head";
    const sacc = el("input", { type: "checkbox" });
    sacc.className = "sacc"; sacc.checked = shank.accepted;
    sacc.onchange = () => patch([{ op: sacc.checked ? "accept_shank" : "reject_shank", shank: shank.name }]);
    head.append(sacc, el("strong", {}, shank.name),
      el("span", { class: "muted" }, `${shank.model || "—"} · ${shank.contacts.length} contacts`));
    box.append(head);

    const cs = document.createElement("div");
    cs.className = "contacts";
    for (const c of shank.contacts) {
      const row = document.createElement("div");
      row.className = "contact" + (c.accepted ? "" : " rejected");
      row.dataset.label = c.name;
      // Click anywhere on the row (except the checkbox / region field) → show
      // the contact in the 3D viewer.
      row.onclick = (ev) => { if (!ev.target.closest("input")) selectInViewer(c.name, shank.name); };
      const acc = el("input", { type: "checkbox" });
      acc.checked = c.accepted; acc.disabled = !shank.accepted;
      acc.onchange = () => patch([{ op: acc.checked ? "accept_contact" : "reject_contact", shank: shank.name, index: c.index }]);
      const name = el("span", { class: "cname", title: "show in 3D" }, c.name);
      const region = el("input", { type: "text", class: "region", value: c.region || "", placeholder: "—" });
      region.disabled = !shank.accepted;
      region.onchange = () => {
        if (region.value.trim())
          patch([{ op: "relabel_contact", shank: shank.name, index: c.index, region: region.value.trim() }]);
      };
      row.append(acc, name, region);
      cs.append(row);
    }
    box.append(cs);
    list.append(box);
  }
  syncVisibility(doc);
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
      const lic = (a.license || "").split("(")[0].trim();
      const o = el("option", { value: a.id }, `${short}${lic ? ` — ${lic}` : ""}`);
      if (!a.available) { o.disabled = true; o.textContent += " (not installed)"; }
      if (a.id === def) o.selected = true;
      sel.append(o);
    }
  } catch (_e) { $("labelstatus").textContent = "· atlas list unavailable"; }
}

async function uploadMri(file) {
  $("labelstatus").textContent = `· uploading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/uploads`, { method: "POST", body: fd });
  if (!r.ok) { $("labelstatus").textContent = "· MRI upload failed"; return; }
  state.mri = await r.json();
  $("labelstatus").textContent = `· MRI ${state.mri.name}`;
  $("labelbtn").disabled = false;
}

async function runLabel() {
  if (!state.mri || !state.jobId) return;
  $("labelbtn").disabled = true;
  $("approvebtn").hidden = true;
  const ll = $("labellog"); ll.hidden = false; ll.textContent = "";
  $("labelmsg").textContent = "Registering MRI → CT and warping atlas (~30 s)…";
  try {
    const job = await jsend(`${API}/jobs/${state.jobId}/label`, "POST",
      { t1: state.mri.path, atlas: $("atlassel").value });
    state.labelJobId = job.id;
    streamInto(job.id, "labellog");     // shows the registration metrics (QC)
    pollLabel(job.id);
  } catch (e) {
    $("labelmsg").textContent = `Failed to start: ${e.message}`;
    $("labelbtn").disabled = false;
  }
}

function pollLabel(id) {
  clearInterval(state.labelPoll);
  state.labelPoll = setInterval(async () => {
    let st;
    try { st = await jget(`${API}/jobs/${id}`); } catch { return; }
    if (["succeeded", "failed", "cancelled"].includes(st.state)) {
      clearInterval(state.labelPoll);
      $("labelbtn").disabled = false;
      if (st.state === "succeeded") showProposed(id);
      else $("labelmsg").textContent =
        `Labeling ${st.state}${st.error ? ": " + st.error : ` (exit ${st.exit_code})`}`;
    }
  }, 1000);
}

async function showProposed(id) {
  try {
    const p = await jget(`${API}/jobs/${id}/labels`);
    $("labelmsg").innerHTML =
      `Proposed <strong>${p.n_labeled}/${p.n_contacts}</strong> labels from ` +
      `<strong>${p.atlas}</strong>. Evaluate the registration below, then approve.`;
    $("approvebtn").hidden = false;
    if (p.has_mri_qc) showQc();
  } catch (e) { $("labelmsg").textContent = `Could not read labels: ${e.message}`; }
}

// Registration QC: three orthogonal planes, each a CT slice with the MRI slice
// stacked on top. A global comparison mode composites the two IN THE BROWSER so
// the sliders are smooth (no re-fetch): Opacity fades MRI over CT; Wipe reveals
// MRI up to a draggable split (⇄ / ⇅); Color uses the server magenta/green blend.
const QC_PLANES = [[2, "Axial"], [1, "Coronal"], [0, "Sagittal"]];

function qcUrl(axis, frac, mode) {
  return `${API}/jobs/${state.labelJobId}/qc?axis=${axis}&mode=${mode}` +
    `&frac=${frac.toFixed(3)}`;
}

function showQc() {
  state.qc = { mode: "opacity", value: 0.5, dir: "h", frac: { 2: 0.5, 1: 0.5, 0: 0.5 } };
  const wrap = $("qcplanes");
  wrap.innerHTML = "";
  for (const [axis, name] of QC_PLANES) {
    const pane = el("div", { class: "qc-pane" });
    const stack = el("div", { class: "qc-stack" });
    const base = el("img", { class: "qc-base", alt: `${name} CT` });
    const over = el("img", { class: "qc-over", alt: `${name} MRI` });
    stack.append(base, over);
    const slice = el("input", { type: "range", min: "2", max: "98", value: "50", class: "qc-slice" });
    slice.dataset.axis = axis;
    slice.addEventListener("input", () => {
      state.qc.frac[axis] = Number(slice.value) / 100;
      loadPlane(axis);
      applyComparison();
    });
    pane.append(stack, slice, el("div", { class: "muted qc-plane-label" }, name));
    pane.dataset.axis = axis;
    wrap.append(pane);
  }
  $("qcbox").hidden = false;
  setActive("qcmodes", $("qcmodes").querySelector('[data-mode="opacity"]'));
  $("qcvalue").value = 50;
  for (const [axis] of QC_PLANES) loadPlane(axis);
  applyComparison();
}

// (Re)point a plane's two <img> at the current slice, per the active mode.
function loadPlane(axis) {
  const pane = $("qcplanes").querySelector(`.qc-pane[data-axis="${axis}"]`);
  if (!pane) return;
  const base = pane.querySelector(".qc-base");
  const over = pane.querySelector(".qc-over");
  const frac = state.qc.frac[axis];
  if (state.qc.mode === "color") {
    base.src = qcUrl(axis, frac, "blend");
    over.removeAttribute("src");
  } else {
    base.src = qcUrl(axis, frac, "ct");
    over.src = qcUrl(axis, frac, "mri");
  }
}

// Composite the MRI over the CT for every plane, per mode + slider value.
function applyComparison() {
  const { mode, value, dir } = state.qc;
  for (const [axis] of QC_PLANES) {
    const pane = $("qcplanes").querySelector(`.qc-pane[data-axis="${axis}"]`);
    if (!pane) continue;
    const over = pane.querySelector(".qc-over");
    if (mode === "color") { over.style.opacity = "0"; over.style.clipPath = ""; continue; }
    if (mode === "opacity") {
      over.style.opacity = String(value);
      over.style.clipPath = "";
    } else { // wipe
      over.style.opacity = "1";
      const pct = Math.round(value * 100);
      over.style.clipPath = dir === "h"
        ? `inset(0 ${100 - pct}% 0 0)`   // reveal MRI from the left
        : `inset(0 0 ${100 - pct}% 0)`;  // reveal MRI from the top
    }
  }
}

function wireQc() {
  $("qcmodes").addEventListener("click", (ev) => {
    const b = ev.target.closest("button"); if (!b || !state.qc) return;
    state.qc.mode = b.dataset.mode;
    setActive("qcmodes", b);
    $("qcvaluewrap").style.visibility = b.dataset.mode === "color" ? "hidden" : "visible";
    $("qcdir").hidden = b.dataset.mode !== "wipe";
    for (const [axis] of QC_PLANES) loadPlane(axis);
    applyComparison();
  });
  $("qcvalue").addEventListener("input", (ev) => {
    if (!state.qc) return;
    state.qc.value = Number(ev.target.value) / 100;
    applyComparison();
  });
  $("qcdir").addEventListener("click", () => {
    if (!state.qc) return;
    state.qc.dir = state.qc.dir === "h" ? "v" : "h";
    $("qcdir").textContent = state.qc.dir === "h" ? "⇄" : "⇅";
    applyComparison();
  });
}

function setActive(groupId, btn) {
  for (const b of $(groupId).querySelectorAll("button")) b.classList.toggle("active", b === btn);
}

async function approveLabels() {
  if (!state.labelJobId) return;
  try {
    const doc = await jsend(`${API}/jobs/${state.labelJobId}/labels/approve`, "POST");
    renderReview(doc);   // regions now populated → visible per contact + in export
    $("labelmsg").textContent = "Labels applied — shown per contact and included in the export.";
    $("approvebtn").hidden = true;
  } catch (e) { $("labelmsg").textContent = `Approve failed: ${e.message}`; }
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

// ---- boot -------------------------------------------------------------

async function boot() {
  wireDrop();
  $("runbtn").onclick = run;
  $("cancelbtn").onclick = cancel;
  $("exportbtn").onclick = doExport;
  $("restartbtn").onclick = restart;
  $("mriinput").addEventListener("change", (ev) => {
    const f = ev.target.files[0]; if (f) uploadMri(f);
  });
  $("labelbtn").onclick = runLabel;
  $("approvebtn").onclick = approveLabels;
  wireQc();
  loadAtlases();
  try {
    const h = await jget("/healthz");
    $("engine").textContent = `engine ${h.engine_version} · ${h.engine_import_ok ? "ready" : "NOT LINKED"}`;
  } catch { $("engine").textContent = "service unreachable"; }
  // Resume the most recent completed run so a reload lands on results, not a
  // blank form (single-user local app).
  try {
    const jobs = await jget(`${API}/jobs`);        // newest first
    const done = jobs.find((j) => j.state === "succeeded");
    if (done) { state.jobId = done.id; loadResults(done.id); }
  } catch (_e) { /* ignore */ }
}
boot();
