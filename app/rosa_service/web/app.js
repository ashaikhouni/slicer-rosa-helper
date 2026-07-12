"use strict";
// ROSA web UI — a single-page wizard over the local service contract.
// Drop CT → run pipeline job → review/edit contacts + 3D viewer → export.
// Talks ONLY to /api/v1 + /healthz (never the engine directly).

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const state = { ct: null, jobId: null, es: null, poll: null,
                mri: null, labelJobId: null, labelPoll: null, qc: null,
                creationMri: null };   // optional MRI (T1) picked at case creation

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
  document.body.classList.toggle("results-active", name === "results");
  // The step bar tracks the new-case wizard (drop → run → review); it's noise on
  // the case list, the import form, and the workspace (which has its own flow).
  document.body.classList.toggle("nosteps", ["cases", "import", "results"].includes(name));
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
  // Optional MRI (T1) at case creation — upload or point at a path.
  $("mrifileinput").addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (f) uploadCreationMri(f);
  });
  $("mripath").addEventListener("input", (ev) => {
    const p = ev.target.value.trim();
    setCreationMri(p ? { path: p, name: p.split("/").pop() } : null);
  });
}

function setCreationMri(mri) {
  state.creationMri = mri;
  $("mricreateinfo").textContent = mri
    ? `MRI: ${mri.name}${mri.bytes ? ` (${(mri.bytes / 1e6).toFixed(1)} MB)` : ""}`
    : "";
}

async function uploadCreationMri(file) {
  $("mricreateinfo").textContent = `Uploading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/uploads`, { method: "POST", body: fd });
  if (!r.ok) { $("mricreateinfo").textContent = `Upload failed: ${await r.text()}`; return; }
  setCreationMri(await r.json());
}

// ---- step 2: run ------------------------------------------------------

async function run() {
  const label = $("label").value.trim() || "case";
  const params = { ct: state.ct.path, label };
  if (state.creationMri) {
    params.t1 = state.creationMri.path;                        // MRI from the start
    params.surface = $("surfacesel").value;                    // brain-surface backend
  }
  showStep("run");
  $("log").textContent = "";
  $("runstate").textContent = "Starting…";
  // The MRI adds brain-strip + surface reconstruction to the run.
  $("runsub").textContent = state.creationMri
    ? "detect → place contacts → strip MRI + build surface → viewer"
    : "detect → place contacts → build viewer (~2–3 min)";
  $("spinner").classList.remove("stopped");
  try {
    const job = await jsend(`${API}/jobs`, "POST",
      { kind: "pipeline", params });
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

function setCaseSlots() {
  const mriOn = !!state.mri;
  const m = $("slot-mri");
  m.className = "slot " + (mriOn ? "on" : "off");
  m.querySelector(".fn").textContent = mriOn ? "loaded" : "add";
  $("slot-reg").hidden = !mriOn;
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
  $("qcspace").hidden = true;
  $("tab-qc").disabled = true;
  $("atlaschip").hidden = true;
  setViewerTab("electrodes");
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
      row.dataset.shank = shank.name;
      row.dataset.cindex = c.index;
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
  setCaseSlots();
}

async function runLabel() {
  if (!state.mri || !state.jobId) return;
  // First label of a case registers CT↔MRI (verify once); later atlas switches
  // reuse that registration and only recompute labels.
  const firstLabel = !state.labeledOnce;
  $("labelbtn").disabled = true;
  $("approvebtn").hidden = true;
  const ll = $("labellog"); ll.hidden = false; ll.textContent = "";
  $("labelmsg").textContent = firstLabel
    ? "Registering MRI → CT and warping atlas (~30 s)…"
    : `Relabeling with ${$("atlassel").value} — CT↔MRI registration is cached…`;
  try {
    const job = await jsend(`${API}/jobs/${state.jobId}/label`, "POST",
      { t1: state.mri.path, atlas: $("atlassel").value });
    state.labelJobId = job.id;
    streamInto(job.id, "labellog");     // shows the registration metrics (QC)
    pollLabel(job.id, firstLabel);
  } catch (e) {
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
  $("labelmsg").innerHTML =
    `Labeling <strong>${st.state}</strong>${st.error ? ": " + st.error : ` (exit ${st.exit_code})`}. ` +
    `Fix and click <em>Register MRI &amp; label</em> to retry.`;
  $("labelbtn").disabled = false;
}

async function showProposed(id, { reloadViewer = true, jumpToQc = true } = {}) {
  try {
    const p = await jget(`${API}/jobs/${id}/labels`);
    $("labelmsg").innerHTML = jumpToQc
      ? `Proposed <strong>${p.n_labeled}/${p.n_contacts}</strong> labels from ` +
        `<strong>${p.atlas}</strong>. Verify the <em>Registration</em> tab, then Apply.`
      : `Labels from <strong>${p.atlas}</strong> ` +
        `(<strong>${p.n_labeled}/${p.n_contacts}</strong>) — Apply to commit.`;
    $("approvebtn").hidden = false;
    if (p.atlas) {
      $("atlassel").value = p.atlas;   // reflect which atlas is shown
      $("atlaschip").textContent = `Atlas: ${p.atlas}`;
      $("atlaschip").hidden = false;   // show the active atlas beside the viewer tabs
    }
    previewProposed(p.contacts);        // show the proposed regions per contact
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
  return `${API}/jobs/${state.labelJobId}/qc?axis=${p.axis}&mode=${mode}` +
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

function showQc(hasMni, jumpToQc = true) {
  // Build the QC panes ONCE per case (registration is per-case). Atlas switches
  // re-enter here but keep the existing panes + the user's mode/slice settings.
  if (!state.qc) {
    // AC-PC (MNI) planes are the neuroanatomical standard, so default to them
    // when available; otherwise slice the CT's native frame.
    state.qc = { mode: "color", value: 0.5, dir: "h", space: hasMni ? "mni" : "ct", panes: [] };
    $("qcspace").hidden = !hasMni;
    if (hasMni) setActive("qcspace", $("qcspace").querySelector('[data-space="mni"]'));
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
  $("tab-qc").disabled = false;
  if (jumpToQc) setViewerTab("qc");                 // first label: verify registration
  else if (!$("viewerqc").hidden) refreshAllPanes();  // already on QC: refresh for the new job
}

// Switch the big pane between the 3D electrode view and the registration QC.
function setViewerTab(tab) {
  const qc = tab === "qc";
  $("viewerframe").hidden = qc;
  $("viewerqc").hidden = !qc;
  $("qctools").hidden = !qc;
  for (const b of $("viewertabs").querySelectorAll("button[data-tab]"))
    b.classList.toggle("active", b.dataset.tab === tab);
  if (qc) refreshAllPanes();      // (re)load images every time the QC is shown
}

function wireQc() {
  $("viewertabs").addEventListener("click", (ev) => {
    const b = ev.target.closest("button[data-tab]");
    if (b && !b.disabled) setViewerTab(b.dataset.tab);
  });
  $("qcspace").addEventListener("click", (ev) => {
    const b = ev.target.closest("button"); if (!b || !state.qc) return;
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

// ---- home: the case list ---------------------------------------------

async function showCases() {
  showStep("cases");
  try { state.cases = await jget(`${API}/cases`); }   // enriched: counts, MRI, labels
  catch (e) { $("caseslist").textContent = `Could not load cases: ${e.message}`; return; }
  renderCases();
}

// Client-side filter (search + type) over the fetched cases — instant, no refetch.
function renderCases() {
  const all = state.cases || [];
  const q = (state.caseSearch || "").trim().toLowerCase();
  const f = state.caseFilter || "all";
  const shown = all.filter((c) =>
    (f === "all" || c.kind === f) &&
    (!q || (c.label || "").toLowerCase().includes(q) || c.id.toLowerCase().includes(q)));
  $("casescount").textContent = all.length ? `· ${all.length}` : "";
  $("casesempty").hidden = all.length > 0;
  $("casesnoresults").hidden = !(all.length > 0 && shown.length === 0);
  const list = $("caseslist");
  list.innerHTML = "";
  for (const c of shown) list.append(caseCard(c));
}

function ccNum(n, one, many) {
  const s = el("span", { class: "cc-n" });
  s.append(el("b", {}, String(n)), " " + (n === 1 ? one : many));
  return s;
}

function caseCard(c) {
  const card = el("button", { class: "casecard", type: "button" });
  card.onclick = () => openCase(c.id);
  const when = c.created_at ? new Date(c.created_at * 1000).toLocaleDateString(
    undefined, { month: "short", day: "numeric", year: "numeric" }) : "";
  const stats = el("div", { class: "cc-stats" });
  stats.append(
    el("span", { class: `cc-badge ${c.kind}` }, c.kind === "import" ? "imported" : "detected"),
    ccNum(c.n_shanks, "electrode", "electrodes"),
    ccNum(c.n_contacts, "contact", "contacts"));
  const sub = [c.has_mri ? "MRI" : "CT-only", c.labeled ? "labeled" : null, when]
    .filter(Boolean).join(" · ");
  const subEl = el("div", { class: "cc-sub muted" }, sub + " · ");
  subEl.append(el("span", { class: "cc-id" }, c.id.slice(0, 8)));   // distinguishes same-named cases
  card.append(el("div", { class: "cc-title" }, c.label || c.id.slice(0, 8)), stats, subEl);
  return card;
}

// Open an existing case: load its results, then restore its newest label job
// (labels/QC) so the case comes back exactly as it was left.
async function openCase(id) {
  await loadResults(id);
  try {
    const jobs = await jget(`${API}/jobs`);
    const labelJobs = jobs.filter((j) => j.kind === "label" && j.parent === id);
    const newest = labelJobs[0];   // /jobs is newest-first
    if (!newest) return;
    state.labeledOnce = labelJobs.some((j) => j.state === "succeeded");
    state.labelJobId = newest.id;
    if (newest.t1) { state.mri = { path: newest.t1, name: "(uploaded MRI)" }; $("labelbtn").disabled = false; }
    if (newest.atlas) $("atlassel").value = newest.atlas;
    if (newest.state === "succeeded") showProposed(newest.id, { reloadViewer: false, jumpToQc: false });
    else if (["failed", "cancelled"].includes(newest.state)) _showLabelError(newest);
    else pollLabel(newest.id, !state.labeledOnce);
  } catch (_e) { /* a case with no label job is fine */ }
}

// ---- import a localization computed elsewhere ------------------------

// browse → upload → fill the paired path field (reuses the /uploads endpoint).
async function importBrowse(fileInputId, pathId) {
  const f = $(fileInputId).files[0];
  if (!f) return;
  $("imp-msg").textContent = `Uploading ${f.name}…`;
  const fd = new FormData(); fd.append("file", f);
  const r = await fetch(`${API}/uploads`, { method: "POST", body: fd });
  if (!r.ok) { $("imp-msg").textContent = `Upload failed: ${await r.text()}`; return; }
  $(pathId).value = (await r.json()).path;
  $("imp-msg").textContent = "";
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

async function runImport() {
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
      body: JSON.stringify({ ct, contacts, trajectories: traj, label, ...(t1 ? { t1 } : {}) }),
    });
    body = await r.json().catch(() => ({}));
  } catch (e) { $("imp-msg").textContent = `Import failed: ${e.message}`; return; }
  $("imp-msg").textContent = "";
  if (!r.ok) {
    const d = body.detail;
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
  $("cancelbtn").onclick = cancel;
  $("exportbtn").onclick = doExport;
  document.querySelectorAll("#wsflow button").forEach((b) => { b.onclick = () => showWs(b.dataset.ws); });
  $("slot-mri").onclick = () => {
    if (!state.mri) { showWs("review"); const lc = $("labelcard"); lc.open = true; lc.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
  };
  $("slot-reg").onclick = () => { showWs("review"); setViewerTab("qc"); };
  $("restartbtn").onclick = showCases;          // workspace → back to the case list
  $("newcasebtn").onclick = restart;            // case list → fresh new-case (drop) form
  $("importbtn").onclick = () => showStep("import");
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
  // Selecting a different atlas re-labels immediately (registration is cached,
  // so only the atlas warp + sampling re-runs) — so the labels track the atlas.
  $("atlassel").addEventListener("change", () => { if (state.mri && state.jobId) runLabel(); });
  wireQc();
  await loadAtlases();   // populate the picker before a resume sets its value
  try {
    const h = await jget("/healthz");
    $("engine").textContent = `engine ${h.engine_version} · ${h.engine_import_ok ? "ready" : "NOT LINKED"}`;
  } catch { $("engine").textContent = "service unreachable"; }
  // Land on the case list (the front door). Opening a case restores its viewer,
  // reviewed contacts, and newest label job — see openCase().
  await showCases();
}
boot();
