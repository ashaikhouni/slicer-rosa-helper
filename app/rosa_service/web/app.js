"use strict";
// ROSA web UI — a single-page wizard over the local service contract.
// Drop CT → run pipeline job → review/edit contacts + 3D viewer → export.
// Talks ONLY to /api/v1 + /healthz (never the engine directly).

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const state = { ct: null, jobId: null, es: null, poll: null };

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
  };
  frame.src = `${API}/jobs/${id}/viewer/`;
  showStep("results");
  try { renderReview(await jget(`${API}/jobs/${id}/review`)); }
  catch (e) { $("reviewlist").textContent = `Could not load review: ${e.message}`; }
}

function renderReview(doc) {
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
      const acc = el("input", { type: "checkbox" });
      acc.checked = c.accepted; acc.disabled = !shank.accepted;
      acc.onchange = () => patch([{ op: acc.checked ? "accept_contact" : "reject_contact", shank: shank.name, index: c.index }]);
      const region = el("input", { type: "text", class: "region", value: c.region || "", placeholder: "—" });
      region.disabled = !shank.accepted;
      region.onchange = () => {
        if (region.value.trim())
          patch([{ op: "relabel_contact", shank: shank.name, index: c.index, region: region.value.trim() }]);
      };
      row.append(acc, el("span", { class: "cname" }, c.name), region);
      cs.append(row);
    }
    box.append(cs);
    list.append(box);
  }
}

async function patch(ops) {
  try { renderReview(await jsend(`${API}/jobs/${state.jobId}/review`, "PATCH", { ops })); }
  catch (e) { $("exportmsg").textContent = `Edit failed: ${e.message}`; }
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
  state.ct = null; state.jobId = null;
  $("ctpath").value = ""; $("fileinput").value = ""; $("exportmsg").textContent = "";
  $("viewerframe").src = "about:blank";
  setCt(null);
  showStep("drop");
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
