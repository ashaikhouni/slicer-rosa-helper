// SEEG GT labeler — single-page client.
// State machine: select subject → list trajectories → load trajectory →
// click slab to set landmarks → save (durable per trajectory).

"use strict";

const state = {
  subjects: [],
  library: { models: [] },
  currentSubject: null,
  trajectories: [],
  currentTraj: null,
  trajData: null,         // /api/trajectory response
  landmarks: {},          // {tip: arc_mm, bone_inner: arc_mm, ...}
  modelId: "",
  notes: "",
  slabImgs: {},           // {axis_perp1: Image, axis_perp2: Image}
};

// Both slabs share the same arc axis; perp differs. Click on either sets
// the same arc landmark; both repaint with the new overlay.
const SLAB_VIEWS = ["axis_perp1", "axis_perp2"];

const COLORS = {
  tip:         "#00bcd4",
  bone_inner:  "#ff9800",
  bolt_start:  "#e91e63",
  bolt_end:    "#9c27b0",
  snap:        "#f44336",
  contact:     "#4caf50",
  algo_walked: "#ffd700",
  algo_bone:   "#ffa726",
};

// --------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------

async function init() {
  await loadSubjects();
  await loadLibrary();
  populateVendorFilter();
  document.getElementById("vendor-filter").addEventListener("change", populateModelDropdown);
  document.getElementById("model-select").addEventListener("change", onModelSelected);
  document.querySelectorAll('input[name="active-landmark"]').forEach(r =>
    r.addEventListener("change", () => render())
  );
  document.getElementById("clear-active-btn").addEventListener("click", clearActiveLandmark);
  document.getElementById("save-btn").addEventListener("click", save);
  document.getElementById("skip-btn").addEventListener("click", () => goRelative(+1, false));
  document.getElementById("next-btn").addEventListener("click", () => goRelative(+1, false));
  document.getElementById("prev-btn").addEventListener("click", () => goRelative(-1, false));
  document.getElementById("delete-btn").addEventListener("click", deleteGT);
  document.getElementById("notes").addEventListener("input", (e) => { state.notes = e.target.value; });
  document.querySelectorAll(".slab-canvas").forEach(canvas => {
    canvas.addEventListener("click", onSlabClick);
    canvas.addEventListener("contextmenu", onSlabRightClick);
  });
  document.addEventListener("keydown", onKey);
}

document.addEventListener("DOMContentLoaded", init);

// --------------------------------------------------------------------
// Sidebar — subjects + trajectories
// --------------------------------------------------------------------

async function loadSubjects() {
  const r = await fetch("/api/subjects").then(x => x.json());
  state.subjects = r;
  const ul = document.getElementById("subjects");
  ul.innerHTML = "";
  r.forEach(s => {
    const li = document.createElement("li");
    li.dataset.sid = s.subject;
    if (s.n_labeled === s.n_trajectories && s.n_trajectories > 0) li.classList.add("labeled");
    li.innerHTML = `<span>${s.subject}</span><span class="progress">${s.n_labeled}/${s.n_trajectories}</span>`;
    li.addEventListener("click", () => selectSubject(s.subject));
    ul.appendChild(li);
  });
  document.getElementById("dataset-info").textContent =
    `${r.length} subjects, ${r.reduce((a, s) => a + s.n_trajectories, 0)} trajectories`;
}

async function selectSubject(sid) {
  state.currentSubject = sid;
  document.querySelectorAll("#subjects li").forEach(li =>
    li.classList.toggle("active", li.dataset.sid === sid));
  const r = await fetch(`/api/subject/${sid}`).then(x => x.json());
  state.trajectories = r.trajectories;
  renderTrajList();
  // Auto-load first unlabeled trajectory.
  const first = r.trajectories.find(t => !t.labeled) || r.trajectories[0];
  if (first) selectTrajectory(first.trajectory);
}

function renderTrajList() {
  const ul = document.getElementById("trajectories");
  ul.innerHTML = "";
  state.trajectories.forEach(t => {
    const li = document.createElement("li");
    li.dataset.tid = t.trajectory;
    if (t.labeled) li.classList.add("labeled");
    if (t.trajectory === state.currentTraj) li.classList.add("active");
    li.innerHTML = `<span>${t.trajectory}</span><span class="progress">${t.electrode_model || (t.labeled ? "✓" : "")}</span>`;
    li.addEventListener("click", () => selectTrajectory(t.trajectory));
    ul.appendChild(li);
  });
}

async function selectTrajectory(tid) {
  if (!state.currentSubject) return;
  state.currentTraj = tid;
  document.querySelectorAll("#trajectories li").forEach(li =>
    li.classList.toggle("active", li.dataset.tid === tid));

  setStatus("loading...");
  state.landmarks = {};
  state.modelId = "";
  state.notes = "";
  const r = await fetch(`/api/trajectory/${state.currentSubject}/${tid}`);
  if (!r.ok) { setStatus("failed to load trajectory", "err"); return; }
  state.trajData = await r.json();
  // Seed landmarks from prior GT if present (handled by the trajectory list — the
  // model_id and arcs come from the GT TSV via the subjects endpoint... we re-fetch).
  await maybeLoadExistingGT(state.currentSubject, tid);
  document.getElementById("current-traj").textContent =
    `${state.currentSubject} / ${tid}`;
  document.getElementById("notes").value = state.notes;
  await loadSlab();
  setStatus("");
}

async function maybeLoadExistingGT(sid, tid) {
  const tjs = state.trajectories.find(x => x.trajectory === tid);
  if (tjs && tjs.electrode_model) {
    state.modelId = tjs.electrode_model;
    document.getElementById("model-select").value = tjs.electrode_model;
    if (tjs.tip_arc_mm != null)        state.landmarks.tip = tjs.tip_arc_mm;
    if (tjs.bone_inner_arc_mm != null) state.landmarks.bone_inner = tjs.bone_inner_arc_mm;
    if (tjs.bolt_start_arc_mm != null) state.landmarks.bolt_start = tjs.bolt_start_arc_mm;
    if (tjs.bolt_end_arc_mm != null)   state.landmarks.bolt_end = tjs.bolt_end_arc_mm;
    state.notes = tjs.notes || "";
  } else {
    state.modelId = "";
  }
}

// --------------------------------------------------------------------
// Library / model picker
// --------------------------------------------------------------------

async function loadLibrary() {
  state.library = await fetch("/api/library").then(x => x.json());
}

function populateVendorFilter() {
  const vendors = Array.from(new Set(state.library.models.map(m => m.vendor || "Other"))).sort();
  const sel = document.getElementById("vendor-filter");
  vendors.forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  });
  populateModelDropdown();
}

function populateModelDropdown() {
  const v = document.getElementById("vendor-filter").value;
  const sel = document.getElementById("model-select");
  sel.innerHTML = '<option value="">— pick a model —</option>';
  state.library.models
    .filter(m => !v || m.vendor === v)
    .forEach(m => {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = `${m.id}  (${m.contact_count}c, ${m.total_length_mm.toFixed(1)} mm)`;
      sel.appendChild(o);
    });
  if (state.modelId) sel.value = state.modelId;
}

function onModelSelected(e) {
  state.modelId = e.target.value;
  const m = state.library.models.find(x => x.id === state.modelId);
  if (m) {
    document.getElementById("model-info").textContent =
      `${m.contact_count} contacts, pitches mm = [${(m.offsets_from_tip_mm.slice(0, 12).map(o => o.toFixed(1)).join(", "))}…]`;
  } else {
    document.getElementById("model-info").textContent = "";
  }
  render();
}

// --------------------------------------------------------------------
// Slab loading + canvas rendering
// --------------------------------------------------------------------

async function loadSlab() {
  const sid = state.currentSubject, tid = state.currentTraj;
  if (!sid || !tid) return;
  state.slabImgs = {};
  // Load both perp views in parallel.
  const loaders = SLAB_VIEWS.map(which => new Promise((resolve) => {
    const img = new Image();
    img.onload = () => { state.slabImgs[which] = img; resolve(); };
    img.onerror = () => { resolve(); };
    img.src = `/api/slab.png?sid=${sid}&tid=${tid}&which=${which}&t=${Date.now()}`;
  }));
  await Promise.all(loaders);
  if (Object.keys(state.slabImgs).length === 0) {
    setStatus("slab load failed", "err");
  }
  render();
}

function render() {
  document.querySelectorAll(".slab-canvas").forEach(canvas => renderOne(canvas));
  updateLegend();
  updateLandmarkValues();
}

function renderOne(canvas) {
  const which = canvas.dataset.which;
  const img = state.slabImgs[which];
  const ctx = canvas.getContext("2d");
  if (!img || !state.trajData) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  drawAlgoLandmarks(ctx, img);
  drawAutoContacts(ctx, img);
  drawUserLandmarks(ctx, img);
}

function arcToX(arc, img) {
  const ext = state.trajData.slab_extent;
  const inset = AXES_INSET;
  const W = (img || _anyImg()).naturalWidth;
  const ax_w = W - inset.left - inset.right;
  return inset.left + ((arc - ext.arc_lo) / (ext.arc_hi - ext.arc_lo)) * ax_w;
}

function xToArc(x, img) {
  const ext = state.trajData.slab_extent;
  const inset = AXES_INSET;
  const W = (img || _anyImg()).naturalWidth;
  const ax_w = W - inset.left - inset.right;
  return ext.arc_lo + ((x - inset.left) / ax_w) * (ext.arc_hi - ext.arc_lo);
}

function perpToY(perp, img) {
  const ext = state.trajData.slab_extent;
  const inset = AXES_INSET;
  const H = (img || _anyImg()).naturalHeight;
  const ax_h = H - inset.top - inset.bottom;
  return inset.top + ((ext.perp_hi - perp) / (ext.perp_hi - ext.perp_lo)) * ax_h;
}

function _anyImg() {
  return state.slabImgs.axis_perp1 || state.slabImgs.axis_perp2;
}

// Calibration constants — approximate matplotlib axes inset relative to
// the figure-with-bbox_inches='tight' the backend renders. Fine-tune if
// landmarks drift relative to visible features.
const AXES_INSET = { left: 88, right: 32, top: 30, bottom: 60 };

function drawAlgoLandmarks(ctx, img) {
  const d = state.trajData;
  if (!d || d.snap_failed) return;
  drawMarker(ctx, arcToX(0, img), perpToY(0, img), COLORS.algo_walked, "square");
  drawMarker(ctx, arcToX(d.tip_arc, img), perpToY(0, img), "#00bfff", "square");
  if (d.bone_arc_mm != null) {
    drawVerticalLine(ctx, arcToX(d.bone_arc_mm, img), COLORS.algo_bone, [4, 4], img);
  }
  d.snap_arcs.forEach(a => drawMarker(ctx, arcToX(a, img), perpToY(0, img), COLORS.snap, "ring"));
}

function drawAutoContacts(ctx, img) {
  const m = state.library.models.find(x => x.id === state.modelId);
  if (!m) return;
  const d = state.trajData;
  if (!d || !d.snap_peaks_ras || d.snap_peaks_ras.length < 2) return;
  const offsets = m.offsets_from_tip_mm;
  if (!offsets || offsets.length === 0) return;

  // Convention: offsets are measured from the PHYSICAL ELECTRODE TIP
  // (the deep edge of bright metal visible in CT). offsets[0] ≈ 1 mm
  // for DIXI = contact 1 sits 1 mm bolt-ward of the physical tip.
  // The user's tip landmark = physical tip (easier to click than the
  // contact center). Walk along the snap polyline in 3D from
  // (physical tip - offsets[0]) which is the deepest contact (≈ deepest
  // snap peak), then project to slab x.
  //
  // Implementation: anchor at the deepest snap peak; subtract offsets[0]
  // from the user's tip-to-deepest shift so contact 1 lands on the
  // deepest peak (not 1 mm beyond it).
  const peaks = d.snap_peaks_ras;
  const deepest = peaks[peaks.length - 1];
  const cum = [0];
  const cum_pts = [deepest];
  for (let i = peaks.length - 2; i >= 0; i--) {
    const dx = peaks[i][0] - peaks[i + 1][0];
    const dy = peaks[i][1] - peaks[i + 1][1];
    const dz = peaks[i][2] - peaks[i + 1][2];
    cum.push(cum[cum.length - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz));
    cum_pts.push(peaks[i]);
  }
  const head = cum_pts[cum_pts.length - 1];
  const head2 = cum_pts[Math.max(0, cum_pts.length - 2)];
  const headDir = [head[0] - head2[0], head[1] - head2[1], head[2] - head2[2]];
  const headLen = Math.hypot(headDir[0], headDir[1], headDir[2]) || 1.0;
  headDir[0] /= headLen; headDir[1] /= headLen; headDir[2] /= headLen;

  const deepest_arc = (deepest[0] - d.entry_ras[0]) * d.axis[0]
                    + (deepest[1] - d.entry_ras[1]) * d.axis[1]
                    + (deepest[2] - d.entry_ras[2]) * d.axis[2];
  // If the user clicks the PHYSICAL TIP (= deep edge of metal), it sits
  // offsets[0] bolt-ward of contact 1 in model coords. We subtract that
  // so contact 1 (placed at offset_from_deepest = 0) lands on the
  // deepest snap peak when user's click ≈ deepest_snap_peak + offsets[0].
  const tip_arc = state.landmarks.tip ?? (deepest_arc + offsets[0]);
  const arc_shift = tip_arc - offsets[0] - deepest_arc;

  // Walk by the offsets from the DEEPEST contact (= offsets[i] - offsets[0])
  // along the polyline. Same arithmetic as before; only arc_shift changes.
  const offsets_from_deepest = offsets.map(o => o - offsets[0]);
  for (const lib_off of offsets_from_deepest) {
    let k = 0;
    while (k < cum.length - 1 && cum[k + 1] < lib_off) k++;
    let pt;
    if (k >= cum.length - 1) {
      const past = lib_off - cum[cum.length - 1];
      const p = cum_pts[cum_pts.length - 1];
      pt = [p[0] + past * headDir[0], p[1] + past * headDir[1], p[2] + past * headDir[2]];
    } else {
      const seg_lo = cum[k], seg_hi = cum[k + 1];
      const t = seg_hi > seg_lo ? (lib_off - seg_lo) / (seg_hi - seg_lo) : 0;
      const a = cum_pts[k], b = cum_pts[k + 1];
      pt = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
    }
    const arc = (pt[0] - d.entry_ras[0]) * d.axis[0]
              + (pt[1] - d.entry_ras[1]) * d.axis[1]
              + (pt[2] - d.entry_ras[2]) * d.axis[2]
              + arc_shift;
    drawMarker(ctx, arcToX(arc, img), perpToY(0, img), COLORS.contact, "dot-sm");
  }
}

function drawUserLandmarks(ctx, img) {
  for (const [name, arc] of Object.entries(state.landmarks)) {
    if (arc == null) continue;
    const color = {
      tip: COLORS.tip,
      bone_inner: COLORS.bone_inner,
      bolt_start: COLORS.bolt_start,
      bolt_end: COLORS.bolt_end,
    }[name];
    if (!color) continue;
    drawVerticalLine(ctx, arcToX(arc, img), color, [], img);
    drawMarker(ctx, arcToX(arc, img), perpToY(0, img), color, "diamond");
  }
}

function drawMarker(ctx, x, y, color, shape) {
  ctx.lineWidth = 2;
  if (shape === "square") {
    ctx.strokeStyle = color;
    ctx.strokeRect(x - 6, y - 6, 12, 12);
  } else if (shape === "ring") {
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.stroke();
  } else if (shape === "dot-sm") {
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
  } else if (shape === "diamond") {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, y - 6); ctx.lineTo(x + 5, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 5, y);
    ctx.closePath(); ctx.fill();
  }
}

function drawVerticalLine(ctx, x, color, dash, img) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(dash || []);
  const H = (img || _anyImg()).naturalHeight;
  ctx.beginPath();
  ctx.moveTo(x, AXES_INSET.top);
  ctx.lineTo(x, H - AXES_INSET.bottom);
  ctx.stroke();
  ctx.restore();
}

function updateLegend() {
  const items = [
    { color: COLORS.algo_walked, label: "algorithm walked entry" },
    { color: "#00bfff",            label: "algorithm tip" },
    { color: COLORS.algo_bone,    label: "algorithm bone arc" },
    { color: COLORS.snap,         label: "snap peaks" },
    { color: COLORS.contact,      label: "auto-derived contacts (from selected model)" },
    { color: COLORS.tip,          label: "your tip" },
    { color: COLORS.bone_inner,   label: "your bone inner" },
    { color: COLORS.bolt_start,   label: "your bolt start" },
    { color: COLORS.bolt_end,     label: "your bolt end" },
  ];
  document.getElementById("legend").innerHTML = items
    .map(i => `<span class="item"><span class="swatch" style="background:${i.color}"></span>${i.label}</span>`)
    .join("");
}

function updateLandmarkValues() {
  document.querySelectorAll(".value[data-landmark]").forEach(el => {
    const lm = el.dataset.landmark;
    const v = state.landmarks[lm];
    el.textContent = v == null ? "—" : `${v.toFixed(2)} mm`;
  });
}

// --------------------------------------------------------------------
// Click handling
// --------------------------------------------------------------------

function onSlabClick(e) {
  if (!state.trajData) return;
  const canvas = e.currentTarget;
  const rect = canvas.getBoundingClientRect();
  // Account for canvas being CSS-scaled.
  const scaleX = canvas.width / rect.width;
  const x = (e.clientX - rect.left) * scaleX;
  const arc = xToArc(x);
  const active = document.querySelector('input[name="active-landmark"]:checked').value;
  state.landmarks[active] = arc;
  // Auto-advance to next landmark for fast labeling.
  const order = ["tip", "bone_inner", "bolt_start", "bolt_end"];
  const idx = order.indexOf(active);
  if (idx >= 0 && idx < order.length - 1) {
    document.querySelector(`input[name="active-landmark"][value="${order[idx + 1]}"]`).checked = true;
  }
  render();
}

function onSlabRightClick(e) {
  e.preventDefault();
  if (confirm("Clear all landmarks for this trajectory?")) {
    state.landmarks = {};
    render();
  }
}

function clearActiveLandmark() {
  const active = document.querySelector('input[name="active-landmark"]:checked').value;
  delete state.landmarks[active];
  render();
}

function onKey(e) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
  if (e.key === "s" || e.key === "Enter") { save(); e.preventDefault(); }
  if (e.key === "ArrowRight" || e.key === "n") { goRelative(+1, false); }
  if (e.key === "ArrowLeft" || e.key === "p")  { goRelative(-1, false); }
  if (e.key === "1") setActive("tip");
  if (e.key === "2") setActive("bone_inner");
  if (e.key === "3") setActive("bolt_start");
  if (e.key === "4") setActive("bolt_end");
}

function setActive(lm) {
  document.querySelector(`input[name="active-landmark"][value="${lm}"]`).checked = true;
  render();
}

// --------------------------------------------------------------------
// Save / delete / navigate
// --------------------------------------------------------------------

async function save() {
  if (!state.currentSubject || !state.currentTraj) return;
  if (!state.modelId) { setStatus("pick an electrode model first", "err"); return; }
  setStatus("saving...");
  const payload = {
    electrode_model: state.modelId,
    tip_arc_mm: state.landmarks.tip,
    bone_inner_arc_mm: state.landmarks.bone_inner,
    bolt_start_arc_mm: state.landmarks.bolt_start,
    bolt_end_arc_mm: state.landmarks.bolt_end,
    notes: state.notes,
  };
  const r = await fetch(`/api/gt/${state.currentSubject}/${state.currentTraj}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) { setStatus("save failed", "err"); return; }
  setStatus("saved ✓", "ok");
  await loadSubjects();
  // Re-fetch trajectories list to refresh checkmarks.
  const sr = await fetch(`/api/subject/${state.currentSubject}`).then(x => x.json());
  state.trajectories = sr.trajectories;
  renderTrajList();
  goRelative(+1, true);
}

async function deleteGT() {
  if (!state.currentSubject || !state.currentTraj) return;
  if (!confirm(`Delete GT for ${state.currentSubject}/${state.currentTraj}?`)) return;
  await fetch(`/api/gt/${state.currentSubject}/${state.currentTraj}`, { method: "DELETE" });
  setStatus("deleted", "ok");
  state.landmarks = {}; state.modelId = "";
  document.getElementById("model-select").value = "";
  await loadSubjects();
  const sr = await fetch(`/api/subject/${state.currentSubject}`).then(x => x.json());
  state.trajectories = sr.trajectories;
  renderTrajList();
}

function goRelative(delta, only_to_unlabeled) {
  const idx = state.trajectories.findIndex(t => t.trajectory === state.currentTraj);
  let next = idx + delta;
  while (next >= 0 && next < state.trajectories.length) {
    if (!only_to_unlabeled || !state.trajectories[next].labeled) {
      selectTrajectory(state.trajectories[next].trajectory);
      return;
    }
    next += delta;
  }
}

// --------------------------------------------------------------------
// Status bar
// --------------------------------------------------------------------

function setStatus(msg, cls) {
  const el = document.getElementById("status-msg");
  el.textContent = msg || "";
  el.className = cls || "";
  if (cls === "ok") setTimeout(() => { if (el.className === "ok") el.textContent = ""; }, 1500);
}
