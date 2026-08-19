const $ = (id) => document.getElementById(id);

const state = {
  bootstrap: null,
  apiToken: "",
  stage: null,
  sceneKind: "",
  currentPdb: "",
  currentAtoms: [],
  currentBonds: [],
  currentExtra: {},
  viewerInfoBase: "",
  metAtoms: [],
  metBonds: [],
  metMetals: [],
  metDonors: [],
  metGroupConstraints: null,
  metCoordination: null,
  metOriginal: null,
  metInsertedMetalIndices: new Set(),
  selectedMetalAtom: null,
  selectedDonors: new Set(),
  selectedAtomIndices: new Set(),
  selectedRespGroupId: null,
  metMetalElements: new Map(),
  overlayComponent: null,
  proteinMetals: [],
  proteinSourceMetals: [],
  proteinMetalActions: new Map(),
  proteinInsertions: [],
  proteinInsertionDonors: [],
  proteinResidues: [],
  proteinHighlightSets: {},
  proteinBindingLinks: [],
  selectedResidueKeys: new Set(),
  lastResidueIndex: -1,
  propkaResidueKeys: new Set(),
  propkaChanges: [],
  disulfideCandidates: [],
  disulfideKeys: new Set(),
  disulfideTokens: [],
  desComponents: [],
  desMetalSites: [],
  desSaltInitialized: false,
  libraryComponents: [],
  libraryCandidates: [],
  expandedLibraryComponents: new Set(),
  selectedLibraryComponent: "",
  selectedLibraryCandidate: -1,
  selectedLibraryFile: "",
  selectedLibraryFileEditable: false,
  structureComponent: null,
  selectedRespJobDir: "",
  proteinSiteRespApproved: false,
  proteinSiteRespClusters: [],
  proteinLoaded: false,
  proteinLoadSerial: 0,
  lastProteinMissingLoopAction: "",
  pickModifiers: {ctrl: false, shift: false, meta: false},
  homeOrientation: null,
  homeCameraDistance: null,
  axisBaseOrientation: null,
  axisSyncActive: false,
  mdStageDefs: [],
  lastWorkflow: "",
  busyCount: 0,
  busyObserver: null,
  busyRefreshQueued: false,
  sorts: {
    metals: {column: "number", dir: "asc"},
    donors: {column: "number", dir: "asc"},
  },
};

const GROUP_COLORS = [
  "#2563eb",
  "#dc2626",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#db2777",
  "#0891b2",
  "#65a30d",
  "#ea580c",
  "#4338ca",
];
const MAX_QUICK_MIN_DONORS = 12;
const MANUAL_CN_VALUE = "manual";
const SUPPORTED_METALS = new Set([
  "Co", "Cu", "Ni", "Mn", "Fe",
  "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
]);
const DONOR_ELEMENTS = new Set(["N", "O", "S", "P"]);
const AVOGADRO = 6.02214076e23;
const ANGSTROM3_TO_LITER = 1e-27;
const MAX_PREVIEW_IONS = 2400;
const ION_COLORS = {
  "Na+": [0.45, 0.15, 0.95],
  "K+": [0.55, 0.2, 0.85],
  "Ca2+": [0.25, 1.0, 0.0],
  "Cl-": [0.1, 0.85, 0.1],
  "Br-": [0.55, 0.2, 0.1],
};
const MD_FIELD_HELP = {
  maxcyc: "Total number of minimization cycles.",
  ncyc: "Switch from steepest descent to conjugate gradient after this many cycles.",
  nstlim: "Number of MD integration steps.",
  dt: "Time step in ps.",
  temp0: "Target temperature in K.",
  tempi: "Initial temperature for assigned velocities in K.",
  pres0: "Target pressure in bar.",
  ntt: "Amber thermostat selector.",
  gamma_ln: "Langevin collision frequency in ps^-1.",
  ntpr: "Print energies and progress every ntpr steps.",
  ntwx: "Write trajectory frames every ntwx steps.",
  ntwr: "Write restart information every ntwr steps.",
  restraint_wt: "Positional restraint weight in kcal/mol/A^2.",
  barostat: "Amber barostat selector for pressure relaxation.",
  taup: "Pressure relaxation time in ps.",
  iwrap: "Wrap molecules back into the primary box.",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value ?? null));
}

function setStatus(message, cls = "") {
  const node = $("status-line");
  node.textContent = message;
  node.className = cls;
  const panel = $("warning-panel");
  panel.textContent = message;
  panel.className = `warning-panel ${cls}`;
}

function applyBusyState() {
  const busy = state.busyCount > 0;
  document.body.classList.toggle("is-busy", busy);
  for (const node of document.querySelectorAll("button, select, input, textarea")) {
    if (node.id === "quit_app") continue;
    if (busy) {
      if (!node.disabled) {
        node.disabled = true;
        node.dataset.busyDisabled = "1";
      }
    } else if (node.dataset.busyDisabled === "1") {
      node.disabled = false;
      delete node.dataset.busyDisabled;
    }
  }
}

function installBusyObserver() {
  if (state.busyObserver || !window.MutationObserver) return;
  state.busyObserver = new MutationObserver(() => {
    if (state.busyCount <= 0 || state.busyRefreshQueued) return;
    state.busyRefreshQueued = true;
    requestAnimationFrame(() => {
      state.busyRefreshQueued = false;
      if (state.busyCount > 0) applyBusyState();
    });
  });
  state.busyObserver.observe(document.body, {childList: true, subtree: true});
}

function setUiBusy(active) {
  state.busyCount = Math.max(0, state.busyCount + (active ? 1 : -1));
  applyBusyState();
}

async function runBusy(message, fn) {
  if (message) setStatus(message, "warn");
  setUiBusy(true);
  try {
    return await fn();
  } finally {
    setUiBusy(false);
  }
}

function option(select, label, value, data) {
  const opt = document.createElement("option");
  opt.textContent = label;
  opt.value = value;
  if (data !== undefined) opt.dataset.item = JSON.stringify(data);
  select.appendChild(opt);
  return opt;
}

function replaceOptions(select, items, selected) {
  select.replaceChildren();
  for (const item of items || []) {
    if (Array.isArray(item)) option(select, item[1], item[0]);
    else if (typeof item === "object") option(select, item.label || item.key || item.value, item.key || item.value || item.label, item);
    else option(select, item, item);
  }
  if (selected !== undefined) select.value = selected;
  if (!select.value && select.options.length) select.selectedIndex = 0;
}

function isRespInputMode() {
  return $("workflow_type")?.value === "metallophore" && $("met_mode")?.value === "resp_input";
}

function groupColor(groupId) {
  if (!groupId) return "#64748b";
  return GROUP_COLORS[(Number(groupId) - 1) % GROUP_COLORS.length];
}

function groupColorRgb(groupId) {
  return hexToRgb(groupColor(groupId));
}

function atomDisplayName(atom) {
  const name = String(atom?.name || "").trim();
  const index = Number(atom?.index);
  if (name && /\d/.test(name)) return name;
  return `${atom?.element || name || "X"}${Number.isFinite(index) ? index : ""}`;
}

function groupIdForAtom(atomIndex) {
  const atom = (state.metGroupConstraints?.atoms || []).find((item) => Number(item.index) === Number(atomIndex));
  return atom?.group_id == null ? null : Number(atom.group_id);
}

function groupForId(groupId) {
  return (state.metGroupConstraints?.groups || []).find((item) => Number(item.group_id) === Number(groupId)) || null;
}

function groupLabelForAtom(atomIndex) {
  const groupId = groupIdForAtom(atomIndex);
  if (!groupId) return "none";
  const group = groupForId(groupId);
  return `Grp ${groupId}${group?.label ? `: ${group.label}` : ""}`;
}

function coordinationHint(element) {
  const record = (state.bootstrap?.supported_metals || []).find((item) => item.element === element);
  return record?.coordination || "";
}

function metalBootstrapRecord(element) {
  return (state.bootstrap?.supported_metals || []).find((item) => item.element === element) || null;
}

function defaultMetalCharge(element) {
  const record = metalBootstrapRecord(element);
  if (record?.default_charge !== null && record?.default_charge !== undefined) return Number(record.default_charge);
  return Number(record?.charges?.[0] || 2);
}

function coordinationRecord(element, charge) {
  const record = metalBootstrapRecord(element);
  return record?.coordination_by_charge?.[String(Number(charge))] || null;
}

function coordinationOptions(element, charge) {
  const record = coordinationRecord(element, charge);
  return record?.allowed_cn?.length ? record.allowed_cn.map(Number) : [2, 3, 4, 5, 6, 7, 8, 9];
}

function defaultCoordinationNumber(element, charge) {
  const record = coordinationRecord(element, charge);
  return Number(record?.default_cn || coordinationOptions(element, charge)[0] || 6);
}

function selectedMetalChargeRow() {
  if (state.selectedMetalAtom === null || state.selectedMetalAtom === undefined || state.selectedMetalAtom === "") return null;
  return document.querySelector(`.metal-charge-row[data-atom-index="${Number(state.selectedMetalAtom)}"]`);
}

function updateCoordinationSelect(row, preferredValue = null) {
  if (!row) return;
  const cnSelect = row.querySelector(".metal-cn-select");
  if (!cnSelect) return;
  const element = row?.dataset?.element || "";
  const charge = Number(row?.querySelector(".metal-charge-select")?.value || defaultMetalCharge(element));
  const options = coordinationOptions(element, charge);
  const selected = preferredValue === MANUAL_CN_VALUE
    ? MANUAL_CN_VALUE
    : preferredValue !== null && options.includes(Number(preferredValue))
      ? String(Number(preferredValue))
      : String(defaultCoordinationNumber(element, charge));
  replaceOptions(
    cnSelect,
    [...options.map((cn) => [`${cn}`, `${cn}`]), [MANUAL_CN_VALUE, "Manual selection"]],
    selected
  );
}

function coordinationStatusForAtom(atomIndex) {
  const numericAtomIndex = Number(atomIndex);
  const hasSelectedMetal = state.selectedMetalAtom !== null && state.selectedMetalAtom !== undefined && state.selectedMetalAtom !== "";
  const isSelected = hasSelectedMetal && Number.isFinite(numericAtomIndex) && numericAtomIndex > 0 && numericAtomIndex === Number(state.selectedMetalAtom);
  const selectedCount = isSelected ? state.selectedDonors.size : 0;
  const auto = state.metCoordination && Number(state.metCoordination.metal_atom_index) === Number(atomIndex)
    ? (state.metCoordination.auto_filled_donor_atom_indices || []).length
    : 0;
  if (!isSelected) return "Select metal";
  const autoText = auto ? `, ${auto} auto-filled` : "";
  return `${selectedCount} selected${autoText}`;
}

function effectiveDonorIndicesForSelectedMetal() {
  const selected = new Set([...state.selectedDonors].map(Number));
  const coordination = state.metCoordination;
  if (coordination && Number(coordination.metal_atom_index) === Number(state.selectedMetalAtom)) {
    for (const donorIndex of coordination.auto_filled_donor_atom_indices || []) selected.add(Number(donorIndex));
    for (const donorIndex of coordination.effective_donor_atom_indices || []) selected.add(Number(donorIndex));
  }
  return selected;
}

function refreshMetalCoordinationStatus() {
  for (const row of document.querySelectorAll(".metal-charge-row")) {
    const hasSelectedMetal = state.selectedMetalAtom !== null && state.selectedMetalAtom !== undefined && state.selectedMetalAtom !== "";
    row.classList.toggle(
      "selected-row",
      hasSelectedMetal && Number(row.dataset.atomIndex) > 0 && Number(row.dataset.atomIndex) === Number(state.selectedMetalAtom)
    );
    const status = row.querySelector(".metal-cn-status");
    if (status) status.textContent = coordinationStatusForAtom(row.dataset.atomIndex);
  }
  refreshRespChargeHint();
}

function collectSelectedMetalCoordination() {
  if (!state.selectedMetalAtom) return null;
  const metal = state.metMetals.find((item) => Number(item.index) === Number(state.selectedMetalAtom));
  const element = metal ? selectedMetalElement(metal) : "";
  const row = selectedMetalChargeRow();
  const charge = Number(row?.querySelector(".metal-charge-select")?.value || defaultMetalCharge(element));
  const cnValue = row?.querySelector(".metal-cn-select")?.value || String(defaultCoordinationNumber(element, charge));
  const manualSelection = cnValue === MANUAL_CN_VALUE;
  const targetCn = manualSelection ? null : Number(cnValue);
  return {
    metal_atom_index: Number(state.selectedMetalAtom),
    element,
    formal_charge: charge,
    coordination_mode: manualSelection ? "manual_selection" : "target_cn",
    target_cn: targetCn,
    required_donor_atom_indices: Array.from(state.selectedDonors).map(Number),
  };
}

function refreshRespChargeHint() {
  const node = $("resp_charge_hint");
  if (!node) return;
  const coordination = collectSelectedMetalCoordination();
  if (!coordination?.element) {
    node.textContent = "For RESP fitting, use a net charge consistent with the atoms included in the RESP input.";
    return;
  }
  const cnText = coordination.coordination_mode === "manual_selection"
    ? "manual donor selection"
    : `target CN ${coordination.target_cn}`;
  node.textContent = `Selected ${coordination.element}+${coordination.formal_charge}, ${cnText}. Include this formal charge only when the RESP input contains the metal; multiplicity controls spin state only.`;
}

async function api(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-SIMPLE-Token": state.apiToken},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

async function postJson(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-SIMPLE-Token": state.apiToken},
    body: JSON.stringify(payload),
  });
  let data = {};
  try {
    data = await response.json();
  } catch (_err) {
    data = {ok: false, error: `Request failed: ${response.status}`};
  }
  return {response, data};
}

async function uploadFile(inputId) {
  const file = $(inputId).files?.[0];
  if (!file) throw new Error("Choose a file first.");
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/upload", {
    method: "POST",
    headers: {"X-SIMPLE-Token": state.apiToken},
    body: form,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || `Upload failed: ${response.status}`);
  return {path: data.path, name: file.name};
}

function ensureStage() {
  if (!window.NGL) {
    $("viewer-status").textContent = "NGL: unavailable";
    return null;
  }
  if (!state.stage) {
    state.stage = new NGL.Stage("viewer", {backgroundColor: "white", quality: "high", sampleLevel: 2});
    try {
      state.stage.setParameters({quality: "high", sampleLevel: 2});
    } catch (_err) {
      // Older NGL builds ignore high-quality viewer parameters.
    }
    configureViewerMouseControls();
    window.addEventListener("resize", () => state.stage.handleResize());
    $("viewer")?.addEventListener("click", () => $("viewer").focus());
    $("viewer")?.addEventListener("contextmenu", (ev) => ev.preventDefault());
    $("viewer")?.addEventListener("pointerdown", (ev) => {
      state.pickModifiers = {ctrl: ev.ctrlKey, shift: ev.shiftKey, meta: ev.metaKey};
    });
    try {
      state.stage.signals.clicked.add((pickingProxy) => handleViewerPick(pickingProxy));
    } catch (_err) {
      // Some older NGL builds expose picking differently; table-based selection still works.
    }
    startAxisOverlaySync();
  }
  $("viewer-status").textContent = "NGL: ready";
  return state.stage;
}

function configureViewerMouseControls() {
  const stage = state.stage;
  const controls = stage?.mouseControls;
  if (!stage || !controls?.clear || !controls?.add) return;
  try {
    controls.clear();
    controls.add("scroll", (activeStage, delta) => {
      activeStage.trackballControls.zoom(delta);
      clampCameraDistance();
    });
    controls.add("drag-left", (activeStage, dx, dy) => {
      activeStage.trackballControls.rotate(dx, dy);
      clampCameraDistance();
    });
    controls.add("drag-middle", (activeStage, dx, dy) => {
      activeStage.trackballControls.pan(dx, dy);
      clampCameraDistance();
    });
    controls.add("drag-right", (activeStage, dx, dy) => {
      activeStage.trackballControls.zRotate(dx, dy);
      clampCameraDistance();
    });
  } catch (_err) {
    // If a bundled NGL build changes the mouse API, the default controls still remain usable.
  }
}

function requestViewerRender() {
  try {
    state.stage?.viewer?.requestRender?.();
  } catch (_err) {
    // NGL render requests are best-effort across bundled builds.
  }
}

function captureOrientation() {
  const controls = state.stage?.viewerControls;
  if (!controls?.getOrientation) return null;
  try {
    const orientation = controls.getOrientation();
    const elements = orientation?.elements || orientation;
    const values = Array.isArray(elements) ? elements.slice() : Array.from(elements || []);
    return validOrientation(values) ? values : null;
  } catch (_err) {
    return null;
  }
}

function setOrientation(elements) {
  const controls = state.stage?.viewerControls;
  if (!controls?.orient || !validOrientation(elements)) return false;
  try {
    controls.orient(elements.slice());
    requestViewerRender();
    return true;
  } catch (_err) {
    return false;
  }
}

function validOrientation(elements) {
  return Array.isArray(elements)
    && elements.length >= 16
    && elements.every((value) => Number.isFinite(Number(value)));
}

function captureHomeOrientation() {
  state.homeOrientation = captureOrientation();
  state.homeCameraDistance = cameraDistanceFromOrientation(state.homeOrientation);
  state.axisBaseOrientation = state.homeOrientation ? state.homeOrientation.slice() : null;
}

function cameraDistanceFromOrientation(orientation) {
  return validOrientation(orientation) ? orientationScale(orientation) : null;
}

function clampCameraDistance() {
  const controls = state.stage?.viewerControls;
  if (!controls?.getCameraDistance || !controls?.distance) return;
  const home = state.homeCameraDistance || cameraDistanceFromOrientation(state.homeOrientation);
  const current = Number(controls.getCameraDistance());
  if (!Number.isFinite(current) || current <= 0) {
    resetCamera();
    return;
  }
  if (!Number.isFinite(home) || home <= 0) return;
  const minDistance = Math.max(0.2, home * 0.04);
  const maxDistance = Math.max(minDistance * 2, home * 12);
  const next = Math.min(maxDistance, Math.max(minDistance, current));
  if (Math.abs(next - current) > 1e-6) controls.distance(next);
}

function stopCameraAnimations() {
  try {
    state.stage?.animationControls?.clear?.();
  } catch (_err) {
    // Animation controls are best-effort across bundled NGL builds.
  }
}

function normalizeVector(vec) {
  const norm = Math.hypot(...vec);
  return norm ? vec.map((value) => value / norm) : [0, 0, 0];
}

function quaternionFromMatrixRows(right, up, forward) {
  const m00 = right[0];
  const m01 = right[1];
  const m02 = right[2];
  const m10 = up[0];
  const m11 = up[1];
  const m12 = up[2];
  const m20 = forward[0];
  const m21 = forward[1];
  const m22 = forward[2];
  const trace = m00 + m11 + m22;
  let qx;
  let qy;
  let qz;
  let qw;
  if (trace > 0) {
    const scale = Math.sqrt(trace + 1.0) * 2.0;
    qw = 0.25 * scale;
    qx = (m21 - m12) / scale;
    qy = (m02 - m20) / scale;
    qz = (m10 - m01) / scale;
  } else if (m00 > m11 && m00 > m22) {
    const scale = Math.sqrt(1.0 + m00 - m11 - m22) * 2.0;
    qw = (m21 - m12) / scale;
    qx = 0.25 * scale;
    qy = (m01 + m10) / scale;
    qz = (m02 + m20) / scale;
  } else if (m11 > m22) {
    const scale = Math.sqrt(1.0 + m11 - m00 - m22) * 2.0;
    qw = (m02 - m20) / scale;
    qx = (m01 + m10) / scale;
    qy = 0.25 * scale;
    qz = (m12 + m21) / scale;
  } else {
    const scale = Math.sqrt(1.0 + m22 - m00 - m11) * 2.0;
    qw = (m10 - m01) / scale;
    qx = (m02 + m20) / scale;
    qy = (m12 + m21) / scale;
    qz = 0.25 * scale;
  }
  return normalizeVector([qx, qy, qz, qw]);
}

function axisViewQuaternion(axis) {
  const basisMap = {
    x: {
      right: [0, 1, 0],
      up: [1, 0, 0],
      forward: [0, 0, -1],
    },
    y: {
      right: [0, 0, 1],
      up: [0, 1, 0],
      forward: [-1, 0, 0],
    },
    z: {
      right: [1, 0, 0],
      up: [0, 0, 1],
      forward: [0, -1, 0],
    },
  };
  const basis = basisMap[String(axis || "").toLowerCase()];
  return basis ? quaternionFromMatrixRows(basis.right, basis.up, basis.forward) : null;
}

function orientationScale(elements) {
  if (!elements?.length) return 1;
  return Math.hypot(Number(elements[0]) || 0, Number(elements[1]) || 0, Number(elements[2]) || 0) || 1;
}

function orientationFromQuaternion(quat, base) {
  if (!quat || !base?.length) return null;
  const [x, y, z, w] = quat;
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;
  const scale = orientationScale(base);
  const next = base.slice();
  next[0] = (1 - (yy + zz)) * scale;
  next[1] = (xy + wz) * scale;
  next[2] = (xz - wy) * scale;
  next[3] = 0;
  next[4] = (xy - wz) * scale;
  next[5] = (1 - (xx + zz)) * scale;
  next[6] = (yz + wx) * scale;
  next[7] = 0;
  next[8] = (xz + wy) * scale;
  next[9] = (yz - wx) * scale;
  next[10] = (1 - (xx + yy)) * scale;
  next[11] = 0;
  next[15] = 1;
  return next;
}

function resetCamera() {
  if (!state.stage) return;
  stopCameraAnimations();
  try {
    if (setOrientation(state.homeOrientation)) return;
    state.stage.autoView(0);
    requestViewerRender();
  } catch (_err) {
    setStatus("Camera reset is not supported by this NGL build.", "warn");
  }
}

function cameraAxis(axis) {
  if (!state.stage) return;
  stopCameraAnimations();
  const quat = axisViewQuaternion(axis);
  const base = state.homeOrientation || state.axisBaseOrientation || captureOrientation();
  const orientation = orientationFromQuaternion(quat, base);
  if (!orientation || !setOrientation(orientation)) {
    setStatus("Axis camera shortcut is not supported by this NGL build.", "warn");
  }
}

function updateAxisOverlay() {
  const overlay = $("axis-overlay");
  const svg = overlay?.querySelector("svg");
  if (!svg) return;
  const orientation = captureOrientation();
  if (!orientation?.length) return;
  const scale = orientationScale(orientation);
  const defs = [
    {vec: [orientation[0] / scale, orientation[1] / scale, orientation[2] / scale], line: overlay.querySelector(".axis-line-x"), label: overlay.querySelector(".axis-label-x")},
    {vec: [orientation[4] / scale, orientation[5] / scale, orientation[6] / scale], line: overlay.querySelector(".axis-line-y"), label: overlay.querySelector(".axis-label-y")},
    {vec: [orientation[8] / scale, orientation[9] / scale, orientation[10] / scale], line: overlay.querySelector(".axis-line-z"), label: overlay.querySelector(".axis-label-z")},
  ];
  const origin = overlay.querySelector(".axis-origin");
  const originX = 36;
  const originY = 36;
  const radius = 17;
  const labelRadius = 23;
  defs.sort((a, b) => a.vec[2] - b.vec[2]);
  for (const item of defs) {
    const [rx, ry, rz] = item.vec;
    const x2 = originX + rx * radius;
    const y2 = originY - ry * radius;
    const lx = originX + rx * labelRadius;
    const ly = originY - ry * labelRadius;
    item.line?.setAttribute("x1", String(originX));
    item.line?.setAttribute("y1", String(originY));
    item.line?.setAttribute("x2", x2.toFixed(2));
    item.line?.setAttribute("y2", y2.toFixed(2));
    item.line?.setAttribute("opacity", rz < -0.1 ? "0.42" : "1");
    item.label?.setAttribute("x", lx.toFixed(2));
    item.label?.setAttribute("y", ly.toFixed(2));
    item.label?.setAttribute("opacity", rz < -0.1 ? "0.55" : "1");
    if (item.line) svg.appendChild(item.line);
    if (item.label) svg.appendChild(item.label);
  }
  if (origin) svg.appendChild(origin);
}

function startAxisOverlaySync() {
  if (state.axisSyncActive) return;
  state.axisSyncActive = true;
  const tick = () => {
    if (!state.axisSyncActive) return;
    updateAxisOverlay();
    window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
}

function clearPreview(message = "Load a system preview.") {
  if (state.stage) state.stage.removeAllComponents();
  state.currentPdb = "";
  state.currentAtoms = [];
  state.currentBonds = [];
  state.currentExtra = {};
  state.viewerInfoBase = "";
  state.sceneKind = "";
  state.overlayComponent = null;
  state.structureComponent = null;
  state.selectedAtomIndices = new Set();
  state.selectedRespGroupId = null;
  state.homeOrientation = null;
  state.homeCameraDistance = null;
  state.axisBaseOrientation = null;
  $("viewer-overlay").textContent = message;
  $("viewer_info").textContent = "";
  resetProteinTables();
}

function hexToRgb(hex) {
  const text = String(hex || "#64748b").replace("#", "");
  return [
    parseInt(text.slice(0, 2), 16) / 255,
    parseInt(text.slice(2, 4), 16) / 255,
    parseInt(text.slice(4, 6), 16) / 255,
  ];
}

function rgbToHex(rgb) {
  return `#${(rgb || [0.5, 0.5, 0.5]).map((value) => {
    const channel = Math.max(0, Math.min(255, Math.round(Number(value) * 255)));
    return channel.toString(16).padStart(2, "0");
  }).join("")}`;
}

function addBoxLines(stage, lines, color = [0.15, 0.35, 0.85]) {
  if (!lines || !lines.length || !window.NGL) return;
  const shape = new NGL.Shape("box");
  for (const line of lines) {
    shape.addCylinder(line.start, line.end, color, 0.08);
  }
  const comp = stage.addComponentFromObject(shape);
  comp.addRepresentation("buffer", {disablePicking: true});
}

function addAtomLabels(stage, atoms) {
  if (!atoms?.length || !window.NGL) return;
  const shape = new NGL.Shape("atom labels");
  for (const atom of atoms) {
    const groupId = groupIdForAtom(atom.index);
    const groupText = groupId ? ` (Grp ${groupId})` : "";
    shape.addText(
      [Number(atom.x) + 0.28, Number(atom.y) + 0.28, Number(atom.z) + 0.28],
      [0.05, 0.05, 0.05],
      0.86,
      `${atomDisplayName(atom)}${groupText}`
    );
  }
  const comp = stage.addComponentFromObject(shape);
  comp.addRepresentation("buffer", {disablePicking: true});
}

function addIonDots(stage, ions) {
  if (!ions?.length || !window.NGL) return;
  const shape = new NGL.Shape("salt ions");
  for (const ion of ions) {
    shape.addSphere(ion.position, ion.color, ion.radius || 0.32, ion.name || "ion");
  }
  const comp = stage.addComponentFromObject(shape);
  comp.addRepresentation("buffer", {disablePicking: true});
}

function atomColor(atom) {
  if (atom?.color) return hexToRgb(atom.color);
  const element = normalizedElement(atom?.element || atom?.name).toUpperCase();
  const fallback = {
    H: "#ffffff",
    C: "#8c8c8c",
    N: "#3050f8",
    O: "#ff0d0d",
    S: "#ffff30",
    P: "#ff8000",
    FE: "#e06633",
    CU: "#c88033",
    NI: "#50d050",
    MN: "#9c7ac7",
    CO: "#f090a0",
    Y: "#94ffff",
    LA: "#70d4ff",
    ND: "#c7ffc7",
    EU: "#61ffc7",
    LU: "#00ab24",
  }[element] || "#9ca3af";
  return hexToRgb(fallback);
}

function previewAtomRadius(atom, showSpacefill, showSticks) {
  if (showSpacefill) {
    const element = normalizedElement(atom?.element || atom?.name).toUpperCase();
    return ({
      H: 0.35,
      C: 0.76,
      N: 0.72,
      O: 0.66,
      S: 1.02,
      P: 1.06,
      FE: 1.12,
      CU: 1.10,
      NI: 1.08,
      MN: 1.12,
      CO: 1.08,
      Y: 1.32,
      LA: 1.46,
      ND: 1.42,
      EU: 1.40,
      LU: 1.34,
    }[element] || 0.78);
  }
  return showSticks ? 0.28 : 0.16;
}

function addFilteredMetallophoreRepresentation(stage, atoms, bonds, options = {}) {
  if (!atoms?.length || !window.NGL) return;
  const showLines = Boolean(options.showLines);
  const showSticks = Boolean(options.showSticks);
  const showSpacefill = Boolean(options.showSpacefill);
  if (!showLines && !showSticks && !showSpacefill) return;
  const atomLookup = new Map((atoms || []).map((atom) => [Number(atom.index), atom]));
  const filteredBonds = filterPreviewBonds(atoms, bonds || []);
  const shape = new NGL.Shape("filtered metallophore");
  const bondRadius = showSticks ? 0.11 : 0.035;
  if (showLines || showSticks) {
    for (const bond of filteredBonds) {
      const first = atomLookup.get(Number(bond.first));
      const second = atomLookup.get(Number(bond.second));
      if (!first || !second) continue;
      const firstElement = normalizedElement(first.element || first.name);
      const secondElement = normalizedElement(second.element || second.name);
      const isMetalDonor = (
        (SUPPORTED_METALS.has(firstElement) && DONOR_ELEMENTS.has(secondElement.toUpperCase()))
        || (SUPPORTED_METALS.has(secondElement) && DONOR_ELEMENTS.has(firstElement.toUpperCase()))
      );
      shape.addCylinder(
        [Number(first.x), Number(first.y), Number(first.z)],
        [Number(second.x), Number(second.y), Number(second.z)],
        isMetalDonor ? [1.0, 0.78, 0.08] : [0.55, 0.55, 0.55],
        bondRadius,
        `bond:${bond.first}-${bond.second}`
      );
    }
  }
  for (const atom of atoms) {
    shape.addSphere(
      [Number(atom.x), Number(atom.y), Number(atom.z)],
      atomColor(atom),
      previewAtomRadius(atom, showSpacefill, showSticks),
      `atom:${atom.index}`
    );
  }
  const comp = stage.addComponentFromObject(shape);
  comp.addRepresentation("buffer", {disablePicking: false});
}

function addProteinBindingLinks(stage, links) {
  if (!links?.length || !window.NGL) return;
  const shape = new NGL.Shape("protein metal-binding links");
  let hasLinks = false;
  for (const link of links) {
    const donor = link.donor || [];
    const metal = link.metal || [];
    if (donor.length < 3 || metal.length < 3) continue;
    shape.addCylinder(donor.map(Number), metal.map(Number), [0.95, 0.55, 0.08], 0.075);
    hasLinks = true;
  }
  if (!hasLinks) return;
  const comp = stage.addComponentFromObject(shape);
  comp.addRepresentation("buffer", {disablePicking: true});
}

function padLeft(value, length) {
  return String(value).slice(0, length).padStart(length, " ");
}

function padRight(value, length) {
  return String(value).slice(0, length).padEnd(length, " ");
}

function formatPdbAtomName(name, element) {
  const atomName = String(name || element || "X").slice(0, 4);
  const elementToken = String(element || "").replace(/[^A-Za-z]/g, "").slice(0, 2);
  return elementToken.length === 2 ? padRight(atomName, 4) : padLeft(atomName, 4);
}

function normalizedElement(element) {
  const token = String(element || "").replace(/[^A-Za-z]/g, "");
  return token ? token.slice(0, 1).toUpperCase() + token.slice(1).toLowerCase() : "";
}

function previewBondAllowed(bond, atomLookup) {
  const first = atomLookup.get(Number(bond.first));
  const second = atomLookup.get(Number(bond.second));
  if (!first || !second) return false;
  const firstElement = normalizedElement(first.element || first.name);
  const secondElement = normalizedElement(second.element || second.name);
  const firstIsMetal = SUPPORTED_METALS.has(firstElement);
  const secondIsMetal = SUPPORTED_METALS.has(secondElement);
  if (firstIsMetal === secondIsMetal) return true;
  const partner = firstIsMetal ? secondElement : firstElement;
  return DONOR_ELEMENTS.has(partner.toUpperCase());
}

function filterPreviewBonds(atoms, bonds) {
  const atomLookup = new Map((atoms || []).map((atom) => [Number(atom.index), atom]));
  return (bonds || []).filter((bond) => previewBondAllowed(bond, atomLookup));
}

function pdbFromAtoms(atoms, bonds, residueName = "LIG") {
  const residue = String(residueName || "LIG").trim().toUpperCase().slice(0, 3) || "LIG";
  const lines = ["HEADER    SIMPLE GUI molecule preview"];
  for (const atom of atoms || []) {
    const serial = Math.max(1, Number.parseInt(atom.index || lines.length, 10));
    const element = String(atom.element || atom.name || "C").replace(/[^A-Za-z]/g, "").slice(0, 2) || "C";
    const x = Number(atom.x || 0).toFixed(3).padStart(8, " ");
    const y = Number(atom.y || 0).toFixed(3).padStart(8, " ");
    const z = Number(atom.z || 0).toFixed(3).padStart(8, " ");
    lines.push(
      `HETATM${padLeft(serial, 5)} ${formatPdbAtomName(atom.name || element, element)} ${residue.padStart(3, " ")} A${padLeft(1, 4)}    ${x}${y}${z}  1.00  0.00          ${element.toUpperCase().padStart(2, " ")}`
    );
  }
  const conect = new Map();
  for (const bond of filterPreviewBonds(atoms, bonds)) {
    const first = Number.parseInt(bond.first, 10);
    const second = Number.parseInt(bond.second, 10);
    if (!Number.isFinite(first) || !Number.isFinite(second)) continue;
    if (!conect.has(first)) conect.set(first, new Set());
    if (!conect.has(second)) conect.set(second, new Set());
    conect.get(first).add(second);
    conect.get(second).add(first);
  }
  for (const first of [...conect.keys()].sort((a, b) => a - b)) {
    const bonded = [...conect.get(first)].sort((a, b) => a - b);
    for (let idx = 0; idx < bonded.length; idx += 4) {
      lines.push(`CONECT${padLeft(first, 5)}${bonded.slice(idx, idx + 4).map((item) => padLeft(item, 5)).join("")}`);
    }
  }
  lines.push("END");
  return `${lines.join("\n")}\n`;
}

function atomsFromPdb(pdb) {
  const atoms = [];
  for (const line of String(pdb || "").split(/\r?\n/)) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const element = (line.slice(76, 78).trim() || line.slice(12, 16).replace(/[^A-Za-z]/g, "").slice(0, 2) || "C");
    atoms.push({
      index: Number.parseInt(line.slice(6, 11), 10) || atoms.length + 1,
      name: line.slice(12, 16).trim(),
      element: element.slice(0, 1).toUpperCase() + element.slice(1).toLowerCase(),
      x: Number.parseFloat(line.slice(30, 38)) || 0,
      y: Number.parseFloat(line.slice(38, 46)) || 0,
      z: Number.parseFloat(line.slice(46, 54)) || 0,
    });
  }
  return atoms;
}

function atomByIndex(atomIndex) {
  return (state.currentAtoms || []).find((atom) => Number(atom.index) === Number(atomIndex)) || null;
}

function ensureSelectedMetalAtom() {
  if (!state.metMetals.length) {
    state.selectedMetalAtom = null;
    return null;
  }
  const current = Number(state.selectedMetalAtom);
  const match = state.metMetals.find((atom) => Number(atom.index) === current) || state.metMetals[0];
  state.selectedMetalAtom = Number(match.index);
  return state.selectedMetalAtom;
}

function addDashedCylinder(shape, startAtom, endAtom, color, radius = 0.045, segments = 13, name = "") {
  const start = [Number(startAtom.x), Number(startAtom.y), Number(startAtom.z)];
  const end = [Number(endAtom.x), Number(endAtom.y), Number(endAtom.z)];
  for (let idx = 0; idx < segments; idx += 2) {
    const t1 = idx / segments;
    const t2 = Math.min((idx + 1) / segments, 1);
    const p1 = [
      start[0] + (end[0] - start[0]) * t1,
      start[1] + (end[1] - start[1]) * t1,
      start[2] + (end[2] - start[2]) * t1,
    ];
    const p2 = [
      start[0] + (end[0] - start[0]) * t2,
      start[1] + (end[1] - start[1]) * t2,
      start[2] + (end[2] - start[2]) * t2,
    ];
    shape.addCylinder(p1, p2, color, radius, name);
  }
}

function highlightedAtomIndices() {
  const out = new Set([...state.selectedAtomIndices].map(Number));
  if (state.sceneKind === "metallophore") ensureSelectedMetalAtom();
  if (state.selectedMetalAtom) out.add(Number(state.selectedMetalAtom));
  for (const donorIndex of effectiveDonorIndicesForSelectedMetal()) out.add(Number(donorIndex));
  const group = groupForId(state.selectedRespGroupId);
  for (const atomIndex of group?.atom_indices || []) out.add(Number(atomIndex));
  return out;
}

function removeOverlayComponent() {
  if (!state.stage || !state.overlayComponent) return;
  const component = state.overlayComponent;
  state.overlayComponent = null;
  try {
    component.setVisibility?.(false);
  } catch (_err) {
    // Overlay visibility is best-effort across bundled NGL builds.
  }
  try {
    state.stage.removeComponent?.(component);
  } catch (_err) {
    // Some NGL builds prefer direct disposal below.
  }
  try {
    component.dispose?.();
  } catch (_err) {
    // Overlay disposal is best-effort across bundled NGL builds.
  }
  requestViewerRender();
}

function refreshMetallophoreOverlays() {
  if (!window.NGL || !state.stage || !["metallophore", "des"].includes(state.sceneKind)) return;
  if (state.sceneKind === "metallophore") ensureSelectedMetalAtom();
  removeOverlayComponent();
  const shape = new NGL.Shape("metallophore selection");
  let hasOverlay = false;
  const highlighted = highlightedAtomIndices();
  for (const atomIndex of highlighted) {
    const atom = atomByIndex(atomIndex);
    if (!atom) continue;
    const groupId = groupIdForAtom(atomIndex);
    const isMetal = Number(atomIndex) === Number(state.selectedMetalAtom);
    const color = isMetal ? [1.0, 0.74, 0.18] : groupId ? groupColorRgb(groupId) : [0.14, 0.48, 0.95];
    shape.addSphere([Number(atom.x), Number(atom.y), Number(atom.z)], color, isMetal ? 0.62 : 0.42, `atom:${atomIndex}`);
    hasOverlay = true;
  }
  const metal = atomByIndex(state.selectedMetalAtom);
  const effectiveDonors = effectiveDonorIndicesForSelectedMetal();
  if (state.sceneKind === "metallophore" && metal && effectiveDonors.size) {
    for (const donorIndex of effectiveDonors) {
      const donor = atomByIndex(donorIndex);
      if (!donor) continue;
      addDashedCylinder(shape, metal, donor, [1.0, 0.78, 0.08], 0.04, 13, `bond:${metal.index}-${donor.index}`);
      hasOverlay = true;
    }
  }
  if (!hasOverlay) return;
  state.overlayComponent = state.stage.addComponentFromObject(shape);
  state.overlayComponent.addRepresentation("buffer", {disablePicking: false});
  requestViewerRender();
}

function updateTableSelectionClasses() {
  const highlighted = highlightedAtomIndices();
  for (const row of document.querySelectorAll("[data-atom-index]")) {
    row.classList.toggle("selected-row", highlighted.has(Number(row.dataset.atomIndex)));
  }
  for (const row of document.querySelectorAll(".resp-group-row")) {
    row.classList.toggle("selected-row", Number(row.dataset.groupId) === Number(state.selectedRespGroupId));
  }
  if (state.busyCount > 0) applyBusyState();
}

function pickedAtomSerial(atom) {
  const serial = Number(atom?.serial);
  if (Number.isFinite(serial) && serial > 0) return serial;
  const index = Number(atom?.index);
  return Number.isFinite(index) ? index + 1 : null;
}

function metAtomRecord(atomIndex) {
  return (state.metAtoms || []).find((item) => Number(item.index) === Number(atomIndex)) || null;
}

function metAtomIsDonorCandidate(atomIndex) {
  return Boolean(metAtomRecord(atomIndex)?.is_donor_candidate);
}

function atomIndexFromDisplayedName(value) {
  if (typeof value !== "string") return null;
  const match = value.trim().match(/^([A-Za-z]{1,2})\s*(\d+)(?:\b|\s|\()/);
  if (!match) return null;
  const element = normalizedElement(match[1]);
  const atomIndex = Number(match[2]);
  const atom = metAtomRecord(atomIndex);
  if (!atom) return null;
  const atomElement = normalizedElement(atom.element || atom.name);
  const atomName = String(atom.name || "").replace(/\s+/g, "");
  if (atomElement === element || atomName.toUpperCase() === `${element}${atomIndex}`.toUpperCase()) return atomIndex;
  return null;
}

function atomIndexFromBondToken(first, second) {
  const a = Number(first);
  const b = Number(second);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  const firstAtom = metAtomRecord(a);
  const secondAtom = metAtomRecord(b);
  if (!firstAtom && !secondAtom) return null;
  if (Number(state.selectedMetalAtom) === a && secondAtom) return b;
  if (Number(state.selectedMetalAtom) === b && firstAtom) return a;
  if (firstAtom?.is_metal && metAtomIsDonorCandidate(b)) return b;
  if (secondAtom?.is_metal && metAtomIsDonorCandidate(a)) return a;
  if (metAtomIsDonorCandidate(a) && !metAtomIsDonorCandidate(b)) return a;
  if (metAtomIsDonorCandidate(b) && !metAtomIsDonorCandidate(a)) return b;
  return firstAtom ? a : b;
}

function atomIndexFromPickToken(value) {
  if (typeof value !== "string") return null;
  const direct = value.match(/^atom:(\d+)$/i);
  if (direct) return Number(direct[1]);
  const label = value.match(/\batom\s+(\d+)\b/i);
  if (label) return Number(label[1]);
  const bond = value.match(/^bond:(\d+)-(\d+)$/i);
  if (bond) return atomIndexFromBondToken(bond[1], bond[2]);
  const displayed = atomIndexFromDisplayedName(value);
  if (displayed) return displayed;
  return null;
}

function findAtomIndexInPickObject(value, depth = 0, seen = new Set()) {
  const fromToken = atomIndexFromPickToken(value);
  if (fromToken) return fromToken;
  if (!value || typeof value !== "object" || depth > 4 || seen.has(value)) return null;
  seen.add(value);
  const preferred = ["name", "label", "shape", "object", "picker", "data", "primitive", "instance", "buffer", "component", "repr", "representation"];
  const skip = new Set(["stage", "viewer", "signals", "mouseControls", "animationControls", "tasks", "children", "parent"]);
  const keys = [
    ...preferred,
    ...Object.keys(value).filter((key) => !preferred.includes(key) && !skip.has(key)),
  ].slice(0, 36);
  for (const key of keys) {
    let child = null;
    try {
      child = value[key];
    } catch (_err) {
      continue;
    }
    const found = findAtomIndexInPickObject(child, depth + 1, seen);
    if (found) return found;
  }
  return null;
}

function pickedMetallophoreAtomIndex(pickingProxy) {
  const serial = pickedAtomSerial(pickingProxy?.atom);
  if (serial) return serial;
  return findAtomIndexInPickObject(pickingProxy);
}

function residueKeyFromPickedAtom(atom) {
  const chain = String(atom?.chainname || atom?.chain || "").trim();
  const resno = String(atom?.resno || atom?.residue?.resno || "").trim();
  const resname = String(atom?.resname || atom?.residue?.resname || "").trim();
  const direct = `${chain}:${resname}:${resno}`;
  if (state.proteinResidues.some((row) => row.key === direct)) return direct;
  const row = state.proteinResidues.find((item) => (
    String(item.chain || "").trim() === chain
    && normalizedSeqid(item.seqid || item.resid) === normalizedSeqid(resno)
  ));
  return row?.key || "";
}

function refreshMetallophoreSelectionUi() {
  renderMetalsBox();
  renderDonorBox();
  renderRespGroupBox();
  refreshMetallophoreOverlays();
  updateTableSelectionClasses();
  updateViewerInfo();
}

function clearViewerSelection() {
  if (state.sceneKind === "metallophore") {
    state.selectedAtomIndices = new Set();
    state.selectedDonors = new Set();
    state.selectedRespGroupId = null;
    state.metCoordination = null;
    ensureSelectedMetalAtom();
    refreshMetallophoreSelectionUi();
    return;
  }
  if (state.sceneKind === "protein") {
    state.selectedResidueKeys = new Set();
    updateResidueRowSelection();
    if (state.currentPdb) rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
    return;
  }
  if (state.sceneKind === "des") {
    state.selectedAtomIndices = new Set();
    refreshMetallophoreOverlays();
    updateTableSelectionClasses();
  }
}

function isUserSelectedMetallophoreAtom(atomIndex) {
  const index = Number(atomIndex);
  return state.selectedAtomIndices.has(index) || state.selectedDonors.has(index);
}

function handleViewerPick(pickingProxy) {
  const atom = pickingProxy?.atom;
  const metallophoreAtomIndex = state.sceneKind === "metallophore" ? pickedMetallophoreAtomIndex(pickingProxy) : null;
  if (!atom && !metallophoreAtomIndex) {
    clearViewerSelection();
    return;
  }
  const multi = Boolean(state.pickModifiers.ctrl || state.pickModifiers.meta || state.pickModifiers.shift);
  if (state.sceneKind === "metallophore") {
    const clicked = state.metAtoms.find((item) => Number(item.index) === Number(metallophoreAtomIndex));
    if (!clicked) return;
    const atomIndex = Number(clicked.index);
    const alreadySelected = isUserSelectedMetallophoreAtom(atomIndex);
    if (multi) {
      if (clicked.is_metal) {
        state.selectedMetalAtom = atomIndex;
        state.selectedAtomIndices.delete(atomIndex);
        state.selectedDonors.delete(atomIndex);
      } else if (alreadySelected) {
        state.selectedAtomIndices.delete(atomIndex);
        state.selectedDonors.delete(atomIndex);
      } else {
        state.selectedAtomIndices.add(atomIndex);
        if (clicked.is_donor_candidate) state.selectedDonors.add(atomIndex);
      }
    } else if (clicked.is_metal) {
      state.selectedAtomIndices = new Set();
      state.selectedDonors = new Set();
      state.selectedRespGroupId = null;
      state.selectedMetalAtom = atomIndex;
    } else if (alreadySelected) {
      state.selectedAtomIndices = new Set();
      state.selectedDonors = new Set();
      state.selectedRespGroupId = null;
    } else {
      state.selectedAtomIndices = new Set([atomIndex]);
      state.selectedDonors = clicked.is_donor_candidate ? new Set([atomIndex]) : new Set();
      state.selectedRespGroupId = null;
    }
    state.metCoordination = null;
    ensureSelectedMetalAtom();
    refreshMetallophoreSelectionUi();
    return;
  }
  if (state.sceneKind === "protein") {
    const key = residueKeyFromPickedAtom(atom);
    if (!key) return;
    if (multi) {
      if (state.selectedResidueKeys.has(key)) state.selectedResidueKeys.delete(key);
      else state.selectedResidueKeys.add(key);
    } else if (state.selectedResidueKeys.has(key) && state.selectedResidueKeys.size === 1) {
      state.selectedResidueKeys = new Set();
    } else {
      state.selectedResidueKeys = new Set([key]);
    }
    renderResiduePanel(state.proteinResidues || []);
    rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
    return;
  }
  if (state.sceneKind === "des") {
    const serial = pickedAtomSerial(atom);
    if (!serial) return;
    if (!multi) state.selectedAtomIndices = new Set([serial]);
    else if (state.selectedAtomIndices.has(serial)) state.selectedAtomIndices.delete(serial);
    else state.selectedAtomIndices.add(serial);
    refreshMetallophoreOverlays();
  }
}

function updateViewerInfo() {
  const base = state.viewerInfoBase || "";
  if (state.sceneKind !== "metallophore") {
    $("viewer_info").textContent = base;
    return;
  }
  const metal = state.metMetals.find((item) => Number(item.index) === Number(state.selectedMetalAtom));
  const element = metal ? selectedMetalElement(metal) : "";
  const hint = coordinationHint(element);
  const donors = state.selectedDonors.size;
  const coordination = collectSelectedMetalCoordination();
  const parts = [];
  if (base) parts.push(base);
  if (element && hint) parts.push(`${element} common coordination # ${hint}`);
  if (element && coordination) parts.push(`charge +${coordination.formal_charge}, target CN ${coordination.target_cn}`);
  if (element) parts.push(`${donors} donor${donors === 1 ? "" : "s"} selected`);
  $("viewer_info").textContent = parts.join(" | ");
}

function setViewerDefaults(kind) {
  const defaults = {
    protein: {cartoon: true, labels: false, lines: false, sticks: false, spacefill: false},
    metallophore: {cartoon: false, labels: false, lines: false, sticks: true, spacefill: false},
    des: {cartoon: false, labels: false, lines: false, sticks: true, spacefill: true},
  }[kind];
  if (!defaults) return;
  if ($("show_cartoon")) $("show_cartoon").checked = defaults.cartoon;
  if ($("show_labels")) $("show_labels").checked = defaults.labels;
  if ($("show_lines")) $("show_lines").checked = defaults.lines;
  if ($("show_sticks")) $("show_sticks").checked = defaults.sticks;
  if ($("show_spacefill")) $("show_spacefill").checked = defaults.spacefill;
  syncViewerControls();
}

async function renderPdbScene(pdb, kind, extra = {}) {
  const stage = ensureStage();
  if (!stage) return;
  removeOverlayComponent();
  stage.removeAllComponents();
  state.structureComponent = null;
  const blob = new Blob([pdb || "END\n"], {type: "text/plain"});
  const comp = await stage.loadFile(blob, {ext: "pdb", name: kind});
  state.structureComponent = comp;
  const showLines = $("show_lines")?.checked;
  const showSticks = $("show_sticks")?.checked;
  const showSpacefill = $("show_spacefill")?.checked;
  const explicitAtoms = Object.prototype.hasOwnProperty.call(extra, "atoms") ? (extra.atoms || []) : null;
  const parsedAtoms = atomsFromPdb(pdb);
  const sceneAtoms = kind === "metallophore" && explicitAtoms ? explicitAtoms : parsedAtoms;
  if (kind === "protein") {
    if ($("show_cartoon").checked) comp.addRepresentation("cartoon", {color: "chainname"});
    if (showLines) comp.addRepresentation("line", {sele: "all", color: "element", linewidth: 1});
    if (showSticks) comp.addRepresentation("ball+stick", {sele: "all", radiusScale: 0.48});
    if (showSpacefill) comp.addRepresentation("spacefill", {sele: "all", scale: 0.82});
    const highlightSets = extra.highlight_sets || state.proteinHighlightSets || {};
    const metalBindingSelection = residueSelectionFromKeys(highlightSets.metal_binding || []);
    const missingSelection = residueSelectionFromKeys(highlightSets.missing || []);
    const disulfideSelection = residueSelectionFromKeys([...state.disulfideKeys, ...(highlightSets.disulfide || [])]);
    const propkaSelection = residueSelectionFromKeys([...state.propkaResidueKeys]);
    const selectedSelection = residueSelectionFromKeys([...state.selectedResidueKeys]);
    const metalSelection = proteinExistingMetalSelection();
    const insertedMetalSelection = proteinInsertedMetalSelection();
    if (metalBindingSelection) {
      comp.addRepresentation("licorice", {sele: metalBindingSelection, color: "element", radius: 0.28});
    }
    if (missingSelection) {
      comp.addRepresentation("licorice", {sele: missingSelection, color: "green", radius: 0.24});
      comp.addRepresentation("spacefill", {sele: missingSelection, color: "green", scale: 0.65, opacity: 0.36, transparent: true});
    }
    if (disulfideSelection) {
      comp.addRepresentation("licorice", {sele: disulfideSelection, color: "yellow", radius: 0.3});
    }
    if (propkaSelection) {
      comp.addRepresentation("licorice", {sele: propkaSelection, color: "red", radius: 0.28});
    }
    if (metalSelection) {
      comp.addRepresentation("spacefill", {sele: metalSelection, color: "element", scale: 0.92});
    }
    if (insertedMetalSelection) {
      comp.addRepresentation("ball+stick", {sele: insertedMetalSelection, color: "magenta", radiusScale: 0.42});
      comp.addRepresentation("spacefill", {sele: insertedMetalSelection, color: "magenta", scale: 0.62});
    }
    if (selectedSelection) {
      comp.addRepresentation("licorice", {sele: selectedSelection, color: "magenta", radius: 0.34});
      comp.addRepresentation("spacefill", {sele: selectedSelection, color: "magenta", scale: 0.82, opacity: 0.52, transparent: true});
    }
    addProteinBindingLinks(stage, extra.metal_binding_links || state.proteinBindingLinks || []);
  } else if (kind === "metallophore") {
    addFilteredMetallophoreRepresentation(stage, sceneAtoms, extra.bonds || state.currentBonds || [], {
      showLines,
      showSticks,
      showSpacefill,
    });
  } else {
    const radiusScale = kind === "des" ? 0.3 : 0.75;
    if (showLines) comp.addRepresentation("line", {color: "element", linewidth: kind === "des" ? 1 : 2});
    if (showSticks) comp.addRepresentation("ball+stick", {radiusScale});
    if (showSpacefill) comp.addRepresentation("spacefill", {scale: kind === "des" ? 0.55 : 1.0});
  }
  const labelAtoms = kind === "metallophore" && $("show_labels")?.checked ? sceneAtoms : [];
  if (labelAtoms.length) addAtomLabels(stage, labelAtoms);
  addBoxLines(stage, extra.box_lines || []);
  addIonDots(stage, extra.ions || []);
  if (!extra.keepCamera) {
    stage.autoView(0);
    captureHomeOrientation();
    setTimeout(captureHomeOrientation, 50);
  }
  state.currentPdb = pdb || "";
  state.currentBonds = Object.prototype.hasOwnProperty.call(extra, "bonds") ? (extra.bonds || []) : state.currentBonds;
  state.currentAtoms = sceneAtoms;
  state.currentExtra = {...extra, bonds: state.currentBonds};
  if (kind === "metallophore") state.currentExtra.atoms = state.currentAtoms;
  else delete state.currentExtra.atoms;
  state.viewerInfoBase = extra.info || "";
  $("viewer-overlay").textContent = extra.message || `${kind} preview loaded.`;
  state.sceneKind = kind;
  refreshMetallophoreOverlays();
  updateTableSelectionClasses();
  updateViewerInfo();
}

async function rerenderCurrentScene() {
  if (!state.currentPdb || !state.sceneKind) return;
  await renderPdbScene(state.currentPdb, state.sceneKind, {...state.currentExtra, keepCamera: true});
}

function formatCharge(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(4);
}

function sortRows(rows, tableKey) {
  const sort = state.sorts[tableKey] || {column: "number", dir: "asc"};
  const factor = sort.dir === "desc" ? -1 : 1;
  return [...rows].sort((a, b) => {
    const av = comparableValue(a, sort.column);
    const bv = comparableValue(b, sort.column);
    if (av < bv) return -1 * factor;
    if (av > bv) return 1 * factor;
    return Number(a.index || 0) - Number(b.index || 0);
  });
}

function comparableValue(row, column) {
  if (column === "number") return Number(row.index || row.site || 0);
  if (column === "x" || column === "y" || column === "z") return Number(row[column] || 0);
  if (column === "charge") return row.partial_charge === null || row.partial_charge === undefined ? 9999 : Number(row.partial_charge);
  return String(row[column] || row.name || "").toLowerCase();
}

function sortableHeader(tableKey, column, label) {
  const sort = state.sorts[tableKey] || {};
  const suffix = sort.column === column ? (sort.dir === "asc" ? " ^" : " v") : "";
  return `<th class="sortable" data-table="${tableKey}" data-column="${column}">${label}${suffix}</th>`;
}

function attachSortHandlers(tableKey, renderFn) {
  for (const th of document.querySelectorAll(`th.sortable[data-table="${tableKey}"]`)) {
    th.addEventListener("click", () => {
      const current = state.sorts[tableKey] || {};
      const column = th.dataset.column;
      state.sorts[tableKey] = {
        column,
        dir: current.column === column && current.dir === "asc" ? "desc" : "asc",
      };
      renderFn();
    });
  }
}

function selectedMetalElement(atom) {
  return state.metMetalElements.get(Number(atom.index)) || atom.element;
}

function snapshotMetallophoreData(data) {
  return cloneJson({
    source_path: data.source_path || "",
    atoms: data.atoms || [],
    bonds: data.bonds || [],
    metals: data.metals || [],
    donor_candidates: data.donor_candidates || [],
    pdb: data.pdb || "",
    mol2: data.mol2 || "",
    group_constraints: data.group_constraints || null,
  });
}

function rememberInitialMetallophore(data) {
  state.metOriginal = {
    data: snapshotMetallophoreData(data),
    inputPath: data.source_path || $("met_input_path").value.trim(),
    residueName: $("met_residue").value.trim() || "LIG",
  };
}

function moleculePayloadFromState(residueName = "LIG") {
  const atoms = cloneJson(state.metAtoms || []);
  const bonds = cloneJson(state.metBonds || []);
  const metals = atoms.filter((atom) => SUPPORTED_METALS.has(normalizedElement(atom.element || atom.name)));
  const donorCandidates = atoms.filter((atom) => {
    const element = normalizedElement(atom.element || atom.name);
    return DONOR_ELEMENTS.has(element.toUpperCase()) && !SUPPORTED_METALS.has(element);
  });
  return {
    source_path: "gui-edited-preview",
    atoms,
    bonds,
    metals,
    donor_candidates: donorCandidates,
    pdb: pdbFromAtoms(atoms, bonds, residueName),
    mol2: "",
    group_constraints: state.metGroupConstraints || null,
    metal_coordination: null,
  };
}

function renderMetalsAndDonors(data) {
  state.metAtoms = data.atoms || [];
  state.metBonds = data.bonds || [];
  state.metMetals = data.metals || [];
  state.metDonors = data.donor_candidates || [];
  state.currentAtoms = state.metAtoms;
  state.currentBonds = state.metBonds;
  const existingAtomIndices = new Set(state.metAtoms.map((atom) => Number(atom.index)));
  state.metInsertedMetalIndices = new Set([...state.metInsertedMetalIndices].filter((index) => existingAtomIndices.has(Number(index))));
  state.metGroupConstraints = data.group_constraints || null;
  state.metCoordination = data.metal_coordination || null;
  state.selectedDonors = new Set();
  state.selectedAtomIndices = new Set();
  state.selectedRespGroupId = null;
  state.metMetalElements = new Map(state.metMetals.map((atom) => [Number(atom.index), atom.element]));
  state.selectedMetalAtom = state.metMetals[0]?.index || null;
  ensureSelectedMetalAtom();
  renderMetalsBox();
  renderDonorBox();
  renderRespGroupControls();
  renderRespGroupBox();
  renderSystemMetalCharges(state.metMetals.map((item, idx) => ({
    site: idx + 1,
    atom_index: Number(item.index),
    element: selectedMetalElement(item),
    label: `${selectedMetalElement(item)} #${item.index}`,
  })));
  refreshMetallophoreOverlays();
  updateTableSelectionClasses();
  updateViewerInfo();
}

function chargeOptions(element) {
  const record = (state.bootstrap?.supported_metals || []).find((item) => item.element === element);
  return record?.charges?.length ? record.charges : [2, 3];
}

function supportedMetalOptions(selected) {
  return (state.bootstrap?.supported_metals || []).map((m) => `<option value="${m.element}" ${m.element === selected ? "selected" : ""}>${m.element}</option>`).join("");
}

function updateMetalAtomElement(atomIndex, element) {
  const normalized = String(element || "C").trim();
  for (const list of [state.metAtoms, state.metMetals, state.currentAtoms]) {
    const atom = list.find((item) => Number(item.index) === Number(atomIndex));
    if (atom) {
      atom.element = normalized;
      atom.name = normalized;
      atom.is_metal = true;
    }
  }
  state.selectedMetalAtom = Number(atomIndex);
  state.metCoordination = null;
  if (state.sceneKind === "metallophore" && state.metAtoms.length) {
    state.currentPdb = pdbFromAtoms(state.metAtoms, state.metBonds, $("met_residue").value || "LIG");
    state.currentExtra = {...state.currentExtra, atoms: state.metAtoms, bonds: state.metBonds};
    rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
  } else {
    refreshMetallophoreOverlays();
  }
  updateViewerInfo();
}

function selectAtomsForHighlight(atomIndices, groupId = null) {
  state.selectedAtomIndices = new Set((atomIndices || []).map(Number));
  state.selectedRespGroupId = groupId == null ? null : Number(groupId);
  updateTableSelectionClasses();
  refreshMetallophoreOverlays();
  refreshMetalCoordinationStatus();
  updateViewerInfo();
}

function renderMetalsBox() {
  if (!state.metMetals.length) {
    $("metals_box").textContent = "No supported 12-6-4 metal atoms were detected.";
    return;
  }
  ensureSelectedMetalAtom();
  const rows = sortRows(state.metMetals, "metals");
  const respInput = isRespInputMode();
  $("metals_box").innerHTML = `
    <table><thead><tr>
      <th>Use</th>
      ${sortableHeader("metals", "name", "Atom")}
      ${sortableHeader("metals", "number", "Number")}
      <th>Element</th>
      ${respInput ? "<th>RESP group</th>" : `
      ${sortableHeader("metals", "x", "X")}
      ${sortableHeader("metals", "y", "Y")}
      ${sortableHeader("metals", "z", "Z")}
      ${sortableHeader("metals", "charge", "Partial charge")}`}
    </tr></thead><tbody>
      ${rows.map((atom) => `
        <tr class="choice-row" data-atom-index="${atom.index}">
          <td><input class="met-metal-radio" name="met_metal" type="radio" value="${atom.index}" ${Number(atom.index) === Number(state.selectedMetalAtom) ? "checked" : ""}></td>
          <td>${escapeHtml(atom.name)}</td>
          <td>${atom.index}</td>
          <td><select class="met-metal-element" data-index="${atom.index}">${supportedMetalOptions(selectedMetalElement(atom))}</select></td>
          ${respInput ? `<td>${escapeHtml(groupLabelForAtom(atom.index))}</td>` : `
          <td>${Number(atom.x).toFixed(2)}</td>
          <td>${Number(atom.y).toFixed(2)}</td>
          <td>${Number(atom.z).toFixed(2)}</td>
          <td>${formatCharge(atom.partial_charge)}</td>`}
        </tr>`).join("")}
    </tbody></table>`;
  for (const node of document.querySelectorAll(".met-metal-radio")) {
    node.addEventListener("change", () => {
      state.selectedMetalAtom = Number(node.value);
      state.metCoordination = null;
      selectAtomsForHighlight([]);
    });
  }
  for (const node of document.querySelectorAll(".met-metal-element")) {
    node.addEventListener("change", () => {
      state.metMetalElements.set(Number(node.dataset.index), node.value);
      updateMetalAtomElement(Number(node.dataset.index), node.value);
      renderSystemMetalCharges(state.metMetals.map((item, idx) => ({
        site: idx + 1,
        atom_index: Number(item.index),
        element: selectedMetalElement(item),
        label: `${selectedMetalElement(item)} #${item.index}`,
      })));
    });
  }
  for (const row of $("metals_box").querySelectorAll("[data-atom-index]")) {
    row.addEventListener("click", (ev) => {
      if (["INPUT", "SELECT", "OPTION"].includes(ev.target.tagName)) return;
      state.selectedMetalAtom = Number(row.dataset.atomIndex);
      state.metCoordination = null;
      for (const node of document.querySelectorAll(".met-metal-radio")) {
        node.checked = Number(node.value) === Number(state.selectedMetalAtom);
      }
      selectAtomsForHighlight([]);
    });
  }
  attachSortHandlers("metals", renderMetalsBox);
  updateTableSelectionClasses();
}

function renderDonorBox() {
  if (!state.metDonors.length) {
    $("donor_box").textContent = "No O/N/S/P donor candidates were detected.";
    return;
  }
  const rows = sortRows(state.metDonors, "donors");
  const respInput = isRespInputMode();
  $("donor_box").innerHTML = `
    <table><thead><tr>
      <th>Select</th>
      ${sortableHeader("donors", "name", "Atom")}
      ${sortableHeader("donors", "number", "Number")}
      ${sortableHeader("donors", "element", "Element")}
      ${respInput ? "<th>RESP group</th>" : `
      ${sortableHeader("donors", "x", "X")}
      ${sortableHeader("donors", "y", "Y")}
      ${sortableHeader("donors", "z", "Z")}
      ${sortableHeader("donors", "charge", "Partial charge")}`}
    </tr></thead><tbody>
      ${rows.map((atom) => `
        <tr class="choice-row" data-atom-index="${atom.index}">
          <td><input class="donor-check" type="checkbox" value="${atom.index}" ${state.selectedDonors.has(Number(atom.index)) ? "checked" : ""}></td>
          <td>${escapeHtml(atom.name)}</td>
          <td>${atom.index}</td>
          <td>${escapeHtml(atom.element)}</td>
          ${respInput ? `<td>${escapeHtml(groupLabelForAtom(atom.index))}</td>` : `
          <td>${Number(atom.x).toFixed(2)}</td>
          <td>${Number(atom.y).toFixed(2)}</td>
          <td>${Number(atom.z).toFixed(2)}</td>
          <td>${formatCharge(atom.partial_charge)}</td>`}
        </tr>`).join("")}
    </tbody></table>`;
  for (const node of document.querySelectorAll(".donor-check")) {
    node.addEventListener("change", () => {
      state.metCoordination = null;
      if (node.checked) state.selectedDonors.add(Number(node.value));
      else state.selectedDonors.delete(Number(node.value));
      selectAtomsForHighlight([...state.selectedDonors, state.selectedMetalAtom].filter(Boolean));
    });
  }
  for (const row of $("donor_box").querySelectorAll("[data-atom-index]")) {
    row.addEventListener("click", (ev) => {
      if (["INPUT", "SELECT", "OPTION"].includes(ev.target.tagName)) return;
      selectAtomsForHighlight([Number(row.dataset.atomIndex)]);
    });
  }
  attachSortHandlers("donors", renderDonorBox);
  updateTableSelectionClasses();
}

function renderRespGroupControls() {
  const controls = $("resp_group_controls");
  if (!controls) return;
  controls.hidden = !isRespInputMode();
  if (controls.hidden) return;
  const constraints = state.metGroupConstraints || {};
  if ($("resp_group_mode").options.length) {
    $("resp_group_mode").value = constraints.auto_group_mode || $("resp_group_mode").value || "hydrogen_and_symmetry";
  }
  if ($("resp_group_graph_method").options.length) {
    $("resp_group_graph_method").value = constraints.auto_group_graph_method || $("resp_group_graph_method").value || "connectivity";
    if (!$("resp_group_graph_method").value) $("resp_group_graph_method").value = "connectivity";
  }
  const warning = constraints.auto_group_graph_warning || "";
  const reason = constraints.auto_group_exclusion_reason || "";
  $("resp_group_note").textContent = [warning, reason].filter(Boolean).join(" ");
}

function renderRespGroupBox() {
  const box = $("resp_group_box");
  if (!box) return;
  if (!isRespInputMode()) {
    box.textContent = "RESP group review is available in RESP input generation mode.";
    return;
  }
  const groups = state.metGroupConstraints?.groups || [];
  if (!groups.length) {
    box.textContent = "No RESP equality groups were suggested for this structure.";
    updateTableSelectionClasses();
    return;
  }
  const atomLookup = new Map((state.metAtoms || []).map((atom) => [Number(atom.index), atom]));
  box.innerHTML = `
    <table><thead><tr><th>Group</th><th>Label</th><th>Atoms</th><th>Count</th></tr></thead><tbody>
      ${groups.map((group) => {
        const atoms = (group.atom_indices || []).map((idx) => atomLookup.get(Number(idx))).filter(Boolean);
        const names = atoms.map((atom) => atomDisplayName(atom)).join(", ");
        return `
          <tr class="choice-row resp-group-row" data-group-id="${group.group_id}">
            <td><span class="group-chip" style="background:${groupColor(group.group_id)}"></span>Grp ${group.group_id}</td>
            <td>${escapeHtml(group.label || "custom")}</td>
            <td>${escapeHtml(names || (group.atom_indices || []).join(", "))}</td>
            <td>${(group.atom_indices || []).length}</td>
          </tr>`;
      }).join("")}
    </tbody></table>`;
  for (const row of box.querySelectorAll(".resp-group-row")) {
    row.addEventListener("click", () => {
      const group = groupForId(row.dataset.groupId);
      selectAtomsForHighlight(group?.atom_indices || [], Number(row.dataset.groupId));
    });
  }
  updateTableSelectionClasses();
}

function applyGroupConstraints(groupConstraints) {
  state.metGroupConstraints = groupConstraints || null;
  state.selectedRespGroupId = null;
  renderMetalsBox();
  renderDonorBox();
  renderRespGroupControls();
  renderRespGroupBox();
  if (state.sceneKind === "metallophore" && state.currentPdb) {
    rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
  } else {
    refreshMetallophoreOverlays();
  }
}

async function regenerateRespGroups() {
  if (!isRespInputMode()) return;
  if (!state.metAtoms.length) {
    renderRespGroupControls();
    renderRespGroupBox();
    return;
  }
  const payload = collectPayload();
  payload.metallophore.auto_group_mode = $("resp_group_mode").value;
  payload.metallophore.auto_group_graph_method = $("resp_group_graph_method").value;
  setStatus("Rebuilding RESP symmetry groups...", "warn");
  const data = await api("/api/metallophore/groups", payload);
  applyGroupConstraints(data.group_constraints);
  setStatus("RESP symmetry groups updated.", "ok");
}

function renderSystemMetalCharges(sites) {
  if (!sites || !sites.length) {
    $("system_metal_charges").textContent = "No current metal sites.";
    refreshRespChargeHint();
    syncSystemC4ParameterSet(false, false);
    return;
  }
  const showCoordination = $("workflow_type")?.value === "metallophore";
  const header = showCoordination
    ? "<tr><th>Site</th><th>Metal</th><th>Charge</th><th>Target CN</th><th>Donors</th></tr>"
    : "<tr><th>Site</th><th>Metal</th><th>Charge</th></tr>";
  $("system_metal_charges").innerHTML = `
    <table><thead>${header}</thead><tbody>
      ${sites.map((site) => {
        const savedCoordination = state.metCoordination && Number(state.metCoordination.metal_atom_index) === Number(site.atom_index);
        const charge = savedCoordination && state.metCoordination.formal_charge !== null && state.metCoordination.formal_charge !== undefined
          ? Number(state.metCoordination.formal_charge)
          : defaultMetalCharge(site.element);
        let coordinationCells = "";
        if (showCoordination) {
          const cn = savedCoordination && state.metCoordination.target_cn === null
            ? MANUAL_CN_VALUE
            : savedCoordination && state.metCoordination.target_cn !== null && state.metCoordination.target_cn !== undefined
              ? String(Number(state.metCoordination.target_cn))
              : String(defaultCoordinationNumber(site.element, charge));
          const cnOptions = [
            ...coordinationOptions(site.element, charge).map((item) => `<option value="${item}" ${String(item) === String(cn) ? "selected" : ""}>${item}</option>`),
            `<option value="${MANUAL_CN_VALUE}" ${cn === MANUAL_CN_VALUE ? "selected" : ""}>Manual selection</option>`,
          ].join("");
          coordinationCells = `
            <td><select class="metal-cn-select">${cnOptions}</select></td>
            <td class="metal-cn-status">${escapeHtml(coordinationStatusForAtom(site.atom_index))}</td>`;
        }
        return `
        <tr class="metal-charge-row" data-site="${site.site}" data-atom-index="${site.atom_index || ""}" data-element="${escapeHtml(site.element)}">
          <td>${site.site}</td>
          <td>${escapeHtml(site.label || site.element)}</td>
          <td><select class="metal-charge-select">${chargeOptions(site.element).map((item) => `<option value="${item}" ${Number(item) === Number(charge) ? "selected" : ""}>+${item}</option>`).join("")}</select></td>
          ${coordinationCells}
        </tr>`;
      }).join("")}
    </tbody></table>`;
  for (const row of document.querySelectorAll(".metal-charge-row")) {
    row.querySelector(".metal-charge-select")?.addEventListener("change", () => {
      updateCoordinationSelect(row);
      syncSystemC4ParameterSet(true, true);
      if (showCoordination) {
        state.metCoordination = null;
        refreshMetalCoordinationStatus();
        updateViewerInfo();
      }
    });
    row.querySelector(".metal-cn-select")?.addEventListener("change", () => {
      state.metCoordination = null;
      refreshMetalCoordinationStatus();
      updateViewerInfo();
    });
  }
  refreshMetalCoordinationStatus();
  syncSystemC4ParameterSet(true, true);
}

function resetProteinTables() {
  state.proteinMetals = [];
  state.proteinSourceMetals = [];
  state.proteinMetalActions = new Map();
  state.proteinSiteRespApproved = false;
  state.proteinSiteRespClusters = [];
  if ($("protein_site_resp_mode")) $("protein_site_resp_mode").value = "standard_ff";
  if ($("protein_site_resp_job_dir")) $("protein_site_resp_job_dir").value = "";
  if ($("protein_site_resp_multiplicity_confirmed")) $("protein_site_resp_multiplicity_confirmed").checked = false;
  state.proteinInsertions = [];
  state.proteinInsertionDonors = [];
  state.proteinResidues = [];
  state.proteinHighlightSets = {};
  state.proteinBindingLinks = [];
  state.selectedResidueKeys = new Set();
  state.lastResidueIndex = -1;
  state.propkaResidueKeys = new Set();
  state.disulfideCandidates = [];
  state.disulfideKeys = new Set();
  state.disulfideTokens = [];
  state.proteinLoaded = false;
  state.lastProteinMissingLoopAction = "";
  if ($("protein_metal_box")) $("protein_metal_box").textContent = "Load a protein to review metals. Metals are preserved by default.";
  if ($("protein_insert_donor_box")) $("protein_insert_donor_box").textContent = "Select residue rows, then load donor candidates.";
  if ($("protein_insert_box")) $("protein_insert_box").textContent = "No inserted metals planned.";
  if ($("protein_residue_summary")) $("protein_residue_summary").textContent = "Load a protein preview to populate residue selection.";
  if ($("residue_info_box")) $("residue_info_box").textContent = "Load a protein preview to list residues.";
  if ($("propka_state")) $("propka_state").textContent = "Current PropKa choice: run PropKa to review selectable residue-state changes.";
  if ($("propka_box")) $("propka_box").textContent = "Run PropKa to review selectable residue-state changes.";
  if ($("disulfide_state")) $("disulfide_state").textContent = "Load a protein preview to detect CYS-CYS candidate pairs.";
  if ($("disulfide_box")) $("disulfide_box").textContent = "No CYS-CYS candidates loaded.";
  if ($("apply_protein_site_resp")) $("apply_protein_site_resp").hidden = true;
  if ($("protein_site_resp_review")) $("protein_site_resp_review").hidden = true;
}

function proteinSelectionFromKey(key) {
  const parts = String(key || "").split(":");
  if (parts.length < 3) return "";
  const chain = parts[0];
  const seq = parts[2].trim().split(/\s+/)[0];
  return chain ? `:${chain} and ${seq}` : `${seq}`;
}

function proteinResidueByKey(key) {
  return (state.proteinResidues || []).find((row) => row.key === key) || null;
}

function normalizedSeqid(value) {
  return String(value || "").trim().split(/\s+/)[0];
}

function proteinResidueKeyForCandidate(item) {
  const chain = String(item?.chain || "").trim();
  const seqid = normalizedSeqid(item?.seqid);
  const row = (state.proteinResidues || []).find((candidate) => (
    String(candidate.chain || "").trim() === chain && normalizedSeqid(candidate.seqid || candidate.resid) === seqid
  ));
  return row?.key || "";
}

function residueSelectionFromKey(key) {
  const row = proteinResidueByKey(key);
  return row?.selection || proteinSelectionFromKey(key);
}

function residueSelectionFromKeys(keys) {
  const selections = [...new Set(keys || [])]
    .map((key) => residueSelectionFromKey(key))
    .filter(Boolean);
  if (!selections.length) return "";
  return selections.length === 1 ? selections[0] : selections.map((selection) => `(${selection})`).join(" or ");
}

function residueFocusSelectionForKey(key, flank = 2) {
  const rows = state.proteinResidues || [];
  const idx = rows.findIndex((row) => row.key === key);
  if (idx < 0) return residueSelectionFromKey(key);
  const chain = String(rows[idx].chain || "");
  const keys = [];
  for (let pos = Math.max(0, idx - flank); pos <= Math.min(rows.length - 1, idx + flank); pos += 1) {
    if (String(rows[pos].chain || "") === chain) keys.push(rows[pos].key);
  }
  return residueSelectionFromKeys(keys) || residueSelectionFromKey(key);
}

function proteinMetalSelection() {
  return residueSelectionFromKeys((state.proteinMetals || []).map((site) => site.key));
}

function proteinInsertedMetalSelection() {
  return residueSelectionFromKeys(state.proteinHighlightSets?.inserted_metals || []);
}

function proteinExistingMetalSelection() {
  const inserted = new Set(state.proteinHighlightSets?.inserted_metals || []);
  return residueSelectionFromKeys((state.proteinMetals || []).map((site) => site.key).filter((key) => !inserted.has(key)));
}

function dynamicResidueNotes(row) {
  const notes = new Set(row.notes || []);
  if (state.proteinHighlightSets?.metal_binding?.includes(row.key)) notes.add("Metal binding");
  if (state.proteinHighlightSets?.missing?.includes(row.key)) notes.add("Missing-loop flank");
  if (state.disulfideKeys.has(row.key) || state.proteinHighlightSets?.disulfide?.includes(row.key)) notes.add("Disulfide");
  if (state.propkaResidueKeys.has(row.key)) notes.add("PropKa");
  return [...notes];
}

function residueRowClass(row) {
  const classes = [];
  if (state.selectedResidueKeys.has(row.key)) classes.push("selected-residue");
  if (state.proteinHighlightSets?.metal_binding?.includes(row.key)) classes.push("note-metal-binding");
  if (state.proteinHighlightSets?.missing?.includes(row.key)) classes.push("note-missing");
  if (state.disulfideKeys.has(row.key) || state.proteinHighlightSets?.disulfide?.includes(row.key)) classes.push("note-disulfide");
  if (state.propkaResidueKeys.has(row.key)) classes.push("note-propka");
  if (row.classification === "metal") classes.push("note-metal");
  return classes.length ? ` class="${classes.join(" ")}"` : "";
}

function updateResidueRowSelection() {
  const box = $("residue_info_box");
  if (!box) return;
  for (const tr of box.querySelectorAll("tbody tr")) {
    tr.classList.toggle("selected-residue", state.selectedResidueKeys.has(tr.dataset.key));
  }
  updateProteinLinkedTableSelection();
}

function updateProteinLinkedTableSelection() {
  for (const row of document.querySelectorAll(".propka-row")) {
    row.classList.toggle("selected-row", state.selectedResidueKeys.has(row.dataset.key));
  }
  for (const row of document.querySelectorAll(".disulfide-row")) {
    row.classList.toggle(
      "selected-row",
      state.selectedResidueKeys.has(row.dataset.keyA) || state.selectedResidueKeys.has(row.dataset.keyB)
    );
  }
}

function selectResidueRows(rows, idx, ev) {
  const key = rows[idx]?.key;
  if (!key) return;
  if (ev?.shiftKey && state.lastResidueIndex >= 0) {
    const lo = Math.min(state.lastResidueIndex, idx);
    const hi = Math.max(state.lastResidueIndex, idx);
    state.selectedResidueKeys = new Set(rows.slice(lo, hi + 1).map((row) => row.key));
    state.lastResidueIndex = idx;
  } else if (ev?.ctrlKey || ev?.metaKey) {
    if (state.selectedResidueKeys.has(key)) state.selectedResidueKeys.delete(key);
    else state.selectedResidueKeys.add(key);
    state.lastResidueIndex = idx;
  } else {
    state.selectedResidueKeys = new Set([key]);
    state.lastResidueIndex = idx;
  }
  updateResidueRowSelection();
}

function renderResiduePanel(rows) {
  const box = $("residue_info_box");
  if (!box) return;
  const residueRows = rows || [];
  if (!residueRows.length) {
    box.textContent = "Load a protein preview to list residues.";
    return;
  }
  const body = residueRows.map((row, idx) => {
    const notes = dynamicResidueNotes(row);
    return `
      <tr data-key="${escapeHtml(row.key)}" data-index="${idx}"${residueRowClass(row)}>
        <td>${escapeHtml((row.resid || row.seqid || "") + (row.icode || ""))}</td>
        <td>${escapeHtml(row.chain || "_")}</td>
        <td>${escapeHtml(row.resname || "")}</td>
        <td>${escapeHtml(notes.join(", "))}</td>
      </tr>`;
  }).join("");
  box.innerHTML = `<table><thead><tr><th>Residue</th><th>Chain</th><th>Name</th><th>Note</th></tr></thead><tbody>${body}</tbody></table>`;
  for (const tr of box.querySelectorAll("tbody tr")) {
    tr.addEventListener("click", (ev) => {
      const idx = Number(tr.dataset.index);
      if (!Number.isFinite(idx)) return;
      selectResidueRows(residueRows, idx, ev);
      if (state.sceneKind === "protein") rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
    });
    tr.addEventListener("dblclick", (ev) => {
      ev.preventDefault();
      const key = tr.dataset.key;
      state.selectedResidueKeys = new Set([key]);
      updateResidueRowSelection();
      if (state.sceneKind === "protein") {
        rerenderCurrentScene()
          .then(() => zoomSelection(residueFocusSelectionForKey(key)))
          .catch((err) => setStatus(err.message, "error"));
      }
    });
  }
}

function applyProteinPreviewData(data) {
  state.proteinResidues = data.protein_residues || [];
  state.proteinHighlightSets = data.highlight_sets || {};
  state.proteinBindingLinks = data.metal_binding_links || [];
  state.disulfideCandidates = data.disulfide_candidates || [];
  if (!state.disulfideTokens.length) {
    state.disulfideTokens = state.disulfideCandidates.map((item) => item.token);
    state.disulfideKeys = new Set(state.disulfideCandidates.flatMap((item) => [item.key_a, item.key_b]).filter(Boolean));
  }
  state.selectedResidueKeys = new Set();
  state.lastResidueIndex = -1;
  state.propkaResidueKeys = new Set();
  const bindingCount = state.proteinHighlightSets?.metal_binding?.length || 0;
  $("protein_residue_summary").textContent = `${state.proteinResidues.length} non-water residues; ${bindingCount} metal-binding residue${bindingCount === 1 ? "" : "s"} highlighted.`;
  renderResiduePanel(state.proteinResidues);
  renderDisulfides(state.disulfideCandidates);
}

function syncMetalActionRow(row) {
  const action = row.querySelector(".protein-metal-action")?.value || "keep";
  const targetCell = row.querySelector(".protein-metal-target-cell");
  if (targetCell) targetCell.hidden = action !== "replace";
  state.proteinMetalActions.set(Number(row.dataset.site), {
    action,
    target: row.querySelector(".protein-metal-target")?.value || row.dataset.element || "Fe",
  });
}

function renderProteinMetalBox(data) {
  state.proteinMetals = data.metals || [];
  state.proteinSourceMetals = data.source_metals || state.proteinSourceMetals || state.proteinMetals;
  const rows = state.proteinSourceMetals || [];
  if (!rows.length) {
    $("protein_metal_box").textContent = "No supported metals were detected.";
    renderSystemMetalCharges([]);
    return;
  }
  $("protein_metal_box").innerHTML = `
    <table><thead><tr><th>Site</th><th>Key</th><th>Element</th><th>Action</th><th>Replace with</th></tr></thead><tbody>
      ${rows.map((site) => {
        const saved = state.proteinMetalActions.get(Number(site.site)) || {action: "keep", target: site.element};
        return `
        <tr class="protein-metal-row choice-row" data-key="${escapeHtml(site.key)}" data-site="${site.site}" data-element="${escapeHtml(site.element)}">
          <td>${site.site}</td>
          <td>${escapeHtml(site.key)}</td>
          <td>${escapeHtml(site.element)}</td>
          <td><select class="protein-metal-action">
            <option value="keep" ${saved.action === "keep" ? "selected" : ""}>Keep</option>
            <option value="delete" ${saved.action === "delete" ? "selected" : ""}>Delete</option>
            <option value="replace" ${saved.action === "replace" ? "selected" : ""}>Replace</option>
          </select></td>
          <td class="protein-metal-target-cell"><select class="protein-metal-target">${(state.bootstrap?.supported_metals || []).map((m) => `<option value="${m.element}" ${m.element === saved.target ? "selected" : ""}>${m.element}</option>`).join("")}</select></td>
        </tr>`;
      }).join("")}
    </tbody></table>`;
  for (const row of document.querySelectorAll(".protein-metal-row")) {
    syncMetalActionRow(row);
    row.querySelector(".protein-metal-action")?.addEventListener("change", () => {
      syncMetalActionRow(row);
      runBusy("Reloading protein preview...", () => reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true})).catch((err) => setStatus(err.message, "error"));
    });
    row.querySelector(".protein-metal-target")?.addEventListener("change", () => {
      syncMetalActionRow(row);
      runBusy("Reloading protein preview...", () => reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true})).catch((err) => setStatus(err.message, "error"));
    });
    row.addEventListener("dblclick", () => zoomSelection(residueFocusSelectionForKey(row.dataset.key) || proteinSelectionFromKey(row.dataset.key)));
  }
  renderSystemMetalCharges(state.proteinMetals.map((item) => ({site: item.site, element: item.element, label: item.key})));
}

function renderProteinInsertionDonors(donors = state.proteinInsertionDonors || []) {
  const box = $("protein_insert_donor_box");
  if (!box) return;
  if (!donors.length) {
    box.textContent = "Select residue rows, then load donor candidates.";
    return;
  }
  box.innerHTML = `
    <table><thead><tr><th>Select</th><th>Serial</th><th>Residue</th><th>Atom</th><th>Element</th></tr></thead><tbody>
      ${donors.map((donor) => `
        <tr class="choice-row" data-serial="${donor.atom_serial}">
          <td><input class="protein-insert-donor-check" type="checkbox" value="${escapeHtml(donor.selector)}" checked></td>
          <td>${donor.atom_serial}</td>
          <td>${escapeHtml(`${donor.chain || "_"}:${donor.resid || donor.seqid} ${donor.residue_name}`)}</td>
          <td>${escapeHtml(donor.atom_name)}</td>
          <td>${escapeHtml(donor.element)}</td>
        </tr>`).join("")}
    </tbody></table>`;
}

function renderProteinInsertions() {
  const box = $("protein_insert_box");
  if (!box) return;
  if (!state.proteinInsertions.length) {
    box.textContent = "No inserted metals planned.";
    return;
  }
  box.innerHTML = `
    <table><thead><tr><th>#</th><th>Metal</th><th>Charge</th><th>CN</th><th>Anchors</th><th>Action</th></tr></thead><tbody>
      ${state.proteinInsertions.map((item, idx) => {
      const cn = item.target_coordination_number == null ? "manual" : `CN ${item.target_coordination_number}`;
      const anchors = (item.anchors || []).join(", ");
      return `<tr>
        <td>${idx + 1}</td>
        <td>${escapeHtml(item.element)}</td>
        <td>+${escapeHtml(item.charge || "?")}</td>
        <td>${escapeHtml(cn)}</td>
        <td>${escapeHtml(anchors)}</td>
        <td><button class="protein-remove-insertion" data-index="${idx}">Remove</button></td>
      </tr>`;
    }).join("")}
    </tbody></table>`;
  for (const node of document.querySelectorAll(".protein-remove-insertion")) {
    node.addEventListener("click", () => {
      removeProteinInsertion(Number(node.dataset.index)).catch((err) => setStatus(err.message, "error"));
    });
  }
}

function currentProteinInsertionFromDonors() {
  const donorAnchors = [...document.querySelectorAll(".protein-insert-donor-check")]
    .filter((node) => node.checked)
    .map((node) => String(node.value));
  if (!donorAnchors.length) throw new Error("Choose at least one donor atom before previewing an inserted metal.");
  const element = $("protein_insert_element").value || "Fe";
  const charge = Number($("protein_insert_charge").value || defaultMetalCharge(element));
  const cnRaw = $("protein_insert_cn").value;
  return {
    element,
    charge,
    anchor_mode: "donor_atoms",
    anchors: donorAnchors,
    target_coordination_number: cnRaw === MANUAL_CN_VALUE ? null : Number(cnRaw || defaultCoordinationNumber(element, charge)),
    label: `gui_${element}_${state.proteinInsertions.length + 1}`,
  };
}

async function loadProteinInsertionDonors() {
  if (!state.proteinLoaded) throw new Error("Load a protein preview before selecting insertion anchors.");
  const residueKeys = Array.from(state.selectedResidueKeys || []);
  if (!residueKeys.length) throw new Error("Select one or more residue rows first.");
  const payload = collectPayload();
  payload.residue_keys = residueKeys;
  const data = await api("/api/protein/metal-donor-candidates", payload);
  state.proteinInsertionDonors = data.donor_candidates || [];
  renderProteinInsertionDonors();
  setStatus(state.proteinInsertionDonors.length ? "Donor candidates loaded." : "No donor candidates found.", state.proteinInsertionDonors.length ? "ok" : "warn");
}

async function previewProteinInsertedMetal() {
  const insertion = currentProteinInsertionFromDonors();
  state.proteinInsertions.push(insertion);
  renderProteinInsertions();
  await reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true});
}

async function removeProteinInsertion(index) {
  if (index < 0 || index >= state.proteinInsertions.length) return;
  state.proteinInsertions.splice(index, 1);
  renderProteinInsertions();
  if (state.proteinLoaded) {
    await reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true});
  }
  setStatus("Inserted protein metal removed from the preview plan.", "ok");
}

async function clearProteinInsertions() {
  state.proteinInsertions = [];
  state.proteinInsertionDonors = [];
  renderProteinInsertionDonors([]);
  renderProteinInsertions();
  if (state.proteinLoaded) {
    await reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true});
  }
}

function zoomSelection(selection) {
  if (!selection || !state.stage) return;
  const comp = state.structureComponent;
  if (!comp?.autoView) {
    resetCamera();
    return;
  }
  try {
    comp.autoView(selection, 350);
  } catch (_err) {
    resetCamera();
    return;
  }
  setTimeout(() => {
    if (!captureOrientation()) {
      resetCamera();
      return;
    }
    requestViewerRender();
  }, 380);
}

function syncPropkaSelections() {
  state.propkaResidueKeys = new Set([...document.querySelectorAll(".propka-check")]
    .filter((node) => node.checked && !node.disabled)
    .map((node) => node.dataset.key)
    .filter(Boolean));
  $("propka_state").textContent = state.propkaResidueKeys.size
    ? `Current PropKa choice: ${state.propkaResidueKeys.size} residue-state row${state.propkaResidueKeys.size === 1 ? "" : "s"} checked.`
    : "Current PropKa choice: no residue-state rows checked.";
  renderResiduePanel(state.proteinResidues || []);
  if (state.sceneKind === "protein" && state.currentPdb) {
    rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
  }
}

function renderPropka(candidates) {
  state.propkaChanges = candidates || [];
  if (!state.propkaChanges.length) {
    $("propka_box").textContent = "No PropKa or direct metal-coordination changes were suggested.";
    state.propkaResidueKeys = new Set();
    renderResiduePanel(state.proteinResidues || []);
    return;
  }
  $("propka_box").innerHTML = `
    <table><thead><tr><th>Use</th><th>Residue</th><th>Change</th><th>Metal</th><th>Reason</th></tr></thead><tbody>
      ${state.propkaChanges.map((item, idx) => {
        const key = proteinResidueKeyForCandidate(item);
        return `
        <tr class="choice-row propka-row ${item.metal_near ? "note-metal-binding" : ""}" data-idx="${idx}" data-key="${escapeHtml(key)}" data-chain="${escapeHtml(item.chain)}" data-seqid="${escapeHtml(item.seqid)}">
          <td><input class="propka-check" type="checkbox" data-idx="${idx}" data-key="${escapeHtml(key)}" ${item.selectable ? "checked" : "disabled"}></td>
          <td>${escapeHtml(item.chain || "_")}:${escapeHtml(item.seqid)}</td>
          <td>${escapeHtml(item.original_residue_name)} -> ${escapeHtml(item.target_residue_name)}</td>
          <td>${item.metal_near ? "Yes" : "No"}</td>
          <td>${escapeHtml(item.reason)}</td>
        </tr>`;
      }).join("")}
    </tbody></table>`;
  for (const node of document.querySelectorAll(".propka-check")) {
    node.addEventListener("change", syncPropkaSelections);
  }
  for (const row of document.querySelectorAll(".propka-row")) {
    row.addEventListener("click", (ev) => {
      if (ev.target.tagName === "INPUT") return;
      const key = row.dataset.key;
      if (!key) return;
      state.selectedResidueKeys = new Set([key]);
      renderResiduePanel(state.proteinResidues || []);
      if (state.sceneKind === "protein") rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
    });
    row.addEventListener("dblclick", () => {
      const key = row.dataset.key;
      if (key) {
        state.selectedResidueKeys = new Set([key]);
        renderResiduePanel(state.proteinResidues || []);
        rerenderCurrentScene()
          .then(() => zoomSelection(residueFocusSelectionForKey(key)))
          .catch((err) => setStatus(err.message, "error"));
      } else {
        const chain = row.dataset.chain || "";
        const seqid = row.dataset.seqid || "";
        zoomSelection(chain ? `:${chain} and ${seqid}` : seqid);
      }
    });
  }
  syncPropkaSelections();
}

function syncDisulfideSelections() {
  state.disulfideTokens = [...document.querySelectorAll(".disulfide-check")]
    .filter((node) => node.checked)
    .map((node) => node.dataset.token)
    .filter(Boolean);
  state.disulfideKeys = new Set();
  for (const node of document.querySelectorAll(".disulfide-check")) {
    if (!node.checked) continue;
    if (node.dataset.keyA) state.disulfideKeys.add(node.dataset.keyA);
    if (node.dataset.keyB) state.disulfideKeys.add(node.dataset.keyB);
  }
  $("disulfide_state").textContent = state.disulfideTokens.length
    ? `Disulfide bonds: ${state.disulfideTokens.length} candidate pair${state.disulfideTokens.length === 1 ? "" : "s"} checked.`
    : "Disulfide bonds: no CYS-CYS candidate pairs checked.";
  renderResiduePanel(state.proteinResidues || []);
  if (state.sceneKind === "protein" && state.currentPdb) rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
}

function renderDisulfides(items) {
  const candidates = items || [];
  if (!candidates.length) {
    $("disulfide_box").textContent = "No CYS-CYS candidates detected.";
    $("disulfide_state").textContent = "Disulfide bonds: no CYS-CYS candidate pairs detected.";
    state.disulfideTokens = [];
    state.disulfideKeys = new Set();
    return;
  }
  $("disulfide_box").innerHTML = `
    <table><thead><tr><th>Use</th><th>Cys A</th><th>Cys B</th><th>Distance A</th></tr></thead><tbody>
      ${candidates.map((item) => `
        <tr class="choice-row disulfide-row" data-key-a="${escapeHtml(item.key_a)}" data-key-b="${escapeHtml(item.key_b)}">
          <td><input class="disulfide-check" type="checkbox" data-token="${escapeHtml(item.token)}" data-key-a="${escapeHtml(item.key_a)}" data-key-b="${escapeHtml(item.key_b)}" ${state.disulfideTokens.includes(item.token) ? "checked" : ""}></td>
          <td>${escapeHtml(item.a)}</td>
          <td>${escapeHtml(item.b)}</td>
          <td>${Number(item.distance_angstrom).toFixed(2)}</td>
        </tr>`).join("")}
    </tbody></table>`;
  for (const node of document.querySelectorAll(".disulfide-check")) node.addEventListener("change", syncDisulfideSelections);
  for (const row of document.querySelectorAll(".disulfide-row")) {
    row.addEventListener("click", (ev) => {
      if (ev.target.tagName === "INPUT") return;
      state.selectedResidueKeys = new Set([row.dataset.keyA, row.dataset.keyB].filter(Boolean));
      renderResiduePanel(state.proteinResidues || []);
      rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
    });
    row.addEventListener("dblclick", () => {
      const keys = [row.dataset.keyA, row.dataset.keyB].filter(Boolean);
      state.selectedResidueKeys = new Set(keys);
      renderResiduePanel(state.proteinResidues || []);
      rerenderCurrentScene()
        .then(() => zoomSelection(residueFocusSelectionForKey(keys[0])))
        .catch((err) => setStatus(err.message, "error"));
    });
  }
  syncDisulfideSelections();
}

function renderRespCandidates(candidates) {
  if (!candidates.length) {
    $("resp_candidates").textContent = "No RESP folders found.";
    return;
  }
  $("resp_candidates").innerHTML = `
    <table><thead><tr><th>Use</th><th>Status</th><th>Residue</th><th>Charge</th><th>Folder</th></tr></thead><tbody>
      ${candidates.map((item) => `
        <tr class="choice-row ${item.job_dir === state.selectedRespJobDir ? "selected-row" : ""}" data-path="${escapeHtml(item.job_dir)}">
          <td><button class="use-resp" data-path="${escapeHtml(item.job_dir)}">Use</button></td>
          <td>${item.ready_to_continue ? "Ready" : "Pending"}</td>
          <td>${escapeHtml(item.residue_name || "LIG")}</td>
          <td>${escapeHtml(item.net_charge ?? "-")}</td>
          <td>${escapeHtml(item.job_dir)}</td>
        </tr>`).join("")}
    </tbody></table>`;
  for (const btn of document.querySelectorAll(".use-resp")) {
    btn.addEventListener("click", () => {
      state.selectedRespJobDir = btn.dataset.path || "";
      $("resp_job_dir").value = state.selectedRespJobDir;
      setStatus(`Selected RESP folder: ${state.selectedRespJobDir}`, "ok");
    });
  }
}

function snapshotDesComponentInputs() {
  const ratios = new Map(
    Array.from(document.querySelectorAll(".des-ratio")).map((node) => [node.dataset.key || "", node.value])
  );
  return new Map(
    Array.from(document.querySelectorAll(".des-component-check")).map((node) => [
      node.dataset.key || "",
      {checked: node.checked, ratio: ratios.get(node.dataset.key || "") || "1"},
    ])
  );
}

function renderDesComponents(values = new Map()) {
  $("des_components").innerHTML = `
    <table><thead><tr><th>Use</th><th>Component</th><th>Ratio</th><th>Description</th></tr></thead><tbody>
      ${state.desComponents.map((item) => `
        <tr>
          <td><input class="des-component-check" data-key="${escapeHtml(item.key)}" type="checkbox"></td>
          <td>${escapeHtml(item.label)}</td>
          <td><input class="des-ratio" data-key="${escapeHtml(item.key)}" type="number" step="1" value="1"></td>
          <td>${escapeHtml(item.description)}</td>
        </tr>`).join("")}
    </tbody></table>`;
  for (const check of document.querySelectorAll(".des-component-check")) {
    const saved = values.get(check.dataset.key || "");
    if (saved) check.checked = Boolean(saved.checked);
  }
  for (const ratio of document.querySelectorAll(".des-ratio")) {
    const saved = values.get(ratio.dataset.key || "");
    if (saved) ratio.value = saved.ratio;
  }
}

function renderLibraryComponents() {
  const container = $("library_components");
  if (!state.libraryComponents.length) {
    container.innerHTML = '<div class="library-empty-state"><span>No DES library components are available.</span></div>';
    $("library_remove").disabled = true;
    return;
  }
  container.innerHTML = state.libraryComponents.map((item, index) => {
    const expanded = state.expandedLibraryComponents.has(item.key);
    const selected = item.removable && state.selectedLibraryComponent === item.key;
    return `
    <div class="library-component-wrap">
      <button class="library-component ${selected ? "selected" : ""}" data-key="${escapeHtml(item.key)}" type="button"
              aria-expanded="${expanded ? "true" : "false"}" aria-controls="library-component-files-${index}">
        <span class="library-caret" aria-hidden="true">&#9656;</span>
        <strong>${escapeHtml(item.label)}</strong>
        <span>${escapeHtml((item.residues || []).join(", "))} &middot; ${item.custom ? "User-added" : "Built-in (protected)"}</span>
      </button>
      <div id="library-component-files-${index}" class="library-component-files" ${expanded ? "" : "hidden"}>
        ${(item.files || []).map((file) => `
          <button class="library-file-row ${state.selectedLibraryFile === file.path ? "selected" : ""}" data-path="${escapeHtml(file.path)}" type="button">
            <strong>${escapeHtml(file.name)}</strong><span>${escapeHtml(file.kind)}</span>
          </button>`).join("")}
      </div>
    </div>`;
  }).join("");
  for (const button of container.querySelectorAll(".library-component")) {
    button.addEventListener("click", () => {
      const key = button.dataset.key || "";
      if (state.expandedLibraryComponents.has(key)) {
        state.expandedLibraryComponents.delete(key);
      } else {
        state.expandedLibraryComponents.add(key);
      }
      const component = state.libraryComponents.find((item) => item.key === key);
      if (component?.removable) state.selectedLibraryComponent = key;
      renderLibraryComponents();
    });
  }
  for (const button of container.querySelectorAll(".library-file-row")) {
    button.addEventListener("click", () => runBusy("Loading Amber library file...", () => loadLibraryFile(button.dataset.path || ""))
      .catch((err) => setStatus(err.message, "error")));
  }
  const selected = state.libraryComponents.find((item) => item.key === state.selectedLibraryComponent);
  $("library_remove").disabled = !selected?.removable;
}

function refreshDesComponentsFromLibrary() {
  const values = snapshotDesComponentInputs();
  state.desComponents = state.libraryComponents.map((item) => ({
    key: item.key,
    label: item.label,
    description: item.description,
  }));
  renderDesComponents(values);
}

function applyLibraryComponents(components) {
  state.libraryComponents = components || [];
  const keys = new Set(state.libraryComponents.map((item) => item.key));
  state.expandedLibraryComponents = new Set(
    Array.from(state.expandedLibraryComponents).filter((key) => keys.has(key))
  );
  if (!keys.has(state.selectedLibraryComponent)) state.selectedLibraryComponent = "";
  refreshDesComponentsFromLibrary();
  renderLibraryComponents();
}

function libraryStatusText(status) {
  return {
    new: "New residue",
    already_registered: "Identical data already registered",
    different_values: "Same residue, different parameter values",
  }[status] || status;
}

function renderLibraryCandidates() {
  const container = $("library_candidates");
  if (!state.libraryCandidates.length) {
    container.className = "library-empty-state";
    container.innerHTML = '<span>No matching .lib/.off + .frcmod pair was found in this directory.</span>';
    state.selectedLibraryCandidate = -1;
    $("library_add").disabled = true;
    return;
  }
  container.className = "library-candidate-list";
  container.innerHTML = state.libraryCandidates.map((item, index) => `
    <div class="library-candidate-wrap">
      <button class="library-candidate ${state.selectedLibraryCandidate === index ? "selected" : ""}" data-index="${index}" type="button">
        <strong>${escapeHtml(item.residue_name)}</strong>
        <span class="library-status-${escapeHtml(item.status)}">${escapeHtml(libraryStatusText(item.status))}${item.matched_label ? ` · ${escapeHtml(item.matched_label)}` : ""}</span>
      </button>
      <div class="library-candidate-files">
        ${(item.files || []).map((file) => `
          <button class="library-file-row ${state.selectedLibraryFile === file.path ? "selected" : ""}" data-path="${escapeHtml(file.path)}" type="button">
            <strong>${escapeHtml(file.name)}</strong><span>${escapeHtml(file.kind)}</span>
          </button>`).join("")}
      </div>
    </div>`).join("");
  for (const button of container.querySelectorAll(".library-candidate")) {
    button.addEventListener("click", () => {
      state.selectedLibraryCandidate = Number.parseInt(button.dataset.index || "-1", 10);
      renderLibraryCandidates();
      $("library_add").disabled = state.selectedLibraryCandidate < 0;
    });
  }
  for (const button of container.querySelectorAll(".library-file-row")) {
    button.addEventListener("click", () => runBusy("Loading Amber library file...", () => loadLibraryFile(button.dataset.path || ""))
      .catch((err) => setStatus(err.message, "error")));
  }
  $("library_add").disabled = state.selectedLibraryCandidate < 0;
}

async function loadLibraryComponents() {
  const response = await fetch("/api/des-library/components");
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "Could not load DES library components.");
  applyLibraryComponents(data.components);
}

async function scanLibraryDirectory() {
  const path = $("library_search_path").value.trim();
  if (!path) throw new Error("Enter a directory containing matching .lib/.off and .frcmod files.");
  const data = await api("/api/des-library/scan", {path});
  applyLibraryScanResult(data);
}

function applyLibraryScanResult(data) {
  state.libraryCandidates = data.candidates || [];
  state.selectedLibraryCandidate = state.libraryCandidates.length === 1 ? 0 : -1;
  renderLibraryCandidates();
  setStatus(`Found ${state.libraryCandidates.length} Amber library bundle(s) under ${data.path}.`, state.libraryCandidates.length ? "ok" : "warn");
}

async function uploadLibrarySelection(fileList, useRelativePaths) {
  const supported = Array.from(fileList || []).filter((file) => /\.(?:lib|off|frcmod)$/i.test(file.name));
  if (!supported.length) {
    throw new Error("No Amber .lib/.off or .frcmod files were selected.");
  }
  if (!supported.some((file) => /\.(?:lib|off)$/i.test(file.name)) || !supported.some((file) => /\.frcmod$/i.test(file.name))) {
    throw new Error("Select both a .lib/.off library file and its matching .frcmod parameter file.");
  }
  const form = new FormData();
  for (const file of supported) {
    form.append("files", file, file.name);
    form.append("relative_paths", useRelativePaths ? (file.webkitRelativePath || file.name) : file.name);
  }
  const response = await fetch("/api/des-library/upload", {
    method: "POST",
    headers: {"X-SIMPLE-Token": state.apiToken},
    body: form,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "Could not open the selected Amber library files.");
  $("library_search_path").value = data.path || "";
  applyLibraryScanResult(data);
}

function openLibraryDirectoryPicker() {
  $("library_directory_picker").click();
}

async function loadLibraryFile(path) {
  const data = await api("/api/des-library/file", {path});
  state.selectedLibraryFile = data.path;
  state.selectedLibraryFileEditable = data.editable !== false;
  $("library_editor_name").textContent = state.selectedLibraryFileEditable
    ? data.path
    : `${data.path} - Built-in component file (read-only)`;
  $("library_editor").value = data.content || "";
  $("library_editor").disabled = false;
  $("library_editor").readOnly = !state.selectedLibraryFileEditable;
  $("library_save_file").disabled = !state.selectedLibraryFileEditable;
  renderLibraryComponents();
  renderLibraryCandidates();
  setStatus(
    state.selectedLibraryFileEditable ? `Loaded ${data.name}.` : `Loaded protected file ${data.name} in read-only mode.`,
    "ok"
  );
}

async function saveLibraryFile() {
  if (!state.selectedLibraryFile) throw new Error("Select a library file before saving.");
  if (!state.selectedLibraryFileEditable) throw new Error("Built-in DES component files are read-only.");
  const data = await api("/api/des-library/file/save", {
    path: state.selectedLibraryFile,
    content: $("library_editor").value,
  });
  if ($("library_search_path").value.trim()) await scanLibraryDirectory();
  setStatus(data.message, "ok");
}

function showLibraryDialog({title, message, actions, showFields = false, defaultKey = "", defaultLabel = ""}) {
  return new Promise((resolve) => {
    const dialog = $("library_dialog");
    $("library_dialog_title").textContent = title;
    $("library_dialog_message").textContent = message;
    $("library_dialog_fields").hidden = !showFields;
    $("library_component_key").value = defaultKey;
    $("library_component_label").value = defaultLabel;
    const actionBox = $("library_dialog_actions");
    actionBox.replaceChildren();
    for (const action of actions) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = action.label;
      if (action.className) button.className = action.className;
      button.addEventListener("click", () => {
        dialog.close();
        resolve(action.value);
      });
      actionBox.appendChild(button);
    }
    dialog.addEventListener("cancel", () => resolve(null), {once: true});
    dialog.showModal();
  });
}

async function addSelectedLibraryCandidate() {
  const candidate = state.libraryCandidates[state.selectedLibraryCandidate];
  if (!candidate) throw new Error("Select a detected library bundle first.");
  const defaults = {
    showFields: true,
    defaultKey: `custom_${String(candidate.residue_name || "residue").toLowerCase()}`,
    defaultLabel: candidate.residue_name || "Custom component",
  };
  let action = null;
  if (candidate.status === "already_registered") {
    action = await showLibraryDialog({
      ...defaults,
      title: "Matching library data already exists",
      message: `${candidate.residue_name} is already registered as ${candidate.matched_label || candidate.matched_component}. Add another component variant anyway?`,
      actions: [{label: "Yes", value: "register_new"}, {label: "No", value: null}],
    });
  } else if (candidate.status === "different_values") {
    const matched = state.libraryComponents.find((item) => item.key === candidate.matched_component);
    const canOverwrite = matched?.editable === true;
    action = await showLibraryDialog({
      ...defaults,
      title: "Parameter values differ",
      message: canOverwrite
        ? `${candidate.residue_name} already exists as ${candidate.matched_label || candidate.matched_component}, but the selected files contain different values.`
        : `${candidate.residue_name} matches a built-in component with different values. Built-in data is protected; add it as a new custom component instead.`,
      actions: [
        ...(canOverwrite ? [{label: "Overwrite Existing", value: "overwrite"}] : []),
        {label: "Add as New", value: "register_new"},
        {label: "No", value: null},
      ],
    });
  } else {
    action = await showLibraryDialog({
      ...defaults,
      title: "Add DES library component",
      message: `Register ${candidate.residue_name} with the selected .lib/.off and .frcmod files?`,
      actions: [{label: "Yes", value: "register_new"}, {label: "No", value: null}],
    });
  }
  if (!action) return;
  const data = await api("/api/des-library/register", {
    lib_path: candidate.lib_path,
    frcmod_path: candidate.frcmod_path,
    action,
    component_key: $("library_component_key").value.trim(),
    label: $("library_component_label").value.trim(),
  });
  applyLibraryComponents(data.components);
  await scanLibraryDirectory();
  setStatus(data.message, "ok");
}

async function removeSelectedLibraryComponent() {
  const component = state.libraryComponents.find((item) => item.key === state.selectedLibraryComponent);
  if (!component) throw new Error("Select a library component first.");
  if (!component.removable) throw new Error("Built-in DES components are protected and cannot be removed.");
  const action = await showLibraryDialog({
    title: "Remove library component",
    message: `Remove ${component.label} and its managed Amber files from the custom DES library?`,
    actions: [{label: "Yes, Remove", value: "remove", className: "danger"}, {label: "No", value: null}],
  });
  if (action !== "remove") return;
  const data = await api("/api/des-library/remove", {component_key: component.key});
  applyLibraryComponents(data.components);
  setStatus(data.message, "ok");
}

function applyDesRecommended(item) {
  for (const node of document.querySelectorAll(".des-component-check")) {
    node.checked = (item.components || []).includes(node.dataset.key);
  }
  for (const ratio of document.querySelectorAll(".des-ratio")) {
    const idx = (item.components || []).indexOf(ratio.dataset.key);
    ratio.value = idx >= 0 ? item.ratios[idx] : "1";
  }
}

function updateDesMetalCharges() {
  const element = $("des_metal_element").value;
  replaceOptions($("des_metal_charge"), chargeOptions(element).map((charge) => [`${charge}`, `+${charge}`]));
}

function parseDesCoordinates(raw) {
  const lines = String(raw || "").split(/\n+/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length) return null;
  return lines.map((line) => {
    const values = line.split(/[\s,]+/).map((token) => Number.parseFloat(token));
    if (values.length !== 3 || values.some((value) => Number.isNaN(value))) {
      throw new Error("DES metal coordinates must use one x y z triplet per line.");
    }
    return values;
  });
}

function renderDesMetalSites() {
  const metals = state.bootstrap?.supported_metals || [];
  const metalOptions = metals.map((metal) => `<option value="${escapeHtml(metal.element)}">${escapeHtml(metal.element)}</option>`).join("");
  $("des_metal_sites").innerHTML = `
    <table><thead><tr><th>Metal</th><th>Charge</th><th>Count</th><th>XYZ override</th><th></th></tr></thead><tbody>
      ${state.desMetalSites.map((site, index) => `
        <tr data-index="${index}">
          <td><select class="des-extra-metal-element">${metalOptions}</select></td>
          <td><select class="des-extra-metal-charge"></select></td>
          <td><input class="des-extra-metal-count" type="number" min="1" step="1" value="${site.count || 1}"></td>
          <td><textarea class="des-extra-metal-coords" rows="2" placeholder="optional: one x y z per line">${escapeHtml(site.coordinatesText || "")}</textarea></td>
          <td><button class="des-extra-metal-remove" type="button">Remove</button></td>
        </tr>`).join("")}
    </tbody></table>`;
  for (const row of document.querySelectorAll("#des_metal_sites tr[data-index]")) {
    const index = Number.parseInt(row.dataset.index || "0", 10);
    const site = state.desMetalSites[index];
    const elementSelect = row.querySelector(".des-extra-metal-element");
    const chargeSelect = row.querySelector(".des-extra-metal-charge");
    elementSelect.value = site.element || "Fe";
    const refreshCharges = () => {
      replaceOptions(chargeSelect, chargeOptions(elementSelect.value).map((charge) => [`${charge}`, `+${charge}`]));
      chargeSelect.value = `${site.charge || defaultMetalCharge(elementSelect.value)}`;
    };
    refreshCharges();
    elementSelect.addEventListener("change", () => {
      site.element = elementSelect.value;
      site.charge = defaultMetalCharge(site.element);
      refreshCharges();
    });
    chargeSelect.addEventListener("change", () => site.charge = Number.parseInt(chargeSelect.value || "2", 10));
    row.querySelector(".des-extra-metal-count").addEventListener("input", (ev) => site.count = Number.parseInt(ev.target.value || "1", 10));
    row.querySelector(".des-extra-metal-coords").addEventListener("input", (ev) => site.coordinatesText = ev.target.value);
    row.querySelector(".des-extra-metal-remove").addEventListener("click", () => {
      state.desMetalSites.splice(index, 1);
      renderDesMetalSites();
    });
  }
}

function addDesMetalSite() {
  const element = $("des_metal_element").value || "Fe";
  state.desMetalSites.push({element, charge: defaultMetalCharge(element), count: 1, coordinatesText: ""});
  renderDesMetalSites();
}

function updateInsertionChargeControls(prefix) {
  const elementSelect = $(`${prefix}_element`);
  const chargeSelect = $(`${prefix}_charge`);
  const cnSelect = $(`${prefix}_cn`);
  if (!elementSelect || !chargeSelect || !cnSelect) return;
  const element = elementSelect.value || "Fe";
  replaceOptions(chargeSelect, chargeOptions(element).map((charge) => [`${charge}`, `+${charge}`]));
  const charge = Number(chargeSelect.value || defaultMetalCharge(element));
  const cnValues = coordinationOptions(element, charge);
  replaceOptions(cnSelect, [
    ...cnValues.map((value) => [`${value}`, `${value}`]),
    [MANUAL_CN_VALUE, "Manual selection"],
  ], String(defaultCoordinationNumber(element, charge)));
}

function currentQmSettings() {
  const net = Number.parseInt($("net_charge").value || "0", 10);
  const mult = Number.parseInt($("multiplicity").value || "1", 10);
  return {
    net_charge: net,
    multiplicity: mult,
    geometry: {
      mode: $("qm_geometry").value,
      dft_optimization: {
        functional: $("qm_functional").value,
        basis: $("qm_basis").value,
      },
    },
    resp: {
      same_as_dft_optimization: $("qm_resp_same").checked,
      functional: $("qm_resp_functional").value,
      basis: $("qm_resp_basis").value,
    },
    resources: {
      memory_mb: Number.parseInt($("qm_memory").value || "2000", 10),
      grid: $("qm_grid").value,
      maxiter: Number.parseInt($("qm_maxiter").value || "200", 10),
    },
  };
}

function collectMetalCharges() {
  if ($("workflow_type")?.value === "deep_eutectic") return [];
  const out = [];
  for (const row of document.querySelectorAll(".metal-charge-row")) {
    const select = row.querySelector(".metal-charge-select");
    out.push({
      site: Number(row.dataset.site),
      charge: Number(select.value),
      element: String(row.dataset.element || "").trim(),
    });
  }
  return out;
}

function collectProteinPrepare() {
  const deletions = [];
  const replacements = [];
  for (const row of document.querySelectorAll(".protein-metal-row")) {
    const site = Number(row.dataset.site);
    const action = row.querySelector(".protein-metal-action").value;
    if (action === "delete") deletions.push(site);
    if (action === "replace") replacements.push({site, target: row.querySelector(".protein-metal-target").value});
  }
  return {
    remove_waters: $("remove_waters").checked,
    remove_other_hetero: $("remove_hetero").checked,
    repair_missing_loops: state.lastProteinMissingLoopAction === "repair",
    kept_ligands: $("kept_ligands").value.split(",").map((x) => x.trim()).filter(Boolean),
    metal_deletions: deletions,
    metal_replacements: replacements,
    metal_insertions: state.proteinInsertions || [],
  };
}

function syncProteinInputFields() {
  const text = $("protein_input")?.value.trim() || "";
  const looksPdbId = /^[A-Za-z0-9]{4}$/.test(text);
  $("protein_input_mode").value = looksPdbId ? "pdb_id" : "path";
  $("protein_pdb_id").value = looksPdbId ? text.toUpperCase() : "";
  $("protein_path").value = looksPdbId ? "" : text;
}

function collectSalt() {
  const kind = $("salt_kind").value;
  const mode = $("salt_mode").value;
  let value = 0;
  if (mode === "count" || mode === "concentration") {
    value = Number.parseFloat($("salt_amount").value || "0");
    if (mode === "concentration" && $("salt_unit").value === "mM") value = value / 1000.0;
  }
  return {
    kind,
    mode,
    value,
    neutralization_ion: $("neutralization_ion").value || "auto",
  };
}

function collectDesConfig() {
  const components = [];
  const ratios = [];
  for (const check of document.querySelectorAll(".des-component-check")) {
    if (!check.checked) continue;
    components.push(check.dataset.key);
    const ratio = document.querySelector(`.des-ratio[data-key="${check.dataset.key}"]`);
    ratios.push(Number.parseInt(ratio?.value || "1", 10));
  }
  const sizeMode = $("des_size_mode").value;
  const metalSites = state.desMetalSites.map((site) => ({
    element: site.element || "Fe",
    charge: Number.parseInt(site.charge || defaultMetalCharge(site.element || "Fe"), 10),
    count: Number.parseInt(site.count || "1", 10),
    coordinates: parseDesCoordinates(site.coordinatesText || ""),
  }));
  return {
    components,
    ratios,
    mixing_mode: $("des_build_mode").value || "random_mix",
    size_mode: sizeMode,
    ratio_units: sizeMode === "ratio_units" ? Number.parseInt($("des_ratio_units").value || "100", 10) : null,
    box_length_angstrom: sizeMode === "box_length" ? Number.parseFloat($("des_box_length").value || "80") : null,
    spacing_angstrom: Number.parseFloat($("des_spacing").value || "1.3"),
    packmol_tolerance_angstrom: Number.parseFloat($("des_packmol_tolerance").value || "2.0"),
    target_density_g_ml: Number.parseFloat($("des_target_density").value || "0.40"),
    apply_1264: $("des_apply_1264").checked,
    c4_parameter_set: $("des_c4_parameter_set").value || "opc_duvail",
    central_metal_enabled: $("des_central_metal").checked,
    central_metal_element: $("des_metal_element").value,
    central_metal_charge: Number.parseInt($("des_metal_charge").value || "2", 10),
    metal_sites: metalSites,
    metal_spacing_angstrom: Number.parseFloat($("des_metal_spacing").value || "8.0"),
  };
}

function collectPayload() {
  syncSystemC4ParameterSet(false, false);
  const workflow = $("workflow_type").value;
  const payload = {
    workflow_type: workflow,
    job_name: $("job_name").value.trim() || "simple_gui",
    output_root: $("output_root").value.trim() || "gui_outputs",
    system: {
      protein_ff: $("protein_ff").value || "ff19SB",
      ligand_ff: $("ligand_ff").value || "gaff2",
      apply_1264: $("apply_1264_metals").checked,
      c4_parameter_set: $("system_c4_parameter_set").value || "opc_duvail",
      water_model: $("water_model").value || "opc",
      box_shape: $("box_shape").value || "oct",
      buffer_angstrom: Number.parseFloat($("buffer_angstrom").value || "10"),
      metal_charges: collectMetalCharges(),
      salt: collectSalt(),
    },
    md: {
      protocol: $("md_protocol").value,
      temperature_k: Number.parseFloat($("temperature").value || "300"),
      pressure_bar: Number.parseFloat($("pressure").value || "1"),
      production_time_ns: Number.parseFloat($("production_ns").value || "10"),
      des_mixing_enabled: workflow === "deep_eutectic" ? $("des_mixing_enabled").checked : false,
      focused_restraint_mask: $("focused_enabled")?.checked ? $("focused_mask").value.trim() : "",
      focused_restraint_weight: $("focused_enabled")?.checked && $("focused_mask").value.trim() ? ($("focused_weight").value.trim() || "5.0") : null,
      stage_overrides: collectMdStageOverrides(),
    },
    slurm: {
      profile: $("slurm_profile").value,
      ntasks: Number.parseInt($("slurm_ntasks").value || "8", 10),
      gpus: Number.parseInt($("slurm_gpus").value || "1", 10),
      walltime: $("slurm_walltime").value.trim() || "24:00:00",
      binary_override: null,
    },
  };
  if (workflow === "metallophore") {
    payload.metallophore = {
      mode: $("met_mode").value,
      residue_name: $("met_residue").value.trim() || "LIG",
      input_path: $("met_input_path").value.trim(),
      smiles_text: $("met_smiles").value.trim(),
      resp_job_dir: $("resp_job_dir").value.trim(),
      edited_atoms: state.metAtoms || [],
      edited_bonds: state.metBonds || [],
      group_constraints: state.metGroupConstraints,
      metal_coordination: collectSelectedMetalCoordination(),
      auto_group_mode: $("resp_group_mode")?.value || "hydrogen_and_symmetry",
      auto_group_graph_method: $("resp_group_graph_method")?.value || "connectivity",
      qm_settings: currentQmSettings(),
      ligands: {
        mode: $("ligand_ff").value || "gaff2",
        charge_method: $("charge_method").value,
        net_charge: Number.parseInt($("net_charge").value || "0", 10),
        multiplicity: Number.parseInt($("multiplicity").value || "1", 10),
      },
    };
  } else if (workflow === "metalloprotein") {
    syncProteinInputFields();
    const siteRespJobDir = $("protein_site_resp_job_dir")?.value.trim() || "";
    payload.protein = {
      input_mode: $("protein_input_mode").value,
      input_value: $("protein_input").value.trim(),
      path: $("protein_path").value.trim(),
      pdb_id: $("protein_pdb_id").value.trim(),
      prepare: collectProteinPrepare(),
      protonation: {
        enabled: true,
        ph: Number.parseFloat($("protein_ph").value || "7.0"),
        selected_changes: Array.from(document.querySelectorAll(".propka-check"))
          .filter((node) => node.checked && !node.disabled)
          .map((node) => state.propkaChanges[Number(node.dataset.idx)]?.change)
          .filter(Boolean),
      },
      site_resp: {
        mode: siteRespJobDir ? "resp" : "standard_ff",
        scope: $("protein_site_resp_scope")?.value || "sidechain",
        apply_mode: state.proteinSiteRespApproved ? "apply_existing" : "detect",
        default_multiplicity: Number.parseInt($("protein_site_resp_multiplicity")?.value || "1", 10),
        multiplicity_confirmed: Boolean(siteRespJobDir),
        search_root: $("protein_site_resp_search_root")?.value.trim() || ".",
        job_dir: siteRespJobDir,
        clusters: state.proteinSiteRespClusters || [],
      },
      ligands: {mode: "manual"},
    };
  } else {
    payload.des = collectDesConfig();
  }
  return payload;
}

function resetMetallophoreTables() {
  state.metAtoms = [];
  state.metBonds = [];
  state.metMetals = [];
  state.metDonors = [];
  state.selectedDonors = new Set();
  state.selectedAtomIndices = new Set();
  state.selectedRespGroupId = null;
  state.selectedMetalAtom = null;
  state.metMetalElements = new Map();
  state.metGroupConstraints = null;
  state.metCoordination = null;
  state.metOriginal = null;
  state.metInsertedMetalIndices = new Set();
  $("metals_box").textContent = "Load a metallophore to detect supported 12-6-4 metals.";
  $("donor_box").textContent = "Donor candidates appear here after loading.";
  if ($("resp_group_box")) $("resp_group_box").textContent = "Load a metallophore to review RESP symmetry groups.";
  renderRespGroupControls();
  renderSystemMetalCharges([]);
  refreshRespChargeHint();
}

function syncViewerSidePanels() {
  const workflow = $("workflow_type").value;
  const showResp = workflow === "metallophore" && $("met_mode").value === "resp_input";
  const showResidues = workflow === "metalloprotein";
  $("met_resp_options").hidden = !showResp;
  $("protein_residue_options").hidden = !showResidues;
  document.querySelector(".viewer-main")?.classList.toggle("has-side-panel", showResp || showResidues);
}

function syncViewerControls() {
  const workflow = $("workflow_type")?.value || "";
  const showCartoon = workflow === "metalloprotein";
  const showLabels = workflow === "metallophore";
  $("show_cartoon_wrap").hidden = !showCartoon;
  $("show_labels_wrap").hidden = !showLabels;
  if (!showCartoon) $("show_cartoon").checked = false;
  if (!showLabels) $("show_labels").checked = false;
}

function syncMetal1264Ui() {
  const workflow = $("workflow_type").value;
  const isMetallophore = workflow === "metallophore";
  const existingResp = $("met_mode").value === "existing_resp";
  $("metal_1264_section").hidden = workflow === "deep_eutectic" || workflow === "add_library" || (isMetallophore && !existingResp);
  $("metallophore_geometry_actions").hidden = !isMetallophore || existingResp;
  syncSystem1264EnabledState(false);
}

function selectedSystemMetalSpecies() {
  const species = [];
  const seen = new Set();
  for (const row of document.querySelectorAll(".metal-charge-row")) {
    const element = String(row.dataset.element || "").trim();
    const charge = Number.parseInt(row.querySelector(".metal-charge-select")?.value || "0", 10);
    if (!element || !Number.isInteger(charge) || charge < 1) continue;
    const key = `${element}:${charge}`;
    if (seen.has(key)) continue;
    seen.add(key);
    species.push({element, charge});
  }
  return species;
}

function unsupportedDuvailSpecies() {
  return selectedSystemMetalSpecies().filter(({element, charge}) => {
    const record = (state.bootstrap?.supported_metals || []).find((item) => item.element === element);
    return !record || !(record.duvail_charges || []).map(Number).includes(Number(charge));
  });
}

function syncSystemC4ParameterSet(updateWater = true, announce = false) {
  const select = $("system_c4_parameter_set");
  const unsupported = unsupportedDuvailSpecies();
  const duvailOption = Array.from(select.options).find((item) => item.value === "opc_duvail");
  if (duvailOption) duvailOption.disabled = unsupported.length > 0;

  let forcedToLimerz = false;
  if (unsupported.length && select.value === "opc_duvail") {
    select.value = "spce_limerz";
    forcedToLimerz = true;
  }
  const value = select.value || "opc_duvail";
  const record = (state.bootstrap?.c4_parameter_sets || []).find((item) => item.key === value);
  const recommendedWater = record?.water_model || (value === "spce_limerz" ? "spce" : "opc");
  if ((updateWater || forcedToLimerz) && Array.from($("water_model").options).some((item) => item.value === recommendedWater)) {
    $("water_model").value = recommendedWater;
  }
  $("system_c4_note").textContent = record?.description || "Select the 12-6-4 parameter family used during MD system setup.";

  const compatibilityWarning = $("system_c4_compat_warning");
  if (unsupported.length) {
    const unsupportedText = unsupported.map(({element, charge}) => `${element}${charge}+`).join(", ");
    const message = `OPC + Duvail does not support the selected ion(s): ${unsupportedText}. Duvail is disabled; SPC/E + Li/Merz and SPC/E solvation are selected instead.`;
    compatibilityWarning.textContent = message;
    compatibilityWarning.hidden = false;
    if (announce && forcedToLimerz) setStatus(message, "warn");
  } else {
    compatibilityWarning.textContent = "";
    compatibilityWarning.hidden = true;
  }
  syncSystemC4WaterWarning(false);
  return unsupported;
}

function waterModelLabel(value) {
  return {opc: "OPC", spce: "SPC/E", tip3p: "TIP3P", opc3: "OPC3", tip5p: "TIP5P"}[String(value || "").toLowerCase()]
    || String(value || "").toUpperCase();
}

function syncSystemC4WaterWarning(announce = false) {
  const warning = $("system_c4_water_warning");
  const parameterSet = $("system_c4_parameter_set").value || "opc_duvail";
  const record = (state.bootstrap?.c4_parameter_sets || []).find((item) => item.key === parameterSet);
  const recommendedWater = record?.water_model || (parameterSet === "spce_limerz" ? "spce" : "opc");
  const selectedWater = $("water_model").value || "";
  const active = $("apply_1264_metals").checked && !$("metal_1264_section").hidden;
  const mismatch = active && selectedWater.toLowerCase() !== recommendedWater.toLowerCase();
  warning.hidden = !mismatch;
  if (!mismatch) {
    warning.textContent = "";
    return false;
  }
  const message = `${record?.label || "The selected 12-6-4 set"} was parameterized for ${waterModelLabel(recommendedWater)} water, but ${waterModelLabel(selectedWater)} is selected. Use the recommended water model unless you have compatible parameters and have independently validated this combination.`;
  warning.textContent = message;
  if (announce) setStatus(message, "warn");
  return true;
}

function syncSystem1264EnabledState(resetWaterOnEnable = false) {
  const enabled = $("apply_1264_metals").checked;
  $("system_c4_parameter_set").disabled = !enabled;
  if (enabled && resetWaterOnEnable) {
    syncSystemC4ParameterSet(true);
  } else {
    syncSystemC4WaterWarning(false);
  }
}

function syncDes1264EnabledState() {
  $("des_c4_parameter_set").disabled = !$("des_apply_1264").checked;
}

function syncWorkflowPanels() {
  const workflow = $("workflow_type").value;
  const libraryMode = workflow === "add_library";
  $("library_workspace").hidden = !libraryMode;
  document.querySelector("main.workspace").hidden = libraryMode;
  $("job_name").closest(".field").hidden = libraryMode;
  $("output_root").closest(".field").hidden = libraryMode;
  $("metallophore_panel").hidden = workflow !== "metallophore";
  $("protein_panel").hidden = workflow !== "metalloprotein";
  $("des_panel").hidden = workflow !== "deep_eutectic";
  $("des_neutralization_note").hidden = workflow !== "deep_eutectic";
  $("met_mode_wrap").hidden = workflow !== "metallophore";
  document.querySelector(".top-controls")?.classList.toggle("no-met-mode", workflow !== "metallophore");
  $("protein_ff").disabled = workflow !== "metalloprotein";
  $("box_shape").disabled = workflow === "deep_eutectic";
  if (workflow === "deep_eutectic") {
    $("box_shape").value = "cubic";
    $("temperature").value = "298.15";
    if (!state.desSaltInitialized) {
      if ($("salt_mode").value === "none") {
        $("salt_kind").value = "NaCl";
        $("salt_mode").value = "neutralize";
        updateSaltUi();
      }
      state.desSaltInitialized = true;
    }
  } else if ($("temperature").value === "298.15") {
    $("temperature").value = "300";
  }
  syncMdProtocolOptions();
  syncMetMode(false);
  syncViewerSidePanels();
  syncViewerControls();
  syncMetal1264Ui();
  if (libraryMode || workflow === "deep_eutectic") {
    loadLibraryComponents().catch((err) => setStatus(err.message, "error"));
  }
  updateMdProtocolUi();
}

function syncMetMode(clear = true) {
  const isMet = $("workflow_type").value === "metallophore";
  const isLibrary = $("workflow_type").value === "add_library";
  const existing = $("met_mode").value === "existing_resp";
  $("met_input_block").hidden = !isMet || existing;
  $("met_resp_block").hidden = !isMet || !existing;
  $("met_residue_wrap").hidden = !isMet || existing;
  $("quick_min").hidden = !isMet || existing;
  $("restore_met").hidden = !isMet || existing;
  $("resp_qm_blocks").hidden = !isMet || existing;
  $("build_resp_assets").hidden = !isMet || existing || $("charge_method").value === "antechamber";
  $("finish").hidden = isLibrary || (isMet && !existing);
  syncViewerSidePanels();
  const solvationTab = document.querySelector('.tab[data-tab="solvation"]');
  const mdTab = document.querySelector('.tab[data-tab="md"]');
  const setupOnly = isMet && !existing;
  solvationTab.hidden = setupOnly;
  mdTab.hidden = setupOnly;
  if (setupOnly) activateTab("setup");
  if (clear) {
    resetMetallophoreTables();
    clearPreview("Mode changed. Load a new preview.");
  }
  renderRespGroupControls();
  renderRespGroupBox();
  syncMetal1264Ui();
  updateChargeMethodUi(false);
}

function updateChargeMethodUi(showNotice = true) {
  const am1bcc = $("charge_method").value === "antechamber";
  $("am1bcc_notice").hidden = !am1bcc;
  $("build_resp_assets").hidden = $("workflow_type").value !== "metallophore" || $("met_mode").value !== "resp_input" || am1bcc;
  for (const node of document.querySelectorAll(".nwchem-only input, .nwchem-only select")) {
    node.disabled = am1bcc;
  }
  if (am1bcc && showNotice) {
    setStatus("AM1-BCC uses AmberTools/Antechamber directly during dry-run, not NWChem RESP.", "warn");
  }
  updateQmGeometryUi();
}

function updateQmGeometryUi() {
  const loaded = $("qm_geometry").value === "use_loaded_geometry";
  $("qm_functional").disabled = loaded || $("charge_method").value === "antechamber";
  $("qm_basis").disabled = loaded || $("charge_method").value === "antechamber";
  const same = $("qm_resp_same").checked;
  $("qm_resp_functional").disabled = same || $("charge_method").value === "antechamber";
  $("qm_resp_basis").disabled = same || $("charge_method").value === "antechamber";
}

function updateSaltUi() {
  const mode = $("salt_mode").value;
  const active = mode === "count" || mode === "concentration";
  const neutralizationIon = $("neutralization_ion").value || "auto";
  const needsSaltPair = active || (mode === "neutralize" && neutralizationIon === "salt_default");
  $("salt_amount_wrap").hidden = !active;
  $("neutralization_ion_wrap").hidden = mode === "none";
  $("neutralization_ion").disabled = mode === "none";
  $("salt_kind_wrap").hidden = !needsSaltPair;
  $("salt_kind").disabled = !needsSaltPair;
  $("salt_unit").hidden = mode !== "concentration";
  if (mode === "concentration") {
    $("salt_amount_label").textContent = "Concentration";
  } else if (mode === "count") {
    $("salt_amount_label").textContent = "Formula units";
  }
}

function updateDesSizeModeUi() {
  const ratioMode = $("des_size_mode").value === "ratio_units";
  $("des_ratio_units").disabled = !ratioMode;
  $("des_box_length").disabled = ratioMode;
}

function updateDesBuildModeUi() {
  const packmol = $("des_build_mode").value === "packmol";
  $("des_target_density").disabled = false;
  $("des_packmol_tolerance").disabled = !packmol;
  $("des_spacing").disabled = packmol;
}

function updateProtocolSummary() {
  const protocol = $("md_protocol").value;
  const text = {
    "15step": "15-step: restrained minimization/heating/equilibration followed by production. Best for final AMBER-ready systems.",
    "4step": "4-step: compact minimization, heating, equilibration, and production. Useful for quick setup and screening.",
    "des_solvent": "DES solvent: conservative warm-up, density relaxation, equilibration, and production for deep eutectic boxes.",
  }[protocol] || "Choose a protocol to preview the major stages.";
  $("protocol_summary").textContent = text;
}

function mdStepsFromPs(ps, dt = 0.002) {
  return Math.round(Number(ps) / Number(dt));
}

function mdStepsFromNs(ns, dt = 0.002) {
  return Math.round((Number(ns) * 1000) / Number(dt));
}

function mdControlDefaults(defaults) {
  const next = {...(defaults || {})};
  if (next.ntt === undefined) next.ntt = 3;
  if (next.pres0 !== undefined) {
    if (next.barostat === undefined) next.barostat = 1;
    if (next.taup === undefined) next.taup = 1.0;
  }
  return next;
}

function mdStage(name, title, stageType, info, defaults) {
  return {name, title, stage_type: stageType, info, defaults: stageType === "md" ? mdControlDefaults(defaults) : defaults};
}

function amberMdStageDefs() {
  const protocol = $("md_protocol").value;
  const temp = Number.parseFloat($("temperature").value || (protocol === "des_solvent" ? "298.15" : "300"));
  const pressure = Number.parseFloat($("pressure").value || "1");
  const prodNs = Number.parseFloat($("production_ns").value || "10");
  if (protocol === "4step") {
    return [
      mdStage("01_min", "Restrained minimization", "min", "Relax clashes under positional restraints.", {maxcyc: 5000, ncyc: 2500, restraint_wt: 10.0}),
      mdStage("02_nvt", "NVT heating", "md", "Heat from 0 K at constant volume.", {nstlim: mdStepsFromPs(100), dt: 0.002, temp0: temp, tempi: 0.0, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
      mdStage("03_npt", "NPT equilibration", "md", "Relax density and pressure.", {nstlim: mdStepsFromPs(200), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, restraint_wt: 2.0, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
      mdStage("04_prod", "Production", "md", "Main production trajectory.", {nstlim: mdStepsFromNs(prodNs), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    ];
  }
  if (protocol === "des_solvent") {
    const stages = [
      mdStage("01_min", "DES minimization", "min", "Unrestrained solvent minimization.", {maxcyc: 10000, ncyc: 5000}),
      mdStage("02_settle_noshake", "1 K no-SHAKE settle", "md", "Rewrite a stable low-temperature restart before enabling SHAKE.", {nstlim: 100, dt: 0.0001, temp0: 1.0, tempi: 1.0, gamma_ln: 2.0, ntc: 1, ntf: 1, ntpr: 10, ntwx: 0, ntwr: 100}),
      mdStage("03_warm_nvt", "Warm NVT", "md", "Continue from the settle restart and warm gently from 1 K.", {nstlim: mdStepsFromPs(50, 0.001), dt: 0.001, temp0: 50.0, tempi: 1.0, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
      mdStage("04_heat_nvt", "Heat NVT", "md", "Heat to target temperature.", {nstlim: mdStepsFromPs(200, 0.001), dt: 0.001, temp0: temp, tempi: 50.0, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
      mdStage("05_density_soft_npt", "Soft NPT density", "md", "Early density relaxation after constant-volume heating.", {nstlim: mdStepsFromPs(250, 0.001), dt: 0.001, temp0: temp, tempi: temp, pres0: pressure, gamma_ln: 2.0, barostat: 1, taup: 0.5, iwrap: 1, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
      mdStage("06_equil_npt", "NPT equilibration", "md", "Longer pressure relaxation.", {nstlim: mdStepsFromPs(1000), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, gamma_ln: 2.0, barostat: 1, taup: 1.0, iwrap: 1, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    ];
    const mixingEnabled = $("des_mixing_enabled")?.checked !== false;
    if (mixingEnabled) {
      stages.push(mdStage("07_mix_500k_npt", "500 K NPT mixing", "md", "High-temperature NPT mixing before final production.", {nstlim: mdStepsFromNs(50), dt: 0.002, temp0: 500.0, tempi: temp, pres0: pressure, gamma_ln: 2.0, barostat: 1, taup: 1.0, iwrap: 1, ntpr: 25000, ntwx: 25000, ntwr: 50000}));
      stages.push(mdStage("08_prod", "DES production", "md", "DES production trajectory.", {nstlim: mdStepsFromNs(prodNs), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, gamma_ln: 2.0, barostat: 1, taup: 1.0, iwrap: 1, ntpr: 2500, ntwx: 2500, ntwr: 5000}));
    } else {
      stages.push(mdStage("07_prod", "DES production", "md", "DES production trajectory.", {nstlim: mdStepsFromNs(prodNs), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, gamma_ln: 2.0, barostat: 1, taup: 1.0, iwrap: 1, ntpr: 2500, ntwx: 2500, ntwr: 5000}));
    }
    return stages;
  }
  const stages = [25.0, 10.0, 5.0, 2.0, 1.0].map((weight, idx) => (
    mdStage(`${String(idx + 1).padStart(2, "0")}_min`, `Minimization ${idx + 1}`, "min", `Release positional restraint to ${weight} kcal/mol/A^2.`, {maxcyc: 5000, ncyc: 2500, restraint_wt: weight})
  ));
  stages.push(
    mdStage("06_nvt_100", "NVT 0->100 K", "md", "First restrained heating step.", {nstlim: mdStepsFromPs(50), dt: 0.002, temp0: 100.0, tempi: 0.0, restraint_wt: 1.0, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("07_nvt_200", "NVT 100->200 K", "md", "Second restrained heating step.", {nstlim: mdStepsFromPs(50), dt: 0.002, temp0: 200.0, tempi: 100.0, restraint_wt: 0.5, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("08_nvt_target", "NVT 200->target", "md", "Reach target temperature.", {nstlim: mdStepsFromPs(50), dt: 0.002, temp0: temp, tempi: 200.0, restraint_wt: 0.1, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("09_nvt_hold", "NVT hold", "md", "Hold target temperature.", {nstlim: mdStepsFromPs(50), dt: 0.002, temp0: temp, tempi: temp, restraint_wt: 0.1, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("10_npt_relax", "NPT relax", "md", "Begin pressure relaxation.", {nstlim: mdStepsFromPs(50), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, restraint_wt: 0.1, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("11_npt_soft", "NPT soft", "md", "Final soft restrained equilibration.", {nstlim: mdStepsFromPs(50), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, restraint_wt: 0.05, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("12_unrestrained_min", "Unrestrained minimization", "min", "Remove positional restraints before final relaxation.", {maxcyc: 5000, ncyc: 2500}),
    mdStage("13_unrestrained_nvt", "Unrestrained NVT", "md", "Unrestrained target-temperature NVT.", {nstlim: mdStepsFromPs(100, 0.001), dt: 0.001, temp0: temp, tempi: 0.0, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("14_unrestrained_npt", "Unrestrained NPT", "md", "Unrestrained pressure equilibration.", {nstlim: mdStepsFromPs(100), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000}),
    mdStage("15_prod", "Production", "md", "Main production trajectory.", {nstlim: mdStepsFromNs(prodNs), dt: 0.002, temp0: temp, tempi: temp, pres0: pressure, gamma_ln: 2.0, ntpr: 1000, ntwx: 1000, ntwr: 1000})
  );
  return stages;
}

function mdStageField(stage, key, value) {
  const id = `md_stage_${stage.name}_${key}`;
  const floatKeys = new Set(["dt", "temp0", "tempi", "pres0", "gamma_ln", "taup", "restraint_wt"]);
  const type = floatKeys.has(key) ? "float" : Number.isInteger(value) ? "int" : typeof value === "number" ? "float" : "string";
  const help = MD_FIELD_HELP[key] || `${key} override for this AMBER input stage.`;
  const attrs = `class="md-stage-field" title="${escapeHtml(help)}" data-stage="${escapeHtml(stage.name)}" data-key="${escapeHtml(key)}" data-type="${type}" data-default='${escapeHtml(JSON.stringify(value))}'`;
  if (key === "barostat") {
    return `<select id="${id}" ${attrs}><option value="1">Monte Carlo</option><option value="2">Berendsen</option></select>`;
  }
  if (key === "ntt") {
    return `<select id="${id}" ${attrs}><option value="3">Langevin</option><option value="1">Berendsen</option><option value="0">None</option></select>`;
  }
  if (key === "iwrap") {
    return `<select id="${id}" ${attrs}><option value="1">Wrap</option><option value="0">No wrap</option></select>`;
  }
  const step = type === "int" ? "1" : (key === "dt" ? "0.001" : "0.1");
  return `<input id="${id}" type="number" step="${step}" ${attrs} value="${escapeHtml(value)}">`;
}

function renderMdStageEditors() {
  const box = $("md_stage_editors");
  if (!box) return;
  state.mdStageDefs = amberMdStageDefs();
  box.innerHTML = state.mdStageDefs.map((stage, idx) => {
    const fields = Object.entries(stage.defaults || {}).map(([key, value]) => {
      const help = MD_FIELD_HELP[key] || `${key} override for this AMBER input stage.`;
      return `
        <div class="field"><label class="help-label" title="${escapeHtml(help)}" for="md_stage_${escapeHtml(stage.name)}_${escapeHtml(key)}">${escapeHtml(key)}</label>${mdStageField(stage, key, value)}</div>
      `;
    }).join("");
    return `
      <details class="stage-card" ${idx === 0 ? "open" : ""}>
        <summary>${escapeHtml(stage.name)} [${escapeHtml(stage.stage_type.toUpperCase())}] | ${escapeHtml(stage.title)}</summary>
        <div class="stage-info">${escapeHtml(stage.info || "")}</div>
        <div class="stage-fields">${fields}</div>
      </details>
    `;
  }).join("");
  for (const stage of state.mdStageDefs) {
    for (const [key, value] of Object.entries(stage.defaults || {})) {
      const node = $(`md_stage_${stage.name}_${key}`);
      if (node && node.tagName === "SELECT") node.value = String(value);
    }
  }
}

function collectMdStageOverrides() {
  const overrides = {};
  for (const node of document.querySelectorAll(".md-stage-field")) {
    const stage = node.dataset.stage;
    const key = node.dataset.key;
    const type = node.dataset.type;
    let value = node.value;
    if (type === "int") value = Number.parseInt(value || "0", 10);
    else if (type === "float") value = Number.parseFloat(value || "0");
    let defaultValue;
    try {
      defaultValue = JSON.parse(node.dataset.default);
    } catch (_err) {
      defaultValue = node.dataset.default;
    }
    const same = typeof value === "number" && typeof defaultValue === "number"
      ? Math.abs(value - defaultValue) < 1e-9
      : String(value) === String(defaultValue);
    if (!same) {
      if (!overrides[stage]) overrides[stage] = {};
      overrides[stage][key] = value;
    }
  }
  return overrides;
}

function syncMdProtocolOptions() {
  const workflow = $("workflow_type").value;
  const select = $("md_protocol");
  for (const opt of select.options) {
    if (opt.value === "des_solvent") {
      opt.hidden = workflow !== "deep_eutectic";
      opt.disabled = workflow !== "deep_eutectic";
    }
  }
  if (workflow === "deep_eutectic") {
    if (state.lastWorkflow !== "deep_eutectic") {
      $("temperature").value = "300";
      $("production_ns").value = "100";
      if ($("des_mixing_enabled")) $("des_mixing_enabled").checked = true;
    }
    select.value = "des_solvent";
    select.disabled = true;
  } else {
    select.disabled = false;
    if (select.value === "des_solvent") select.value = "15step";
  }
  state.lastWorkflow = workflow;
}

function updateMdProtocolUi() {
  syncMdProtocolOptions();
  updateProtocolSummary();
  renderMdStageEditors();
}

function syncFocusedRestraintUi() {
  const enabled = Boolean($("focused_enabled")?.checked);
  if ($("focused_fields")) $("focused_fields").hidden = !enabled;
}

function formatMissingLoopBlocks(missing) {
  const blocks = missing?.internal_blocks || [];
  if (!blocks.length) return "No internal missing loop blocks were reported.";
  const rows = blocks.slice(0, 8).map((block) => `
    <tr>
      <td>${escapeHtml(block.chain_id || "_")}</td>
      <td>${escapeHtml(block.range_label || "")}</td>
      <td>${escapeHtml(block.length || "")}</td>
    </tr>`).join("");
  const more = blocks.length > 8 ? `<div>${blocks.length - 8} more block(s)</div>` : "";
  return `<table><thead><tr><th>Chain</th><th>Range</th><th>Length</th></tr></thead><tbody>${rows}</tbody></table>${more}`;
}

function askMissingLoopAction(payload) {
  return new Promise((resolve) => {
    const dialog = $("missing_loop_dialog");
    $("missing_loop_message").textContent = `${payload.message || "Missing loops were detected"} Choose whether PDBFixer should rebuild internal loops before preview.`;
    $("missing_loop_summary").innerHTML = formatMissingLoopBlocks(payload.missing_loops || {});
    const close = () => {
      dialog.removeEventListener("close", close);
      resolve(dialog.returnValue || "cancel");
    };
    dialog.addEventListener("close", close);
    dialog.showModal();
  });
}

async function loadMetallophore() {
  const payload = collectPayload();
  if (payload.metallophore.mode === "resp_input" && !payload.metallophore.input_path && !payload.metallophore.smiles_text) {
    throw new Error("Upload a PDB/MOL2/SDF/SMILES file or enter one valid SMILES string.");
  }
  if (payload.metallophore.mode === "existing_resp" && !payload.metallophore.resp_job_dir) {
    throw new Error("Choose a RESP candidate or enter a RESP job/result folder.");
  }
  setStatus("Loading metallophore preview...", "warn");
  const data = await api("/api/metallophore/load", payload);
  rememberInitialMetallophore(data);
  state.metInsertedMetalIndices = new Set();
  renderMetalsAndDonors(data);
  setViewerDefaults("metallophore");
  await renderPdbScene(data.pdb, "metallophore", {
    atoms: data.atoms || [],
    bonds: data.bonds || [],
    message: payload.metallophore.mode === "existing_resp" ? "Existing RESP preview loaded." : "Metallophore preview loaded.",
    info: `${data.atoms.length} atoms, ${data.metals.length} supported metals`,
  });
  setStatus("Metallophore preview loaded.", "ok");
}

async function restoreMetallophoreGeometry() {
  if (!state.metOriginal?.data) throw new Error("Load a metallophore preview before restoring geometry.");
  const data = cloneJson(state.metOriginal.data);
  const inputPath = state.metOriginal.inputPath || data.source_path || "";
  $("met_residue").value = state.metOriginal.residueName || $("met_residue").value || "LIG";
  $("met_input_path").value = inputPath;
  $("met_smiles").value = "";
  $("met_file").value = "";
  $("met_file_name").textContent = inputPath ? `Restored initial structure: ${inputPath}` : "Restored initial structure.";
  state.metCoordination = null;
  state.metInsertedMetalIndices = new Set();
  renderMetalsAndDonors(data);
  setViewerDefaults("metallophore");
  await renderPdbScene(data.pdb, "metallophore", {
    atoms: data.atoms || [],
    bonds: data.bonds || [],
    message: "Initial metallophore geometry restored.",
    info: `${(data.atoms || []).length} atoms, ${(data.metals || []).length} supported metals`,
  });
  setStatus("Initial metallophore geometry restored.", "ok");
}

async function quickMinimize() {
  if (!state.selectedMetalAtom) throw new Error("Select a supported metal atom first.");
  if (state.selectedDonors.size > MAX_QUICK_MIN_DONORS) {
    throw new Error(`Select ${MAX_QUICK_MIN_DONORS} or fewer donor atoms for quick minimization.`);
  }
  const selectedMetal = state.selectedMetalAtom;
  const payload = collectPayload();
  payload.metallophore.metal_atom_index = state.selectedMetalAtom;
  payload.metallophore.donor_atom_indices = Array.from(state.selectedDonors);
  payload.metallophore.metal_coordination = collectSelectedMetalCoordination();
  const coordination = payload.metallophore.metal_coordination;
  if (coordination?.coordination_mode !== "manual_selection" && state.selectedDonors.size > Number(coordination?.target_cn || 0)) {
    throw new Error(`Selected donor count (${state.selectedDonors.size}) exceeds Target CN ${coordination.target_cn}. Choose Manual selection to minimize with all selected donors, or deselect donors.`);
  }
  setStatus("Running quick Open Babel UFF cleanup...", "warn");
  const data = await api("/api/metallophore/quick-minimize", payload);
  $("met_input_path").value = data.output_path;
  $("met_smiles").value = "";
  $("met_file_name").textContent = `Quick-minimized file: ${data.output_path}`;
  renderMetalsAndDonors(data);
  state.metCoordination = data.metal_coordination || state.metCoordination;
  state.selectedMetalAtom = state.metMetals.some((atom) => Number(atom.index) === Number(selectedMetal)) ? Number(selectedMetal) : state.selectedMetalAtom;
  ensureSelectedMetalAtom();
  const requiredDonors = data.metal_coordination?.required_donor_atom_indices || payload.metallophore.donor_atom_indices || [];
  state.selectedDonors = new Set(requiredDonors.map(Number).filter((idx) => state.metAtoms.some((atom) => Number(atom.index) === Number(idx))));
  state.selectedAtomIndices = new Set([selectedMetal, ...effectiveDonorIndicesForSelectedMetal()].filter(Boolean).map(Number));
  renderMetalsBox();
  renderDonorBox();
  refreshMetalCoordinationStatus();
  setViewerDefaults("metallophore");
  await renderPdbScene(data.pdb, "metallophore", {
    atoms: data.atoms || [],
    bonds: data.bonds || [],
    message: "Quick-minimized metallophore loaded.",
    info: (data.warnings || []).join(" "),
  });
  setStatus(data.warnings?.length ? data.warnings.join(" ") : "Quick minimization finished.", data.warnings?.length ? "warn" : "ok");
}

async function addMetallophoreMetal() {
  if (!state.metAtoms.length) throw new Error("Load a metallophore preview before adding a metal.");
  if (!state.selectedDonors.size) throw new Error("Select one or more donor atoms first.");
  const element = $("met_insert_element").value || "Fe";
  const charge = Number($("met_insert_charge").value || defaultMetalCharge(element));
  const cnRaw = $("met_insert_cn").value;
  const selectedDonors = Array.from(state.selectedDonors);
  const payload = collectPayload();
  payload.metallophore.metal_insertion = {
    element,
    charge,
    target_cn: cnRaw === MANUAL_CN_VALUE ? null : Number(cnRaw || defaultCoordinationNumber(element, charge)),
    coordination_mode: cnRaw === MANUAL_CN_VALUE ? "manual_selection" : "target_cn",
    donor_atom_indices: selectedDonors,
  };
  const data = await api("/api/metallophore/add-metal", payload);
  const selectedMetal = data.metal_coordination?.metal_atom_index || null;
  $("met_input_path").value = data.output_path;
  $("met_smiles").value = "";
  $("met_file_name").textContent = `Edited metallophore file: ${data.output_path}`;
  renderMetalsAndDonors(data);
  state.metCoordination = data.metal_coordination || state.metCoordination;
  if (selectedMetal) {
    state.selectedMetalAtom = Number(selectedMetal);
    state.metInsertedMetalIndices.add(Number(selectedMetal));
  }
  const requiredDonors = data.metal_coordination?.required_donor_atom_indices || selectedDonors;
  state.selectedDonors = new Set(requiredDonors.map(Number).filter((idx) => state.metAtoms.some((atom) => Number(atom.index) === Number(idx))));
  state.selectedAtomIndices = new Set([selectedMetal, ...effectiveDonorIndicesForSelectedMetal()].filter(Boolean).map(Number));
  renderMetalsBox();
  renderDonorBox();
  refreshMetalCoordinationStatus();
  setViewerDefaults("metallophore");
  await renderPdbScene(data.pdb, "metallophore", {
    atoms: data.atoms || [],
    bonds: data.bonds || [],
    message: "Inserted metallophore metal loaded.",
    info: (data.warnings || []).join(" "),
  });
  setStatus(data.warnings?.length ? data.warnings.join(" ") : "Metal inserted into metallophore preview.", data.warnings?.length ? "warn" : "ok");
}

async function removeSelectedMetallophoreInsertedMetal() {
  if (!state.metAtoms.length) throw new Error("Load a metallophore preview before removing an inserted metal.");
  const selected = Number(state.selectedMetalAtom);
  if (!Number.isFinite(selected) || selected <= 0) throw new Error("Select an inserted metal atom first.");
  if (!state.metInsertedMetalIndices.has(selected)) {
    throw new Error("Only metals added with Add Metal in this GUI session can be removed here.");
  }
  state.metAtoms = state.metAtoms.filter((atom) => Number(atom.index) !== selected);
  state.metBonds = state.metBonds.filter((bond) => Number(bond.first) !== selected && Number(bond.second) !== selected);
  state.metInsertedMetalIndices.delete(selected);
  state.metCoordination = null;
  state.selectedDonors = new Set([...state.selectedDonors].filter((index) => Number(index) !== selected));
  state.selectedAtomIndices = new Set([...state.selectedAtomIndices].filter((index) => Number(index) !== selected));
  const residueName = $("met_residue").value || "LIG";
  const data = moleculePayloadFromState(residueName);
  renderMetalsAndDonors(data);
  state.selectedMetalAtom = state.metMetals[0]?.index || null;
  ensureSelectedMetalAtom();
  $("met_file_name").textContent = "Edited metallophore in memory.";
  setViewerDefaults("metallophore");
  await renderPdbScene(data.pdb, "metallophore", {
    atoms: data.atoms || [],
    bonds: data.bonds || [],
    message: "Inserted metallophore metal removed.",
    info: `${data.atoms.length} atoms, ${data.metals.length} supported metals`,
  });
  setStatus("Inserted metallophore metal removed.", "ok");
}

async function reloadProteinPreview(options = {}) {
  const missingLoopAction = options.missingLoopAction || "";
  const keepSelection = Boolean(options.keepSelection);
  const requestSerial = options.requestSerial ?? state.proteinLoadSerial;
  const selectedBefore = new Set(state.selectedResidueKeys);
  const propkaBefore = new Set(state.propkaResidueKeys);
  const disulfideTokensBefore = state.disulfideTokens.slice();
  const payload = collectPayload();
  if (missingLoopAction) payload.ui = {missing_loop_action: missingLoopAction};
  const {response, data} = await postJson("/api/protein/load", payload);
  if (response.status === 409 && data.action_required === "missing_loops") {
    if (requestSerial !== state.proteinLoadSerial) return null;
    const choice = await askMissingLoopAction(data);
    if (choice === "repair" || choice === "skip") {
      state.lastProteinMissingLoopAction = choice;
      return reloadProteinPreview({missingLoopAction: choice, keepSelection, requestSerial});
    }
    setStatus("Protein loading canceled before missing-loop repair choice.", "warn");
    return null;
  }
  if (!response.ok || data.ok === false) throw new Error(data.error || `Request failed: ${response.status}`);
  if (requestSerial !== state.proteinLoadSerial) return null;
  applyProteinPreviewData(data);
  if (keepSelection) {
    state.selectedResidueKeys = selectedBefore;
    state.propkaResidueKeys = propkaBefore;
    state.disulfideTokens = disulfideTokensBefore;
    state.disulfideKeys = new Set(
      state.disulfideCandidates
        .filter((item) => state.disulfideTokens.includes(item.token))
        .flatMap((item) => [item.key_a, item.key_b])
        .filter(Boolean)
    );
    renderResiduePanel(state.proteinResidues || []);
    renderDisulfides(state.disulfideCandidates || []);
  }
  renderProteinMetalBox(data);
  setViewerDefaults("protein");
  await renderPdbScene(data.pdb, "protein", {
    message: "Protein preview loaded.",
    info: `${data.summary.residue_counts.standard} standard residues, ${data.metals.length} metals`,
    highlight_sets: data.highlight_sets || {},
    metal_binding_links: data.metal_binding_links || [],
  });
  state.proteinLoaded = true;
  if (data.warnings?.length) setStatus(data.warnings.join(" "), "warn");
  else setStatus("Protein preview loaded.", "ok");
  return data;
}

async function loadProtein() {
  syncProteinInputFields();
  const requestSerial = ++state.proteinLoadSerial;
  state.proteinMetalActions = new Map();
  state.proteinInsertions = [];
  state.proteinInsertionDonors = [];
  renderProteinInsertionDonors([]);
  renderProteinInsertions();
  state.propkaChanges = [];
  state.propkaResidueKeys = new Set();
  state.disulfideTokens = [];
  state.disulfideKeys = new Set();
  clearPreview("Loading protein preview...");
  setStatus("Loading protein preview...", "warn");
  return reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, requestSerial});
}

async function runPropka() {
  setStatus("Running PropKa...", "warn");
  const data = await api("/api/protein/propka", collectPayload());
  renderPropka(data.candidates || []);
  setStatus(data.warnings?.length ? data.warnings.join(" ") : "PropKa review loaded.", data.warnings?.length ? "warn" : "ok");
}

async function buildDesPreview() {
  setStatus("Building DES heavy-atom preview...", "warn");
  const data = await api("/api/des/preview", collectPayload());
  const plan = data.plan || {};
  const metalCount = (plan.metal_sites || []).length;
  const ionSummary = Object.entries(plan.added_ions || {}).map(([name, count]) => `${name} x ${count}`).join(", ") || "none";
  $("des_plan_box").textContent = `Atoms: ${plan.total_atoms}, residues: ${plan.total_residues}, metals: ${metalCount}, box A: ${(plan.box_lengths_angstrom || []).map((v) => Number(v).toFixed(1)).join(" x ")}, initial density: ${Number(plan.estimated_initial_density_g_ml || 0).toFixed(3)} g/mL, charge: ${Number(plan.charge_before_ions || 0).toFixed(3)} -> ${Number(plan.final_charge || 0).toFixed(3)}, added ions: ${ionSummary}, C4 mask: ${plan.c4_mask || "none"}${data.preview_note ? ` — ${data.preview_note}` : ""}`;
  setViewerDefaults("des");
  await renderPdbScene(data.pdb, "des", {box_lines: data.box_lines, message: "DES heavy-atom preview loaded.", info: $("des_plan_box").textContent});
  setStatus(data.preview_note || "DES preview loaded.", data.preview_note ? "warn" : "ok");
}

function atomBounds(atoms) {
  if (!atoms.length) return {min: [0, 0, 0], max: [20, 20, 20], center: [10, 10, 10]};
  const min = [Math.min(...atoms.map((a) => Number(a.x))), Math.min(...atoms.map((a) => Number(a.y))), Math.min(...atoms.map((a) => Number(a.z)))];
  const max = [Math.max(...atoms.map((a) => Number(a.x))), Math.max(...atoms.map((a) => Number(a.y))), Math.max(...atoms.map((a) => Number(a.z)))];
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  return {min, max, center};
}

function cubeBoxLines(center, side) {
  const h = side / 2;
  const corners = [
    [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
    [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
  ].map((p) => [p[0] + center[0], p[1] + center[1], p[2] + center[2]]);
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  return edges.map(([a, b]) => ({start: corners[a], end: corners[b]}));
}

function octBoxLines(center, side) {
  const {axial: a, bevel: b} = octBoxLimits(side);
  const vertices = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) vertices.push([sx * a, sy * b, 0], [sx * b, sy * a, 0]);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) vertices.push([sx * a, 0, sz * b], [sx * b, 0, sz * a]);
  for (const sy of [-1, 1]) for (const sz of [-1, 1]) vertices.push([0, sy * a, sz * b], [0, sy * b, sz * a]);
  const points = vertices.map((p) => [p[0] + center[0], p[1] + center[1], p[2] + center[2]]);
  const lines = [];
  for (let i = 0; i < points.length; i++) {
    const distances = [];
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const d = Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1], points[i][2] - points[j][2]);
      distances.push([d, j]);
    }
    distances.sort((x, y) => x[0] - y[0]);
    for (const [, j] of distances.slice(0, 3)) {
      if (i < j) lines.push({start: points[i], end: points[j]});
    }
  }
  return lines;
}

function octBoxLimits(side) {
  const value = Math.max(0, Number(side) || 0);
  const axial = value * 0.42;
  const bevel = value * 0.22;
  return {axial, bevel, l1: axial + bevel};
}

function estimateSaltFormulaUnits(molarity, side, shape) {
  const volumeAngstrom3 = Math.pow(Math.max(0, Number(side) || 0), 3) * (shape === "oct" ? 0.77 : 1.0);
  return Math.max(0, Math.round((Number(molarity) || 0) * volumeAngstrom3 * ANGSTROM3_TO_LITER * AVOGADRO));
}

function ionStoichiometry(kind) {
  if (kind === "CaCl2") return {cation: "Ca2+", cations: 1, anion: "Cl-", anions: 2};
  if (kind === "KCl") return {cation: "K+", cations: 1, anion: "Cl-", anions: 1};
  return {cation: "Na+", cations: 1, anion: "Cl-", anions: 1};
}

function seededRandom(seed) {
  let value = Number(seed) >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomPointSampler(center, side, shape, seed) {
  const random = seededRandom(seed);
  const cubicLimit = side * 0.44;
  const oct = octBoxLimits(side);
  const octMargin = Math.min(1.4, Math.max(0.35, side * 0.015));
  const octAxialLimit = Math.max(0, oct.axial - octMargin);
  const octL1Limit = Math.max(0, oct.l1 - octMargin * 1.7);
  return () => {
    for (let attempt = 0; attempt < 200; attempt++) {
      if (shape === "oct") {
        const dx = (random() * 2 - 1) * octAxialLimit;
        const dy = (random() * 2 - 1) * octAxialLimit;
        const dz = (random() * 2 - 1) * octAxialLimit;
        if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > octL1Limit) continue;
        return [center[0] + dx, center[1] + dy, center[2] + dz];
      }
      const fx = random() * 2 - 1;
      const fy = random() * 2 - 1;
      const fz = random() * 2 - 1;
      return [center[0] + fx * cubicLimit, center[1] + fy * cubicLimit, center[2] + fz * cubicLimit];
    }
    return [center[0], center[1], center[2]];
  };
}

function occupiedSolutePoints() {
  return (state.currentAtoms || [])
    .map((atom) => [Number(atom.x), Number(atom.y), Number(atom.z)])
    .filter((point) => point.every((value) => Number.isFinite(value)));
}

function hasMinimumDistance(point, occupied, minDistance) {
  const min2 = minDistance * minDistance;
  for (const other of occupied) {
    const dx = point[0] - other[0];
    const dy = point[1] - other[1];
    const dz = point[2] - other[2];
    if (dx * dx + dy * dy + dz * dz < min2) return false;
  }
  return true;
}

function sampleIonPositions(center, side, shape, count, occupied, seed) {
  const positions = [];
  if (!count) return positions;
  const sampler = randomPointSampler(center, side, shape, seed);
  const distanceScales = [1.0, 0.85, 0.7, 0.55];
  for (const scale of distanceScales) {
    const minDistance = 3.0 * scale;
    let guard = 0;
    const maxTries = Math.max(500, count * 80);
    while (positions.length < count && guard < maxTries) {
      guard += 1;
      const point = sampler();
      if (!hasMinimumDistance(point, occupied, minDistance)) continue;
      occupied.push(point);
      positions.push(point);
    }
    if (positions.length >= count) break;
  }
  while (positions.length < count) {
    const point = sampler();
    occupied.push(point);
    positions.push(point);
  }
  return positions;
}

function solvationIons(center, side, shape) {
  const salt = collectSalt();
  if (salt.kind === "none" || salt.mode === "none") return {ions: [], summary: "Ion colors: no salt ions are drawn for the current salt mode."};
  if (salt.mode === "neutralize") {
    return {ions: [], summary: "Neutralize mode: LEaP will add counterions based on the system net charge; this preview does not estimate those positions."};
  }
  const stoich = ionStoichiometry(salt.kind);
  const formulaUnits = salt.mode === "count"
    ? Math.max(0, Math.round(Number(salt.value || 0)))
    : estimateSaltFormulaUnits(salt.value || 0, side, shape);
  const totalIons = formulaUnits * (stoich.cations + stoich.anions);
  const drawCount = Math.min(MAX_PREVIEW_IONS, totalIons);
  const occupied = occupiedSolutePoints();
  const ions = [];
  const cationCount = formulaUnits * stoich.cations;
  const anionCount = formulaUnits * stoich.anions;
  const drawnCations = totalIons ? Math.round(drawCount * cationCount / totalIons) : 0;
  const drawnAnions = drawCount - drawnCations;
  const cationPositions = sampleIonPositions(center, side, shape, drawnCations, occupied, 1264 + drawnCations);
  const anionPositions = sampleIonPositions(center, side, shape, drawnAnions, occupied, 4312 + drawnAnions);
  const positions = [
    ...cationPositions.map((position) => ({position, name: stoich.cation})),
    ...anionPositions.map((position) => ({position, name: stoich.anion})),
  ];
  for (let i = 0; i < positions.length; i++) {
    const ionName = positions[i].name;
    ions.push({
      name: ionName,
      color: ION_COLORS[ionName] || [0.2, 0.8, 0.2],
      radius: totalIons > 900 ? 0.24 : 0.34,
      position: positions[i].position,
    });
  }
  const concentration = salt.mode === "concentration" ? `${Number(salt.value || 0).toFixed(3)} M, ` : "";
  const drawn = drawCount < totalIons ? ` Drawing ${drawCount} representative ions for browser performance.` : "";
  return {
    ions,
    summary: `Ion colors: ${concentration}${formulaUnits} ${salt.kind} formula unit${formulaUnits === 1 ? "" : "s"} -> ${totalIons} ion${totalIons === 1 ? "" : "s"}.${drawn}`,
  };
}

function ionLegendHtml(ions, summary = "") {
  const names = [...new Set((ions || []).map((ion) => ion.name).filter(Boolean))];
  if (!names.length) return escapeHtml(summary || "Ion colors: no salt ions are drawn for the current salt mode.");
  return `${escapeHtml(summary)} ${names.map((name) => (
    `<span class="legend-item"><span class="legend-swatch" style="background:${rgbToHex(ION_COLORS[name] || [0.2, 0.8, 0.2])}"></span>${escapeHtml(name)}</span>`
  )).join(" ")}`;
}

async function previewSolvation() {
  if (!state.currentPdb) throw new Error("Load a structure preview before previewing solvation.");
  const bounds = atomBounds(state.currentAtoms);
  const span = Math.max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]);
  const side = span + Number.parseFloat($("buffer_angstrom").value || "10") * 2;
  const shape = $("box_shape").value;
  const lines = shape === "oct" ? octBoxLines(bounds.center, side) : cubeBoxLines(bounds.center, side);
  const ionPreview = solvationIons(bounds.center, side, shape);
  const ions = ionPreview.ions || [];
  await renderPdbScene(state.currentPdb, state.sceneKind || "metallophore", {
    ...(state.sceneKind === "metallophore" ? {atoms: state.currentAtoms} : {}),
    bonds: state.currentBonds,
    box_lines: lines,
    ions,
    message: `${shape.toUpperCase()} solvation preview loaded.`,
    info: `Illustrative ${shape} box, side approx ${side.toFixed(1)} A. Water is omitted; LEaP may produce different final box dimensions and ion coordinates.`,
  });
  if ($("ion_legend")) $("ion_legend").innerHTML = ionLegendHtml(ions, ionPreview.summary);
  setStatus("Solvation preview loaded.", "ok");
}

async function shutdownGui() {
  setStatus("Shutting down SIMPLE Web GUI...", "warn");
  await api("/api/quit", {});
  window.open("", "_self");
  window.close();
  setTimeout(() => {
    document.body.innerHTML = "<div class=\"closed-message\">SIMPLE Web GUI closed. You can return to the command prompt.</div>";
  }, 250);
}

async function finishWorkflow() {
  setStatus("Building final AMBER inputs. This may take a moment...", "warn");
  const data = await api("/api/finish", collectPayload());
  const siteResp = data.protein_site_resp || data.result?.protein_site_resp || null;
  if (siteResp?.status === "cluster_review_required") {
    $("finish_note").textContent = "This result is missing the reviewed cluster metadata required for GUI import. Reopen it through main.py.";
    setStatus("Use main.py to review this protein-site RESP cluster before importing it in the GUI.", "error");
    return data;
  }
  if (siteResp?.status === "setup_pending") {
    renderProteinSiteRespReview(siteResp);
    $("finish_note").textContent = "The selected site-RESP job is not complete. Finish it through main.py/NWChem, then scan the case folder again.";
    setStatus("Only completed main.py site-RESP results can be imported by the GUI.", "warn");
    return data;
  }
  if (siteResp?.status === "reference_pending") {
    $("finish_note").textContent = siteResp.message || "An executed TLeap reference topology is required.";
    setStatus("Protein-site RESP reference topology is pending.", "warn");
    return data;
  }
  if (siteResp?.status === "review_required") {
    state.proteinSiteRespApproved = false;
    renderProteinSiteRespReview(siteResp);
    $("finish_note").textContent = "A fingerprint-matching RESP result was found. Review the charge table and approve it to resume MD generation.";
    setStatus("Protein-site RESP charge review is required before application.", "warn");
    return data;
  }
  if (siteResp?.status === "applied") {
    renderProteinSiteRespReview(siteResp);
  }
  const outputs = data.result?.system?.output_files || {};
  const outputText = [outputs.prmtop, outputs.inpcrd].filter(Boolean).join(" | ");
  $("finish_note").textContent = outputText
    ? `Build complete. ${outputText}`
    : `Build complete. Config: ${data.config_path}`;
  setStatus(`Build complete. Config: ${data.config_path}`, "ok");
  return data;
}

function renderProteinSiteRespCandidates(candidates) {
  const box = $("protein_site_resp_candidates");
  if (!box) return;
  if (!candidates?.length) {
    box.textContent = "No protein-site RESP manifests were found below this search root.";
    return;
  }
  box.innerHTML = `<table><thead><tr><th>Source / site</th><th>Scope</th><th>Spin</th><th>Status</th><th>Folder</th></tr></thead><tbody>${candidates.map((item, index) => `
    <tr>
      <td>${escapeHtml(item.description || `${item.source_label || "protein"}; sites ${(item.metal_sites || []).join(",")}`)}</td>
      <td>${escapeHtml(item.scope || "")}</td>
      <td>${escapeHtml(item.multiplicity ?? "")}</td>
      <td>${item.completed ? "Completed" : "Pending"}</td>
      <td><button type="button" class="select-protein-site-resp" data-index="${index}" ${item.completed ? "" : "disabled"}>${item.completed ? "Select" : "Not ready"}</button><br>${escapeHtml(item.job_dir || "")}</td>
    </tr>`).join("")}</tbody></table>`;
  for (const button of document.querySelectorAll(".select-protein-site-resp")) {
    button.addEventListener("click", () => {
      const selected = candidates[Number(button.dataset.index)];
      $("protein_site_resp_job_dir").value = selected.job_dir || "";
      $("protein_site_resp_mode").value = "resp";
      $("protein_site_resp_scope").value = selected.scope || "sidechain";
      $("protein_site_resp_multiplicity").value = String(selected.multiplicity || 1);
      $("protein_site_resp_multiplicity_confirmed").checked = true;
      const cluster = selected.cluster || {};
      state.proteinSiteRespClusters = cluster.metal_sites?.length ? [{
        metal_sites: cluster.metal_sites,
        donor_residues: cluster.donor_residue_keys || [],
        fixed_environment: cluster.fixed_environment_keys || [],
        multiplicity: selected.multiplicity || cluster.multiplicity || 1,
        job_dir: selected.job_dir || null,
      }] : [];
      state.proteinSiteRespApproved = false;
      setStatus("Completed main.py site-RESP result selected. SIMPLE will still require an exact prepared-system fingerprint match.", "ok");
    });
  }
}

function renderProteinSiteRespReview(siteResp) {
  const box = $("protein_site_resp_review");
  if (!box) return;
  const jobs = siteResp.jobs || [];
  const reviews = jobs.map((job) => job.review).filter(Boolean);
  if (!reviews.length) {
    box.hidden = false;
    box.innerHTML = `<strong>${escapeHtml(siteResp.message || siteResp.status || "Protein-site RESP")}</strong>${jobs.length ? `<br>${jobs.map((job) => escapeHtml(job.job_dir || "")).join("<br>")}` : ""}`;
    if ($("apply_protein_site_resp")) $("apply_protein_site_resp").hidden = true;
    return;
  }
  const rows = reviews.flatMap((review) => (review.changes || []).map((change) => `
    <tr><td>${escapeHtml(`${change.residue_key || ""}@${change.atom_name || ""}`)}</td>
    <td>${Number(change.original_charge).toFixed(6)}</td><td>${Number(change.charge).toFixed(6)}</td>
    <td>${Number(change.delta).toFixed(6)}</td></tr>`));
  const metrics = reviews.map((review) => escapeHtml(`${review.description || "site"}: ESP RMSE ${review.esp_rmse ?? "N/A"}; constraint residual ${review.maximum_constraint_residual ?? "N/A"}`)).join("<br>");
  const residueSums = reviews.flatMap((review) => review.residue_sums || []);
  const sumText = residueSums.map((item) => escapeHtml(`${item.label}: ${Number(item.baseline).toFixed(6)} -> ${Number(item.fitted).toFixed(6)}`)).join("<br>");
  const symmetryCount = reviews.reduce((count, review) => count + (review.symmetry_constraints || []).length, 0);
  const warnings = reviews.flatMap((review) => review.warnings || []);
  box.hidden = false;
  box.innerHTML = `${metrics}${sumText ? `<br><strong>Residue totals</strong><br>${sumText}` : ""}<br>Verified symmetry constraints: ${symmetryCount}${warnings.length ? `<div class="warn-box">${warnings.map(escapeHtml).join("<br>")}</div>` : ""}
    <table><thead><tr><th>Atom</th><th>Baseline</th><th>RESP</th><th>Delta</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
  if ($("apply_protein_site_resp")) $("apply_protein_site_resp").hidden = false;
}

async function scanProteinSiteRespFolders() {
  const data = await api("/api/protein-site-resp/candidates", {
    search_root: $("protein_site_resp_search_root")?.value.trim() || ".",
  });
  renderProteinSiteRespCandidates(data.candidates || []);
  setStatus(`Protein-site RESP scan found ${(data.candidates || []).length} candidate(s). Exact compatibility is checked after TLeap.`, "ok");
}

async function uploadProteinSiteRespCase(fileList) {
  const allowed = /\.(?:json|grid|xyz|nw|sbatch|py|txt|log|out|pdb|prmtop|inpcrd|toml)$/i;
  const selected = Array.from(fileList || []).filter((file) => allowed.test(file.name));
  if (!selected.length) {
    throw new Error("The selected case folder contains no recognizable protein-site RESP files.");
  }
  if (!selected.some((file) => file.name === "site_resp_manifest.json")) {
    throw new Error("Choose the generated case folder that contains a site_resp_manifest.json in one of its subdirectories.");
  }
  const form = new FormData();
  for (const file of selected) {
    form.append("files", file, file.name);
    form.append("relative_paths", file.webkitRelativePath || file.name);
  }
  const response = await fetch("/api/protein-site-resp/upload", {
    method: "POST",
    headers: {"X-SIMPLE-Token": state.apiToken},
    body: form,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "Could not open the selected RESP case folder.");
  $("protein_site_resp_search_root").value = data.search_root || ".";
  $("protein_site_resp_job_dir").value = "";
  $("protein_site_resp_mode").value = "standard_ff";
  $("protein_site_resp_multiplicity_confirmed").checked = false;
  state.proteinSiteRespApproved = false;
  state.proteinSiteRespClusters = [];
  renderProteinSiteRespCandidates(data.candidates || []);
  const completed = (data.candidates || []).filter((item) => item.completed).length;
  setStatus(
    `Selected case folder scanned recursively: ${(data.candidates || []).length} RESP job(s), ${completed} completed.`,
    completed ? "ok" : "warn"
  );
}

function openProteinSiteRespDirectoryPicker() {
  $("protein_site_resp_directory_picker").click();
}

async function approveProteinSiteResp() {
  state.proteinSiteRespApproved = true;
  await finishWorkflow();
}

async function buildRespAssets() {
  if ($("charge_method").value === "antechamber") {
    throw new Error("AM1-BCC uses AmberTools/Antechamber during dry-run; no NWChem RESP assets are needed.");
  }
  setStatus("Building RESP input assets...", "warn");
  const data = await api("/api/resp/build-assets", collectPayload());
  const assets = data.assets || {};
  $("resp_job_dir").value = assets.job_dir || "";
  $("finish_note").textContent = `RESP assets: ${assets.job_dir || ""}`;
  setStatus(`RESP assets written: ${assets.job_dir}`, "ok");
}

async function exportMetallophorePdb() {
  setStatus("Writing metallophore PDB...", "warn");
  const data = await api("/api/metallophore/export-pdb", collectPayload());
  $("finish_note").textContent = `PDB output: ${data.path}`;
  setStatus(`PDB written: ${data.path}`, "ok");
}

async function scanRespFolders() {
  const data = await api("/api/resp/candidates", {search_root: $("resp_search_root").value.trim() || "."});
  renderRespCandidates(data.candidates || []);
  setStatus(`RESP scan found ${(data.candidates || []).length} candidate(s).`, "ok");
}

function activateTab(name) {
  for (const item of document.querySelectorAll(".tab")) item.classList.toggle("active", item.dataset.tab === name);
  for (const item of document.querySelectorAll(".tab-panel")) item.classList.toggle("active", item.id === `tab-${name}`);
  if (state.stage) setTimeout(() => state.stage.handleResize(), 50);
}

function setupEvents() {
  for (const btn of document.querySelectorAll(".tab")) btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  $("workflow_type").addEventListener("change", () => {
    clearPreview("Workflow changed. Load a new preview.");
    resetMetallophoreTables();
    syncWorkflowPanels();
  });
  $("met_mode").addEventListener("change", () => syncMetMode(true));
  $("des_size_mode").addEventListener("change", updateDesSizeModeUi);
  $("des_build_mode").addEventListener("change", updateDesBuildModeUi);
  $("des_metal_element").addEventListener("change", updateDesMetalCharges);
  $("des_mixing_enabled").addEventListener("change", updateMdProtocolUi);
  $("des_apply_1264").addEventListener("change", syncDes1264EnabledState);
  $("apply_1264_metals").addEventListener("change", () => syncSystem1264EnabledState(true));
  $("system_c4_parameter_set").addEventListener("change", () => syncSystemC4ParameterSet(true));
  $("water_model").addEventListener("change", () => syncSystemC4WaterWarning(true));
  $("library_browse_directory").addEventListener("click", openLibraryDirectoryPicker);
  $("library_choose_files").addEventListener("click", () => $("library_file_picker").click());
  $("library_scan").addEventListener("click", () => runBusy("Scanning for Amber library files...", scanLibraryDirectory).catch((err) => setStatus(err.message, "error")));
  $("library_scan_empty").addEventListener("click", openLibraryDirectoryPicker);
  $("library_directory_picker").addEventListener("change", (ev) => {
    const input = ev.currentTarget;
    runBusy("Opening the selected library directory...", () => uploadLibrarySelection(input.files, true))
      .catch((err) => setStatus(err.message, "error"))
      .finally(() => { input.value = ""; });
  });
  $("library_file_picker").addEventListener("change", (ev) => {
    const input = ev.currentTarget;
    runBusy("Opening the selected Amber library files...", () => uploadLibrarySelection(input.files, false))
      .catch((err) => setStatus(err.message, "error"))
      .finally(() => { input.value = ""; });
  });
  $("library_search_path").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      runBusy("Scanning for Amber library files...", scanLibraryDirectory).catch((err) => setStatus(err.message, "error"));
    }
  });
  $("library_add").addEventListener("click", () => addSelectedLibraryCandidate().catch((err) => setStatus(err.message, "error")));
  $("library_remove").addEventListener("click", () => removeSelectedLibraryComponent().catch((err) => setStatus(err.message, "error")));
  $("library_save_file").addEventListener("click", () => runBusy("Saving Amber library file...", saveLibraryFile).catch((err) => setStatus(err.message, "error")));
  $("add_des_metal_site").addEventListener("click", addDesMetalSite);
  $("met_insert_element").addEventListener("change", () => updateInsertionChargeControls("met_insert"));
  $("met_insert_charge").addEventListener("change", () => updateInsertionChargeControls("met_insert"));
  $("protein_insert_element").addEventListener("change", () => updateInsertionChargeControls("protein_insert"));
  $("protein_insert_charge").addEventListener("change", () => updateInsertionChargeControls("protein_insert"));
  $("charge_method").addEventListener("change", () => updateChargeMethodUi(true));
  $("net_charge").addEventListener("input", refreshRespChargeHint);
  $("multiplicity").addEventListener("input", refreshRespChargeHint);
  $("qm_geometry").addEventListener("change", updateQmGeometryUi);
  $("qm_resp_same").addEventListener("change", updateQmGeometryUi);
  $("salt_mode").addEventListener("change", updateSaltUi);
  $("neutralization_ion").addEventListener("change", updateSaltUi);
  $("md_protocol").addEventListener("change", updateMdProtocolUi);
  for (const id of ["temperature", "pressure", "production_ns"]) {
    $(id).addEventListener("change", renderMdStageEditors);
  }
  $("focused_enabled").addEventListener("change", syncFocusedRestraintUi);
  $("load_metallophore").addEventListener("click", () => runBusy("Loading metallophore preview...", loadMetallophore).catch((err) => setStatus(err.message, "error")));
  $("add_met_metal").addEventListener("click", () => runBusy("Adding metallophore metal...", addMetallophoreMetal).catch((err) => setStatus(err.message, "error")));
  $("remove_met_inserted_metal").addEventListener("click", () => runBusy("Removing inserted metallophore metal...", removeSelectedMetallophoreInsertedMetal).catch((err) => setStatus(err.message, "error")));
  $("quick_min").addEventListener("click", () => runBusy("Running quick Open Babel minimization...", quickMinimize).catch((err) => setStatus(err.message, "error")));
  $("restore_met").addEventListener("click", () => runBusy("Restoring initial metallophore geometry...", restoreMetallophoreGeometry).catch((err) => setStatus(err.message, "error")));
  $("build_resp_assets").addEventListener("click", () => runBusy("Building RESP input assets...", buildRespAssets).catch((err) => setStatus(err.message, "error")));
  $("export_met_pdb").addEventListener("click", () => runBusy("Writing metallophore PDB...", exportMetallophorePdb).catch((err) => setStatus(err.message, "error")));
  $("scan_resp").addEventListener("click", () => runBusy("Scanning RESP folders...", scanRespFolders).catch((err) => setStatus(err.message, "error")));
  $("scan_protein_site_resp").addEventListener("click", openProteinSiteRespDirectoryPicker);
  $("protein_site_resp_directory_picker").addEventListener("change", (ev) => {
    const input = ev.currentTarget;
    runBusy("Uploading and recursively scanning the selected RESP case folder...", () => uploadProteinSiteRespCase(input.files))
      .catch((err) => setStatus(err.message, "error"))
      .finally(() => { input.value = ""; });
  });
  $("apply_protein_site_resp").addEventListener("click", () => runBusy("Applying reviewed protein-site RESP charges...", approveProteinSiteResp).catch((err) => {
    state.proteinSiteRespApproved = false;
    setStatus(err.message, "error");
  }));
  $("load_protein").addEventListener("click", () => runBusy("Loading protein preview...", loadProtein).catch((err) => setStatus(err.message, "error")));
  $("protein_use_selected_residues").addEventListener("click", () => runBusy("Loading insertion donor candidates...", loadProteinInsertionDonors).catch((err) => setStatus(err.message, "error")));
  $("protein_preview_insert_metal").addEventListener("click", () => runBusy("Previewing inserted protein metal...", previewProteinInsertedMetal).catch((err) => setStatus(err.message, "error")));
  $("protein_clear_insertions").addEventListener("click", () => runBusy("Clearing inserted protein metals...", clearProteinInsertions).catch((err) => setStatus(err.message, "error")));
  $("run_propka").addEventListener("click", () => runBusy("Running PropKa...", runPropka).catch((err) => setStatus(err.message, "error")));
  $("build_des_preview").addEventListener("click", () => runBusy("Building DES heavy-atom preview...", buildDesPreview).catch((err) => setStatus(err.message, "error")));
  $("preview_solvation").addEventListener("click", () => runBusy("Building solvation preview...", previewSolvation).catch((err) => setStatus(err.message, "error")));
  $("finish").addEventListener("click", () => runBusy("Building final AMBER inputs...", finishWorkflow).catch((err) => {
    $("finish_note").textContent = `Build failed: ${err.message}`;
    setStatus(err.message, "error");
  }));
  $("quit_app").addEventListener("click", async () => {
    try {
      await shutdownGui();
    } catch (err) {
      setStatus(err.message, "error");
    }
  });
  $("met_file").addEventListener("change", () => uploadFile("met_file").then((file) => {
    $("met_input_path").value = file.path;
    $("met_smiles").value = "";
    $("met_file_name").textContent = `Selected: ${file.name}`;
    clearPreview("Molecule uploaded. Click Load Preview.");
    resetMetallophoreTables();
    setStatus(`Uploaded molecule: ${file.name}`, "ok");
  }).catch((err) => setStatus(err.message, "error")));
  $("met_smiles").addEventListener("input", () => {
    if ($("met_smiles").value.trim()) {
      $("met_input_path").value = "";
      $("met_file").value = "";
      $("met_file_name").textContent = "Using SMILES text.";
    }
  });
  $("upload_protein").addEventListener("click", () => runBusy("Uploading and loading protein preview...", async () => {
    const file = await uploadFile("protein_file");
    $("protein_input").value = file.path;
    $("protein_path").value = file.path;
    $("protein_pdb_id").value = "";
    $("protein_input_mode").value = "path";
    state.lastProteinMissingLoopAction = "";
    setStatus(`Uploaded PDB: ${file.name}`, "ok");
    await loadProtein();
  }).catch((err) => setStatus(err.message, "error")));
  $("protein_input").addEventListener("input", () => {
    syncProteinInputFields();
    state.lastProteinMissingLoopAction = "";
  });
  $("remove_waters").addEventListener("change", () => {
    if (state.proteinLoaded) runBusy("Reloading protein preview...", () => reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true})).catch((err) => setStatus(err.message, "error"));
  });
  $("remove_hetero").addEventListener("change", () => {
    if (!$("remove_hetero").checked) {
      setStatus("Non-standard molecules are retained; provide matching force-field parameters before production runs.", "warn");
    }
    if (state.proteinLoaded) runBusy("Reloading protein preview...", () => reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true})).catch((err) => setStatus(err.message, "error"));
  });
  $("kept_ligands").addEventListener("change", () => {
    if (state.proteinLoaded) runBusy("Reloading protein preview...", () => reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true})).catch((err) => setStatus(err.message, "error"));
  });
  $("protein_metal_all").addEventListener("click", () => {
    for (const row of document.querySelectorAll(".protein-metal-row")) {
      row.querySelector(".protein-metal-action").value = "replace";
      syncMetalActionRow(row);
    }
    if (state.proteinLoaded) runBusy("Reloading protein preview...", () => reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true})).catch((err) => setStatus(err.message, "error"));
  });
  $("protein_metal_clear").addEventListener("click", () => {
    state.proteinMetalActions = new Map();
    for (const row of document.querySelectorAll(".protein-metal-row")) {
      row.querySelector(".protein-metal-action").value = "keep";
      syncMetalActionRow(row);
    }
    if (state.proteinLoaded) runBusy("Reloading protein preview...", () => reloadProteinPreview({missingLoopAction: state.lastProteinMissingLoopAction, keepSelection: true})).catch((err) => setStatus(err.message, "error"));
  });
  $("propka_all").addEventListener("click", () => {
    for (const node of document.querySelectorAll(".propka-check")) {
      if (!node.disabled) node.checked = true;
    }
    syncPropkaSelections();
  });
  $("propka_none").addEventListener("click", () => {
    for (const node of document.querySelectorAll(".propka-check")) node.checked = false;
    syncPropkaSelections();
  });
  $("disulfide_all").addEventListener("click", () => {
    for (const node of document.querySelectorAll(".disulfide-check")) node.checked = true;
    syncDisulfideSelections();
  });
  $("disulfide_none").addEventListener("click", () => {
    for (const node of document.querySelectorAll(".disulfide-check")) node.checked = false;
    syncDisulfideSelections();
  });
  $("resp_group_mode").addEventListener("change", () => runBusy("Rebuilding RESP symmetry groups...", regenerateRespGroups).catch((err) => setStatus(err.message, "error")));
  $("resp_group_graph_method").addEventListener("change", () => {
    const label = $("resp_group_graph_method").selectedOptions?.[0]?.textContent || "selected graph method";
    runBusy(`Running ${label}...`, regenerateRespGroups).catch((err) => setStatus(err.message, "error"));
  });
  for (const id of ["show_cartoon", "show_labels", "show_lines", "show_sticks", "show_spacefill"]) {
    $(id).addEventListener("change", () => {
      rerenderCurrentScene().catch((err) => setStatus(err.message, "error"));
    });
  }
  for (const btn of document.querySelectorAll("[data-camera]")) {
    btn.addEventListener("click", () => btn.dataset.camera === "reset" ? resetCamera() : cameraAxis(btn.dataset.camera));
  }
  document.addEventListener("keydown", (ev) => {
    const tag = ev.target?.tagName || "";
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
    if (ev.key === "=" || ev.key === "+") resetCamera();
    if (ev.key.toLowerCase() === "x") cameraAxis("x");
    if (ev.key.toLowerCase() === "y") cameraAxis("y");
    if (ev.key.toLowerCase() === "z") cameraAxis("z");
  });
}

async function init() {
  ensureStage();
  installBusyObserver();
  const boot = await fetch("/api/bootstrap").then((r) => r.json());
  state.apiToken = String(boot.api_token || "");
  state.bootstrap = boot;
  replaceOptions($("workflow_type"), boot.workflow_options.map(([label, value]) => [value, label]), "metalloprotein");
  replaceOptions($("protein_ff"), boot.protein_force_fields, boot.protein_force_fields?.[0] || "ff19SB");
  replaceOptions($("ligand_ff"), boot.ligand_force_fields, "gaff2");
  replaceOptions($("water_model"), boot.water_models, boot.water_models?.includes("opc") ? "opc" : boot.water_models?.[0]);
  replaceOptions($("system_c4_parameter_set"), boot.c4_parameter_sets || [
    {key: "opc_duvail", label: "OPC + Duvail (default)"},
    {key: "spce_limerz", label: "SPC/E + Li/Merz"},
  ], "opc_duvail");
  syncSystemC4ParameterSet(false);
  syncSystem1264EnabledState(false);
  syncDes1264EnabledState();
  replaceOptions($("box_shape"), [["oct", "Oct"], ["cubic", "Cubic"]].filter(([value]) => (boot.box_shapes || ["oct", "cubic"]).includes(value)), "oct");
  replaceOptions($("salt_kind"), boot.salt_kinds, "none");
  replaceOptions($("salt_mode"), boot.salt_modes || ["none", "neutralize", "count", "concentration"], "none");
  replaceOptions($("md_protocol"), [...boot.md_protocols, "des_solvent"], "15step");
  replaceOptions($("slurm_profile"), boot.slurm_profiles, "gpu");
  replaceOptions($("charge_method"), [
    ["resp_antechamber", "RESP"],
    ["antechamber", "Antechamber"],
  ], "resp_antechamber");
  replaceOptions($("qm_geometry"), boot.qm.geometry_modes, "use_loaded_geometry");
  replaceOptions($("qm_functional"), boot.qm.functionals, "r2scan");
  replaceOptions($("qm_basis"), boot.qm.basis, "def2-tzvp");
  replaceOptions($("qm_resp_functional"), boot.qm.functionals, "hf");
  replaceOptions($("qm_resp_basis"), boot.qm.basis, "6-31g*");
  replaceOptions($("qm_grid"), boot.qm.grids, "fine");
  replaceOptions($("resp_group_mode"), boot.resp_group_modes || [
    ["hydrogen_and_symmetry", "H + Symmetry"],
    ["hydrogen_only", "H Only"],
  ], "hydrogen_and_symmetry");
  replaceOptions($("resp_group_graph_method"), boot.resp_group_graph_methods || [
    ["connectivity", "Connectivity (fast)"],
    ["graph_automorphism", "Exact graph symmetry"],
  ], "connectivity");
  replaceOptions($("met_insert_element"), boot.supported_metals.map((m) => [m.element, m.element]), "Fe");
  replaceOptions($("protein_insert_element"), boot.supported_metals.map((m) => [m.element, m.element]), "Fe");
  updateInsertionChargeControls("met_insert");
  updateInsertionChargeControls("protein_insert");
  replaceOptions($("des_metal_element"), boot.supported_metals.map((m) => [m.element, m.element]), "Fe");
  updateDesMetalCharges();
  renderDesMetalSites();
  updateDesSizeModeUi();
  updateDesBuildModeUi();
  updateSaltUi();
  updateQmGeometryUi();
  refreshRespChargeHint();
  updateMdProtocolUi();
  syncFocusedRestraintUi();
  syncViewerControls();
  state.desComponents = boot.des_components || [];
  renderDesComponents();
  for (const item of boot.des_recommended_sets || []) {
    const btn = document.createElement("button");
    btn.textContent = `${item.key}: ${item.label} (${item.ratios.join(":")})`;
    btn.addEventListener("click", () => applyDesRecommended(item));
    $("des_recommended").appendChild(btn);
  }
  if ((boot.des_recommended_sets || [])[0]) applyDesRecommended(boot.des_recommended_sets[0]);
  setupEvents();
  syncWorkflowPanels();
  syncMetMode(false);
  setStatus("Ready.", "ok");
}

init().catch((err) => setStatus(err.message, "error"));
