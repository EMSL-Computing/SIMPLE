let bridge = null;
let stage = null;
let component = null;
let currentViewerMode = null;
let loadedMol2Preview = null;
let loadedRepresentationKey = null;
let baseRepresentation = null;
let metalRepresentation = null;
let metalLabelRepresentation = null;
let selectedRepresentation = null;
let pendingRenderState = null;
let renderFrameHandle = 0;
let renderInFlight = false;
let lastViewportClickToken = 0;
let lastHandledPickToken = 0;
let lastClickAdditive = false;

function readEmbeddedInitialState() {
  const element = document.getElementById("resp-initial-state");
  if (!element) {
    return null;
  }
  try {
    return JSON.parse(element.textContent || "null");
  } catch (error) {
    console.error("RESP viewer could not parse the embedded initial state", error);
    return null;
  }
}

let currentState = readEmbeddedInitialState();

const GROUP_COLORS = [
  0x2563eb,
  0xdc2626,
  0x059669,
  0xd97706,
  0x7c3aed,
  0xdb2777,
  0x0891b2,
  0x65a30d,
  0xea580c,
  0x4338ca,
];
const METAL_ELEMENTS = new Set(["CO", "CU", "NI", "MN", "FE", "Y", "LA", "ND", "EU", "LU"]);
const METAL_COLORS = {
  FE: 0x7f1d1d,
  CO: 0x6d28d9,
  CU: 0x92400e,
  NI: 0x14532d,
  MN: 0x581c87,
  Y: 0x0f766e,
  LA: 0x1d4ed8,
  ND: 0x4338ca,
  EU: 0x9f1239,
  LU: 0x0f172a,
};
const ELEMENT_COLORS = {
  H: 0xffffff,
  C: 0x8b95a1,
  N: 0x2563eb,
  O: 0xef4444,
  S: 0xfacc15,
  P: 0xf97316,
  F: 0x22c55e,
  CL: 0x16a34a,
  BR: 0xb45309,
  I: 0x7c3aed,
  B: 0xf59e0b,
};

function elementKey(atom) {
  return String((atom && (atom.element || atom.name)) || "").trim().toUpperCase();
}

function isMetalAtom(atom) {
  return METAL_ELEMENTS.has(elementKey(atom));
}

function metalColor(atom) {
  return METAL_COLORS[elementKey(atom)] || 0x0f172a;
}

function elementColor(atom) {
  const key = elementKey(atom);
  if (METAL_ELEMENTS.has(key)) {
    return metalColor(atom);
  }
  return ELEMENT_COLORS[key] || 0xcbd5e1;
}

function groupColor(groupId) {
  if (!groupId) {
    return 0x94a3b8;
  }
  return GROUP_COLORS[(groupId - 1) % GROUP_COLORS.length];
}

function colorHex(groupId) {
  return `#${groupColor(groupId).toString(16).padStart(6, "0")}`;
}

function atomColorHex(atom, groupId) {
  const value = groupId ? groupColor(groupId) : elementColor(atom);
  return `#${value.toString(16).padStart(6, "0")}`;
}

function updateLegend(state) {
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  const groups = (state.group_constraints && state.group_constraints.groups) || [];
  const selectedAtoms = new Set((state.selected_atom_indices || []).map((value) => Number(value)));
  if (!groups.length) {
    legend.innerHTML = "<div class='muted'>No equality groups defined.</div>";
    return;
  }
  groups.forEach((group) => {
    const row = document.createElement("div");
    row.className = "legend-row";
    if (bridge && typeof bridge.groupPicked === "function") {
      row.classList.add("clickable");
      row.tabIndex = 0;
      row.addEventListener("click", () => bridge.groupPicked(group.group_id));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          bridge.groupPicked(group.group_id);
        }
      });
    }
    const allSelected = group.atom_indices.every((atomIndex) => selectedAtoms.has(Number(atomIndex)));
    if (allSelected) {
      row.classList.add("active");
    }
    const chip = document.createElement("span");
    chip.className = "legend-chip";
    chip.style.backgroundColor = `#${groupColor(group.group_id).toString(16).padStart(6, "0")}`;
    const label = document.createElement("span");
    label.textContent = `Group ${group.group_id}: ${group.label || "custom"} (${group.atom_indices.join(", ")})`;
    row.appendChild(chip);
    row.appendChild(label);
    legend.appendChild(row);
  });
}

function buildColorScheme(state) {
  const atomGroupIds = {};
  const atoms = (state.group_constraints && state.group_constraints.atoms) || [];
  atoms.forEach((atom) => {
    atomGroupIds[atom.index] = atom.group_id;
  });
  return NGL.ColormakerRegistry.addScheme(function () {
    this.atomColor = function (atom) {
      return groupColor(atomGroupIds[atom.index + 1]);
    };
  });
}

function updateSummary(state) {
  const atomCount = (((state || {}).molecule || {}).atoms || []).length;
  const summary = document.getElementById("summary");
  summary.textContent = `${state.residue_name || "LIG"} | ${atomCount} atoms | ${(state.qm_settings || {}).label || "custom QM settings"}`;
}

function updateSelection(state) {
  const selection = document.getElementById("selection");
  const selected = (state.selected_atom_indices || []).map((value) => Number(value));
  if (!selected.length) {
    selection.textContent = "Click an atom to select it, Ctrl+click to add atoms, or click empty space to clear.";
    return;
  }
  selection.textContent = `${selected.length} ${selected.length === 1 ? "atom" : "atoms"} highlighted in the molecule view.`;
}

function selectionForAtoms(atomIndices) {
  const zeroBased = atomIndices
    .map((value) => Number(value) - 1)
    .filter((value) => Number.isFinite(value) && value >= 0);
  if (!zeroBased.length) {
    return "";
  }
  return `@${zeroBased.join(",")}`;
}

function metalAtomIndices(state) {
  const atoms = (((state || {}).molecule || {}).atoms || []);
  return atoms.filter((atom) => isMetalAtom(atom)).map((atom) => Number(atom.index));
}

function representationKey(state) {
  const atoms = (((state || {}).group_constraints || {}).atoms || []);
  const groups = atoms.map((atom) => `${Number(atom.index)}:${atom.group_id || ""}`).join("|");
  return `${state.mol2_preview || ""}::${groups}`;
}

function additiveFromEvent(event) {
  return Boolean(event && (event.ctrlKey || event.metaKey));
}

function resetRepresentationHandles() {
  loadedRepresentationKey = null;
  baseRepresentation = null;
  metalRepresentation = null;
  metalLabelRepresentation = null;
  selectedRepresentation = null;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function clearViewport() {
  const viewport = document.getElementById("viewport");
  if (!viewport) {
    return null;
  }
  viewport.innerHTML = "";
  viewport.classList.remove("fallback-mode");
  return viewport;
}

function atomScreenPosition(atom, bounds, size, padding) {
  const width = Math.max(1, size.width - (padding * 2));
  const height = Math.max(1, size.height - (padding * 2));
  const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
  const scale = Math.min(width / spanX, height / spanY);
  const offsetX = (size.width - (spanX * scale)) / 2;
  const offsetY = (size.height - (spanY * scale)) / 2;
  return {
    x: offsetX + ((atom.x - bounds.minX) * scale),
    y: size.height - (offsetY + ((atom.y - bounds.minY) * scale)),
    z: Number(atom.z || 0),
  };
}

function viewportSize(viewport) {
  const rect = viewport.getBoundingClientRect();
  return {
    width: Math.max(Math.round(rect.width), 360),
    height: Math.max(Math.round(rect.height), 360),
  };
}

function renderFallbackState(state, message) {
  const viewport = clearViewport();
  if (!viewport) {
    return;
  }
  viewport.classList.add("fallback-mode");
  currentViewerMode = "svg_fallback";

  const wrapper = document.createElement("div");
  wrapper.className = "viewport-fallback";

  if (message) {
    const banner = document.createElement("div");
    banner.className = "viewport-fallback-banner";
    banner.textContent = message;
    wrapper.appendChild(banner);
  }

  const molecule = (state && state.molecule) || {};
  const atoms = Array.isArray(molecule.atoms) ? molecule.atoms : [];
  const bonds = Array.isArray(molecule.bonds) ? molecule.bonds : [];
  if (!atoms.length) {
    const empty = document.createElement("div");
    empty.className = "viewport-fallback-empty";
    empty.textContent = "No atom coordinates were available for the molecule preview.";
    wrapper.appendChild(empty);
    viewport.appendChild(wrapper);
    return;
  }

  const selectedAtoms = new Set((state.selected_atom_indices || []).map((value) => Number(value)));
  const atomGroups = new Map();
  const groupAtoms = (((state || {}).group_constraints || {}).atoms || []);
  groupAtoms.forEach((atom) => {
    atomGroups.set(Number(atom.index), atom.group_id ? Number(atom.group_id) : null);
  });

  const size = viewportSize(viewport);
  const padding = 48;
  const bounds = atoms.reduce(
    (acc, atom) => ({
      minX: Math.min(acc.minX, Number(atom.x || 0)),
      maxX: Math.max(acc.maxX, Number(atom.x || 0)),
      minY: Math.min(acc.minY, Number(atom.y || 0)),
      maxY: Math.max(acc.maxY, Number(atom.y || 0)),
    }),
    { minX: Number.POSITIVE_INFINITY, maxX: Number.NEGATIVE_INFINITY, minY: Number.POSITIVE_INFINITY, maxY: Number.NEGATIVE_INFINITY },
  );

  const positionedAtoms = atoms.map((atom) => ({
    atom,
    ...atomScreenPosition(atom, bounds, size, padding),
    groupId: atomGroups.get(Number(atom.index)) || null,
    selected: selectedAtoms.has(Number(atom.index)),
    isMetal: isMetalAtom(atom),
  }));

  const byIndex = new Map(positionedAtoms.map((entry) => [Number(entry.atom.index), entry]));
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "viewport-fallback-svg");
  svg.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${state.residue_name || "Ligand"} molecule preview`);
  if (bridge && typeof bridge.clearSelection === "function") {
    svg.addEventListener("click", () => bridge.clearSelection());
  }

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const shadow = document.createElementNS("http://www.w3.org/2000/svg", "filter");
  shadow.setAttribute("id", "atomShadow");
  shadow.innerHTML = "<feDropShadow dx='0' dy='3' stdDeviation='4' flood-color='#0f172a' flood-opacity='0.15'/>";
  defs.appendChild(shadow);
  svg.appendChild(defs);

  bonds.forEach((bond) => {
    const first = byIndex.get(Number(bond.first));
    const second = byIndex.get(Number(bond.second));
    if (!first || !second) {
      return;
    }
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(first.x));
    line.setAttribute("y1", String(first.y));
    line.setAttribute("x2", String(second.x));
    line.setAttribute("y2", String(second.y));
    line.setAttribute("stroke", "#64748b");
    line.setAttribute("stroke-width", String(Math.max(2, Number(bond.order || 1) * 1.5)));
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("opacity", "0.9");
    svg.appendChild(line);
  });

  positionedAtoms
    .sort((left, right) => left.z - right.z)
    .forEach((entry) => {
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", entry.selected ? "atom-group selected" : "atom-group");
      if (bridge && typeof bridge.atomPicked === "function") {
        group.style.cursor = "pointer";
        group.addEventListener("click", (event) => {
          event.stopPropagation();
          bridge.atomPicked(Number(entry.atom.index), additiveFromEvent(event));
        });
      }

      if (entry.selected) {
        const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        halo.setAttribute("cx", String(entry.x));
        halo.setAttribute("cy", String(entry.y));
        halo.setAttribute("r", entry.isMetal ? "40" : "20");
        halo.setAttribute("fill", "#fbbf24");
        halo.setAttribute("opacity", "0.42");
        group.appendChild(halo);
      }

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      const atomRadius = (entry.selected ? 13 : 11) * (entry.isMetal ? 2.4 : 1);
      circle.setAttribute("cx", String(entry.x));
      circle.setAttribute("cy", String(entry.y));
      circle.setAttribute("r", String(atomRadius));
      circle.setAttribute("fill", entry.groupId ? colorHex(entry.groupId) : atomColorHex(entry.atom, entry.groupId));
      circle.setAttribute("stroke", entry.selected ? "#fbbf24" : "rgba(226, 232, 240, 0.65)");
      circle.setAttribute("stroke-width", entry.selected ? "3.4" : "1.2");
      circle.setAttribute("filter", "url(#atomShadow)");
      group.appendChild(circle);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(entry.x));
      label.setAttribute("y", String(entry.y + 4));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("class", "atom-label");
      label.textContent = entry.isMetal ? elementKey(entry.atom) : String(entry.atom.element || entry.atom.name || entry.atom.index);
      group.appendChild(label);

      if (entry.isMetal) {
        const serial = document.createElementNS("http://www.w3.org/2000/svg", "text");
        serial.setAttribute("x", String(entry.x));
        serial.setAttribute("y", String(entry.y + atomRadius + 12));
        serial.setAttribute("text-anchor", "middle");
        serial.setAttribute("class", "metal-serial-label");
        serial.textContent = `${entry.atom.name || entry.atom.element} (${entry.atom.index})`;
        group.appendChild(serial);
      }

      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = `${entry.atom.name || entry.atom.element || "Atom"} (${entry.atom.index})`;
      group.appendChild(title);
      svg.appendChild(group);
    });

  wrapper.appendChild(svg);

  const caption = document.createElement("div");
  caption.className = "viewport-fallback-caption";
  caption.innerHTML = `${escapeHtml(state.residue_name || "LIG")} projected in 2D for this session. Atom colors still reflect RESP equality groups.`;
  wrapper.appendChild(caption);

  viewport.appendChild(wrapper);
}

function handlePick(pickingProxy) {
  if (!bridge) {
    return;
  }
  lastHandledPickToken = lastViewportClickToken;
  if (!pickingProxy || !pickingProxy.atom) {
    if (typeof bridge.clearSelection === "function") {
      bridge.clearSelection();
    }
    return;
  }
  const atomIndex = pickingProxy.atom.index + 1;
  bridge.atomPicked(atomIndex, lastClickAdditive);
}

function initViewerClickTracking() {
  const viewport = document.getElementById("viewport");
  if (!viewport || viewport.dataset.respClickTracking === "1") {
    return;
  }
  viewport.dataset.respClickTracking = "1";
  viewport.addEventListener("mousedown", (event) => {
    lastViewportClickToken += 1;
    lastClickAdditive = additiveFromEvent(event);
  }, true);
  viewport.addEventListener("click", () => {
    const clickToken = lastViewportClickToken;
    window.setTimeout(() => {
      if (
        bridge
        && typeof bridge.clearSelection === "function"
        && clickToken !== 0
        && lastHandledPickToken < clickToken
      ) {
        bridge.clearSelection();
      }
    }, 80);
  }, true);
}

function initLayoutResize() {
  const layout = document.getElementById("layout");
  const divider = document.getElementById("viewport-divider");
  if (!layout || !divider) {
    return;
  }

  let dragging = false;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function applySidebarWidth(clientX) {
    const rect = layout.getBoundingClientRect();
    const nextWidth = clamp(rect.right - clientX, 200, Math.min(480, rect.width - 320));
    document.documentElement.style.setProperty("--resp-sidebar-width", `${nextWidth}px`);
    if (stage && typeof stage.handleResize === "function") {
      stage.handleResize();
    }
  }

  divider.addEventListener("mousedown", (event) => {
    dragging = true;
    divider.classList.add("dragging");
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragging) {
      return;
    }
    applySidebarWidth(event.clientX);
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) {
      return;
    }
    dragging = false;
    divider.classList.remove("dragging");
  });

  divider.addEventListener("dblclick", () => {
    document.documentElement.style.setProperty("--resp-sidebar-width", "240px");
    if (stage && typeof stage.handleResize === "function") {
      stage.handleResize();
    }
  });
}

async function flushScheduledRender() {
  if (renderInFlight) {
    return;
  }
  renderInFlight = true;
  try {
    while (pendingRenderState) {
      const nextState = pendingRenderState;
      pendingRenderState = null;
      await renderState(nextState);
    }
  } finally {
    renderInFlight = false;
  }
}

function scheduleRenderState(state) {
  pendingRenderState = state;
  if (renderFrameHandle) {
    return;
  }
  renderFrameHandle = window.requestAnimationFrame(() => {
    renderFrameHandle = 0;
    void flushScheduledRender();
  });
}

async function renderState(state) {
  currentState = state;
  updateLegend(state);
  updateSummary(state);
  updateSelection(state);
  if (!window.NGL) {
    renderFallbackState(
      state,
      "The local NGL 3D viewer bundle was unavailable in this session, so a built-in 2D molecule preview is being shown instead. Check that ngl.js is present in the editor assets directory.",
    );
    return;
  }
  try {
    if (!stage) {
      clearViewport();
      stage = new NGL.Stage("viewport", { backgroundColor: "#050814" });
      stage.signals.clicked.add(handlePick);
      initViewerClickTracking();
      window.addEventListener("resize", () => {
        if (stage && typeof stage.handleResize === "function") {
          stage.handleResize();
        }
      }, false);
      window.addEventListener("keydown", (event) => {
        if ((event.key === "=" || event.key === "+") && component) {
          event.preventDefault();
          component.autoView(350);
        }
      });
    }
    const previewText = state.mol2_preview || "";
    const canReuseComponent = (
      component
      && loadedMol2Preview === previewText
      && typeof component.removeAllRepresentations === "function"
    );
    if (component && !canReuseComponent) {
      stage.removeComponent(component);
      component = null;
      resetRepresentationHandles();
    }
    if (!canReuseComponent) {
      const blob = new Blob([previewText], { type: "text/plain" });
      component = await stage.loadFile(blob, { ext: "mol2" });
      loadedMol2Preview = previewText;
      resetRepresentationHandles();
    }
    const nextRepresentationKey = representationKey(state);
    if (loadedRepresentationKey !== nextRepresentationKey) {
      const schemeId = buildColorScheme(state);
      component.removeAllRepresentations();
      baseRepresentation = component.addRepresentation("ball+stick", {
        color: schemeId,
        multipleBond: "symmetric",
        radiusScale: 0.74,
        bondScale: 0.30,
        roughness: 0.45,
      });
      const metalSelection = selectionForAtoms(metalAtomIndices(state));
      if (metalSelection) {
        metalRepresentation = component.addRepresentation("spacefill", {
          sele: metalSelection,
          color: "element",
          radiusScale: 1.70,
        });
        metalLabelRepresentation = component.addRepresentation("label", {
          sele: metalSelection,
          labelType: "element",
          labelSize: 1.65,
          color: "white",
          zOffset: 1.4,
        });
      } else {
        metalRepresentation = null;
        metalLabelRepresentation = null;
      }
      selectedRepresentation = null;
      loadedRepresentationKey = nextRepresentationKey;
    }
    if (selectedRepresentation && typeof component.removeRepresentation === "function") {
      component.removeRepresentation(selectedRepresentation);
      selectedRepresentation = null;
    } else if (selectedRepresentation && typeof selectedRepresentation.dispose === "function") {
      selectedRepresentation.dispose();
      selectedRepresentation = null;
    }
    const selectedSelection = selectionForAtoms(state.selected_atom_indices || []);
    if (selectedSelection) {
      selectedRepresentation = component.addRepresentation("spacefill", {
        sele: selectedSelection,
        color: "uniform",
        colorValue: 0xfbbf24,
        radiusScale: 1.85,
        opacity: 0.58,
        transparent: true,
      });
    }
    if (!canReuseComponent) {
      component.autoView();
    } else if (stage && typeof stage.handleResize === "function") {
      stage.handleResize();
    }
    currentViewerMode = "ngl";
  } catch (error) {
    console.error("RESP viewer fell back to SVG preview", error);
    component = null;
    loadedMol2Preview = null;
    resetRepresentationHandles();
    if (stage && typeof stage.dispose === "function") {
      stage.dispose();
    }
    stage = null;
    renderFallbackState(
      state,
      "The 3D viewer could not render in this Qt session, so a built-in 2D molecule preview is being shown instead. Check QtWebEngine, the Qt bridge, and Linux DISPLAY/WAYLAND_DISPLAY settings.",
    );
  }
}

function initQtBridge() {
  if (typeof QWebChannel !== "function") {
    throw new Error("QWebChannel was not available in the Qt WebEngine page.");
  }
  if (!window.qt || !window.qt.webChannelTransport) {
    throw new Error("Qt WebChannel transport was not available.");
  }
  new QWebChannel(window.qt.webChannelTransport, (channel) => {
    bridge = channel.objects.bridge;
    if (!bridge || typeof bridge.getStateJson !== "function") {
      throw new Error("Qt bridge object was not registered.");
    }
    scheduleRenderState(JSON.parse(bridge.getStateJson()));
    if (bridge.stateChanged && typeof bridge.stateChanged.connect === "function") {
      bridge.stateChanged.connect((payload) => {
        scheduleRenderState(JSON.parse(payload));
      });
    }
  });
}

function initFromAvailableState() {
  if (currentState) {
    scheduleRenderState(currentState);
    return true;
  }
  return false;
}

function reportViewerIssue(message) {
  const summary = document.getElementById("summary");
  if (summary) {
    summary.textContent = message;
  }
}

function startViewer() {
  const renderedInitialState = initFromAvailableState();
  if (window.qt && window.qt.webChannelTransport) {
    try {
      initQtBridge();
      return;
    } catch (error) {
      console.error("RESP Qt bridge initialization failed", error);
      if (renderedInitialState) {
        reportViewerIssue("The Qt bridge did not fully initialize in this session, but the embedded molecule preview is available. If this persists, reinstall PySide6 QtWebEngine and check Linux DISPLAY/WAYLAND_DISPLAY settings.");
        return;
      }
      renderFallbackState(
        currentState || { molecule: { atoms: [], bonds: [] }, selected_atom_indices: [] },
        "The Qt bridge failed to initialize in this session, so the interactive viewer could not be started. Reinstall PySide6 QtWebEngine, restart the popup, and on Linux verify DISPLAY or WAYLAND_DISPLAY is set.",
      );
      return;
    }
  }

  if (renderedInitialState) {
    reportViewerIssue("Using the embedded molecule preview because the Qt bridge was unavailable in this session. If the local 3D viewer is expected, check PySide6 QtWebEngine and Linux display forwarding.");
    return;
  }

  renderFallbackState(
    currentState || { molecule: { atoms: [], bonds: [] }, selected_atom_indices: [] },
    "The Qt bridge was unavailable in this session, so the interactive viewer could not be started. Check PySide6 QtWebEngine installation and Linux DISPLAY/WAYLAND_DISPLAY availability.",
  );
}

document.addEventListener("DOMContentLoaded", () => {
  initLayoutResize();
  startViewer();
});
