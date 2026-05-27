/**
 * c2.js – Main C2 display controller.
 *
 * Connects to the FastAPI WebSocket, updates the map and all HUD panels
 * on each state message received.
 *
 * DISCLAIMER: All simulation data is fictional. Educational/research use only.
 */

'use strict';

// ============================================================
// Constants
// ============================================================

const WS_URL      = `ws://${location.host}/ws`;
const API_BASE    = `http://${location.host}`;
const RECONNECT_DELAY_MS = 3000;

const NODE_TYPE_ICONS = {
  airbase:      '▲',
  air_defense:  '◆',
  radar:        '●',
  fuel_depot:   '■',
  ammo_depot:   '■',
  c2:           '★',
  civil_infra:  '□',
};

// ============================================================
// State
// ============================================================

let ws = null;
let mapRenderer = null;
let graphData   = null;
let episodeLen  = 150;
let stepCount   = 0;
let isPaused    = false;
let selectedNode = null;

// ============================================================
// Initialisation
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  mapRenderer = new MapRenderer('map-canvas');
  mapRenderer.startAnimation();

  _initControls();
  _initClock();
  _connectWS();
  _loadInitialGraph();

  // Show disclaimer briefly
  setTimeout(() => {
    const d = document.getElementById('disclaimer');
    d.classList.add('show');
    setTimeout(() => d.classList.remove('show'), 6000);
  }, 1000);
});

// ============================================================
// WebSocket
// ============================================================

function _connectWS() {
  setConnStatus('connecting');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setConnStatus('connected');
    logEvent('SYSTEM: WebSocket connected', 'info');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      _handleMessage(msg);
    } catch (e) {
      console.warn('WS parse error:', e);
    }
  };

  ws.onclose = () => {
    setConnStatus('disconnected');
    logEvent('SYSTEM: Connection lost – reconnecting…', 'info');
    setTimeout(_connectWS, RECONNECT_DELAY_MS);
  };

  ws.onerror = (err) => {
    console.error('WS error:', err);
  };
}

function _handleMessage(msg) {
  if (msg.type === 'episode_start') {
    _onEpisodeStart(msg);
  } else if (msg.type === 'state') {
    _onStateUpdate(msg);
  } else if (msg.type === 'episode_end') {
    _onEpisodeEnd(msg);
  } else if (msg.type === 'pong') {
    // heartbeat
  }
}

// ============================================================
// Episode handling
// ============================================================

function _onEpisodeStart(msg) {
  graphData  = msg.graph;
  episodeLen = msg.config?.episode_length || 150;
  stepCount  = 0;
  mapRenderer.updateGraph(graphData);
  logEvent(`EPISODE ${msg.episode_id} START`, 'info');
  _updateUnitList(graphData);
}

function _onStateUpdate(msg) {
  stepCount = msg.step;
  graphData = msg.graph || graphData;

  // Update map
  mapRenderer.updateGraph(graphData);
  mapRenderer.updateThreats(msg.threats || []);
  mapRenderer.addEngagementResults(msg.engagement_results || []);

  // HUD metrics
  _setMetric('metric-defense', msg.defense_ratio, v => `${(v * 100).toFixed(1)}%`,
             v => v < 0.4 ? 'alert' : v < 0.7 ? 'warn' : '');
  _setMetric('metric-stop-rate', msg.last_stop_rate, v => `${(v * 100).toFixed(0)}%`,
             v => v < 0.4 ? 'warn' : '');
  _setMetric('metric-threats', msg.active_threats, v => String(v),
             v => v > 6 ? 'alert' : v > 3 ? 'warn' : '');
  _setMetric('metric-sorties', msg.sortie_pool, v => String(v), () => '');

  // Timeline
  const pct = Math.min(100, (stepCount / episodeLen) * 100);
  document.getElementById('timeline-fill').style.width  = pct + '%';
  document.getElementById('timeline-label').textContent =
    `T+${String(stepCount).padStart(4,'0')} / ${episodeLen}`;

  // Event log
  _processEventLog(msg.event_log || [], msg.engagement_results || [], msg.cascade_events || []);

  // Unit list update
  if (graphData) _updateUnitList(graphData);

  // Selected node popup
  if (selectedNode && graphData?.nodes[selectedNode]) {
    _showNodePopup(selectedNode, graphData.nodes[selectedNode]);
  }
}

function _onEpisodeEnd(msg) {
  logEvent(
    `EPISODE ${msg.episode_id} END │ Defense: ${(msg.defense_ratio * 100).toFixed(1)}% │ Steps: ${msg.steps}`,
    'info'
  );
}

// ============================================================
// Event log
// ============================================================

const _seenEvents = new Set();

function _processEventLog(events, engResults, cascadeEvents) {
  // Raw simulator event strings
  for (const ev of events) {
    if (_seenEvents.has(ev)) continue;
    _seenEvents.add(ev);
    const type = ev.includes('INTERCEPTED') ? 'intercept'
               : ev.includes('HIT')         ? 'hit'
               : ev.includes('MISS')        ? 'miss'
               : ev.includes('Wave')        ? 'wave'
               : ev.includes('CASCADE')     ? 'cascade' : 'info';
    logEvent(ev, type);
  }
}

function logEvent(text, type = 'info') {
  const log  = document.getElementById('event-log');
  const div  = document.createElement('div');
  div.className = `log-entry ${type}`;
  div.textContent = text;
  log.appendChild(div);
  // Auto-scroll
  log.scrollTop = log.scrollHeight;
  // Trim old entries
  while (log.children.length > 200) log.removeChild(log.firstChild);
}

// ============================================================
// Unit list (left panel)
// ============================================================

function _updateUnitList(graph) {
  const list = document.getElementById('unit-list');
  // Sort by importance descending
  const nodes = Object.values(graph.nodes)
    .sort((a, b) => b.importance - a.importance);

  list.innerHTML = '';
  for (const node of nodes) {
    const div    = document.createElement('div');
    div.className = 'unit-entry' + (node.id === selectedNode ? ' selected' : '');
    div.dataset.id = node.id;

    const health = node.health;
    const hColor = health > 0.6 ? '#00ff88' : health > 0.3 ? '#ffd700' : '#ff2244';
    const icon   = NODE_TYPE_ICONS[node.type] || '?';

    div.innerHTML = `
      <span class="unit-icon" style="color:${hColor}">${icon}</span>
      <span class="unit-name">${node.id}</span>
      <div class="unit-health-bar">
        <div class="unit-health-fill"
             style="width:${(health * 100).toFixed(0)}%; background:${hColor}">
        </div>
      </div>`;

    div.addEventListener('click', () => _selectNode(node.id));
    list.appendChild(div);
  }
}

// ============================================================
// Node selection / popup
// ============================================================

function _selectNode(nid) {
  selectedNode = nid;
  if (!graphData?.nodes[nid]) return;
  const node = graphData.nodes[nid];
  _showNodePopup(nid, node);
  _updateUnitList(graphData);
}

function _showNodePopup(nid, node) {
  const popup = document.getElementById('node-popup');
  const pos   = mapRenderer.getNodeCanvasPos(nid);
  if (!pos) { popup.classList.remove('visible'); return; }

  popup.querySelector('h3').textContent = node.id;
  const rows = popup.querySelector('.popup-rows');
  rows.innerHTML = `
    <div class="popup-row"><span class="popup-label">NAME</span><span class="popup-val">${node.name || nid}</span></div>
    <div class="popup-row"><span class="popup-label">TYPE</span><span class="popup-val">${node.type}</span></div>
    <div class="popup-row"><span class="popup-label">HEALTH</span><span class="popup-val" style="color:${_healthColor(node.health)}">${(node.health * 100).toFixed(1)}%</span></div>
    <div class="popup-row"><span class="popup-label">EFF.CAP</span><span class="popup-val">${(node.effective_capacity * 100).toFixed(1)}%</span></div>
    <div class="popup-row"><span class="popup-label">IMPORTANCE</span><span class="popup-val">${node.importance.toFixed(2)}</span></div>
    <div class="popup-row"><span class="popup-label">PROTECTION</span><span class="popup-val">${(node.protection_level * 100).toFixed(0)}%</span></div>
    ${node.range_km ? `<div class="popup-row"><span class="popup-label">AD RANGE</span><span class="popup-val">${node.range_km} km</span></div>` : ''}
    ${node.coverage_radius_km ? `<div class="popup-row"><span class="popup-label">RADAR RANGE</span><span class="popup-val">${node.coverage_radius_km} km</span></div>` : ''}
  `;

  const canvas = document.getElementById('map-canvas');
  const rect   = canvas.getBoundingClientRect();
  let px = rect.left + pos.x + 14;
  let py = rect.top  + pos.y - 20;
  // Keep in viewport
  if (px + 170 > window.innerWidth)  px = rect.left + pos.x - 175;
  if (py + 180 > window.innerHeight) py = rect.top  + pos.y - 185;

  popup.style.left = px + 'px';
  popup.style.top  = py + 'px';
  popup.classList.add('visible');
}

// Click on canvas: hit-test nodes
document.addEventListener('click', (e) => {
  const canvas = document.getElementById('map-canvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (e.clientX < rect.left || e.clientX > rect.right ||
      e.clientY < rect.top  || e.clientY > rect.bottom) {
    // Click outside map → close popup
    document.getElementById('node-popup').classList.remove('visible');
    selectedNode = null;
    if (graphData) _updateUnitList(graphData);
    return;
  }
  const cx  = e.clientX - rect.left;
  const cy  = e.clientY - rect.top;
  const nid = mapRenderer.hitTest(cx, cy);
  if (nid) {
    _selectNode(nid);
  } else {
    document.getElementById('node-popup').classList.remove('visible');
    selectedNode = null;
  }
});

// ============================================================
// Controls
// ============================================================

function _initControls() {
  document.getElementById('btn-play').addEventListener('click', () => {
    _apiPost('/api/control', { cmd: 'play', agent: 'rule' });
    isPaused = false;
    _updatePlayPauseBtn();
  });
  document.getElementById('btn-pause').addEventListener('click', () => {
    if (isPaused) {
      _apiPost('/api/control', { cmd: 'resume' });
      isPaused = false;
    } else {
      _apiPost('/api/control', { cmd: 'pause' });
      isPaused = true;
    }
    _updatePlayPauseBtn();
  });
  document.getElementById('btn-reset').addEventListener('click', () => {
    _seenEvents.clear();
    document.getElementById('event-log').innerHTML = '';
    _apiPost('/api/control', { cmd: 'reset', agent: 'rule' });
    isPaused = false;
    _updatePlayPauseBtn();
  });

  const speedSlider = document.getElementById('speed-slider');
  const speedLabel  = document.getElementById('speed-label');
  speedSlider.addEventListener('input', () => {
    const spd = parseFloat(speedSlider.value);
    speedLabel.textContent = `${spd.toFixed(1)}×`;
    _apiPost('/api/control', { cmd: 'set_speed', speed: spd });
  });

  // Map toggles
  _initToggle('toggle-edges',    () => mapRenderer.showEdges       = !mapRenderer.showEdges);
  _initToggle('toggle-radar',    () => mapRenderer.showRadarRings  = !mapRenderer.showRadarRings);
  _initToggle('toggle-logistics',() => mapRenderer.showLogistics   = !mapRenderer.showLogistics);
  _initToggle('toggle-sensors',  () => mapRenderer.showSensorLinks = !mapRenderer.showSensorLinks);
}

function _initToggle(id, cb) {
  const btn = document.getElementById(id);
  if (!btn) return;
  btn.addEventListener('click', () => {
    btn.classList.toggle('on');
    cb();
  });
}

function _updatePlayPauseBtn() {
  const btn = document.getElementById('btn-pause');
  btn.textContent = isPaused ? '▶ RESUME' : '⏸ PAUSE';
  btn.classList.toggle('active', isPaused);
}

// ============================================================
// Metrics
// ============================================================

function _setMetric(id, value, formatter, classFn) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = formatter(value);
  el.className   = 'metric-value ' + classFn(value);
}

// ============================================================
// Clock
// ============================================================

function _initClock() {
  const clockEl = document.getElementById('clock');
  function tick() {
    const now = new Date();
    clockEl.textContent =
      now.toISOString().slice(0, 19).replace('T', ' ') + ' UTC';
    setTimeout(tick, 1000);
  }
  tick();
}

// ============================================================
// Connection status
// ============================================================

function setConnStatus(status) {
  const el = document.getElementById('conn-status');
  el.className  = `conn-status ${status}`;
  el.textContent = {
    connected:    '● CONNECTED',
    disconnected: '● OFFLINE',
    connecting:   '● CONNECTING',
  }[status] || status;
}

// ============================================================
// API helpers
// ============================================================

async function _apiPost(url, body) {
  try {
    await fetch(API_BASE + url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    console.warn('API error:', e);
  }
}

async function _loadInitialGraph() {
  try {
    const res  = await fetch(API_BASE + '/api/graph');
    const data = await res.json();
    // If nodes are an array (initial file) convert to dict
    if (Array.isArray(data.nodes)) {
      const nodeMap = {};
      for (const n of data.nodes) nodeMap[n.id] = { ...n, is_functional: true };
      graphData = { nodes: nodeMap, edges: {} };
    } else {
      graphData = data;
    }
    mapRenderer.updateGraph(graphData);
    _updateUnitList(graphData);
  } catch (e) {
    console.warn('Could not load initial graph:', e);
  }
}

// ============================================================
// Utilities
// ============================================================

function _healthColor(h) {
  return h > 0.6 ? '#00ff88' : h > 0.3 ? '#ffd700' : '#ff2244';
}
