/**
 * map.js – Sweden map rendering on HTML5 Canvas.
 *
 * Renders a stylised outline of Sweden + a sparse grid,
 * plus all simulation nodes, edges, threats, and engagement effects.
 *
 * DISCLAIMER: All data is fictional. Educational/research use only.
 */

'use strict';

// ============================================================
// Coordinate projection
// ============================================================

const MAP_BOUNDS = {
  latMin: 55.0, latMax: 69.5,
  lonMin: 10.5, lonMax: 24.5,
};

function project(lat, lon, canvasW, canvasH, padding = 40) {
  const { latMin, latMax, lonMin, lonMax } = MAP_BOUNDS;
  const x = padding + ((lon - lonMin) / (lonMax - lonMin)) * (canvasW - 2 * padding);
  // Invert lat (north = top)
  const y = padding + ((latMax - lat) / (latMax - latMin)) * (canvasH - 2 * padding);
  return { x, y };
}

// ============================================================
// Sweden simplified outline (approximate lat/lon waypoints)
// ============================================================

const SWEDEN_OUTLINE = [
  // South tip → east coast → north → west coast → back
  [55.35, 13.85], [55.60, 14.10], [55.90, 14.30], [56.20, 15.60],
  [56.50, 16.20], [57.00, 16.80], [57.30, 17.00], [57.80, 18.80],
  [58.30, 17.40], [58.80, 17.90], [59.20, 18.60], [59.60, 18.80],
  [60.00, 18.40], [60.50, 17.60], [61.00, 17.30], [61.40, 17.30],
  [62.20, 17.90], [63.00, 18.40], [63.50, 19.80], [64.00, 20.80],
  [65.00, 21.70], [65.50, 22.10], [66.00, 23.00], [67.00, 23.50],
  [67.90, 20.30], [68.40, 18.00], [68.30, 16.00], [67.70, 14.80],
  [67.00, 15.50], [66.50, 14.00], [65.80, 14.30], [65.30, 14.10],
  [64.50, 13.80], [63.50, 12.50], [62.60, 12.20], [61.50, 12.20],
  [60.50, 12.10], [59.60, 11.80], [58.80, 11.10], [58.40, 11.50],
  [57.80, 11.60], [57.50, 11.90], [56.80, 12.20], [56.20, 12.80],
  [55.80, 12.90], [55.35, 13.85], // close
];

// Gotland island
const GOTLAND_OUTLINE = [
  [57.35, 18.20], [57.60, 18.10], [57.90, 18.20], [58.00, 18.50],
  [57.90, 18.80], [57.60, 18.90], [57.35, 18.70], [57.25, 18.50], [57.35, 18.20],
];

// ============================================================
// Node visual config
// ============================================================

const NODE_VISUAL = {
  airbase:      { symbol: '▲', color: '#00e5ff', size: 10 },
  air_defense:  { symbol: '◆', color: '#00e5ff', size: 10 },
  radar:        { symbol: '●', color: '#00ff88', size: 8  },
  fuel_depot:   { symbol: '■', color: '#ffd700', size: 7  },
  ammo_depot:   { symbol: '■', color: '#ff8c00', size: 7  },
  c2:           { symbol: '★', color: '#00e5ff', size: 12 },
  civil_infra:  { symbol: '□', color: '#8aa8cc', size: 7  },
};

const EDGE_COLORS = {
  command_link: 'rgba(0, 229, 255, 0.35)',
  sensor_link:  'rgba(0, 255, 136, 0.30)',
  logistics:    'rgba(255, 215, 0,   0.25)',
};

// ============================================================
// MapRenderer class
// ============================================================

class MapRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');

    this.graphData   = null;
    this.threats     = [];
    this.engResults  = [];

    // Flash effects: Map<nodeId, {endTime, type}>
    this._flashes    = new Map();
    // Engagement beams: [{from: {x,y}, to: {x,y}, endTime, hit}]
    this._beams      = [];

    // Display toggles
    this.showRadarRings  = true;
    this.showEdges       = true;
    this.showLogistics   = false;
    this.showSensorLinks = true;

    this._animFrame = null;
    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const parent = this.canvas.parentElement;
    this.canvas.width  = parent.clientWidth;
    this.canvas.height = parent.clientHeight;
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  updateGraph(graphData) {
    this.graphData = graphData;
  }

  updateThreats(threats) {
    this.threats = threats || [];
  }

  addEngagementResults(results) {
    if (!results) return;
    const now = performance.now();
    for (const r of results) {
      if (!this.graphData) continue;
      // Beam from each engaging AD node to threat last position
      for (const adId of (r.engaged_by || [])) {
        const adNode = this.graphData.nodes[adId];
        if (!adNode) continue;
        const p1 = this._proj(adNode.lat, adNode.lon);
        // Approximate threat position as target node position
        const targetNode = r.hit_node_id ? this.graphData.nodes[r.hit_node_id] : null;
        if (!targetNode) continue;
        const p2 = this._proj(targetNode.lat, targetNode.lon);
        this._beams.push({ from: p1, to: p2, endTime: now + 600, hit: !r.intercepted && !r.miss });
      }
      // Flash on hit
      if (r.hit_node_id && !r.miss) {
        this._flashes.set(r.hit_node_id, { endTime: now + 800, type: 'hit' });
      }
    }
  }

  startAnimation() {
    if (this._animFrame) return;
    const loop = () => {
      this._draw();
      this._animFrame = requestAnimationFrame(loop);
    };
    this._animFrame = requestAnimationFrame(loop);
  }

  stopAnimation() {
    if (this._animFrame) {
      cancelAnimationFrame(this._animFrame);
      this._animFrame = null;
    }
  }

  // ------------------------------------------------------------------
  // Projection helper
  // ------------------------------------------------------------------

  _proj(lat, lon) {
    return project(lat, lon, this.canvas.width, this.canvas.height, 48);
  }

  // ------------------------------------------------------------------
  // Drawing
  // ------------------------------------------------------------------

  _draw() {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    const now = performance.now();

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#050d1a';
    ctx.fillRect(0, 0, W, H);

    this._drawGrid(ctx, W, H);
    this._drawSwedenOutline(ctx);

    if (this.graphData) {
      if (this.showEdges || this.showLogistics || this.showSensorLinks) {
        this._drawEdges(ctx);
      }
      if (this.showRadarRings) {
        this._drawRadarRings(ctx, now);
      }
      this._drawEngagementBeams(ctx, now);
      this._drawNodes(ctx, now);
    }

    this._drawThreats(ctx, now);
    this._cleanupExpired(now);
  }

  _drawGrid(ctx, W, H) {
    ctx.strokeStyle = 'rgba(0, 80, 120, 0.3)';
    ctx.lineWidth   = 0.5;
    const step = 60;
    for (let x = step; x < W; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0); ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = step; y < H; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(W, y);
      ctx.stroke();
    }
    // Lat/lon labels
    ctx.fillStyle = 'rgba(0, 80, 120, 0.5)';
    ctx.font = '9px JetBrains Mono, monospace';
    for (let lat = 56; lat <= 68; lat += 4) {
      const { x, y } = this._proj(lat, MAP_BOUNDS.lonMin + 0.3);
      ctx.fillText(`${lat}°N`, x, y);
    }
  }

  _drawSwedenOutline(ctx) {
    this._drawLandmass(ctx, SWEDEN_OUTLINE, 'rgba(0, 60, 100, 0.25)', 'rgba(0, 120, 180, 0.6)');
    this._drawLandmass(ctx, GOTLAND_OUTLINE, 'rgba(0, 60, 100, 0.25)', 'rgba(0, 120, 180, 0.6)');
  }

  _drawLandmass(ctx, coords, fill, stroke) {
    if (coords.length < 2) return;
    ctx.beginPath();
    const p0 = this._proj(coords[0][0], coords[0][1]);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < coords.length; i++) {
      const p = this._proj(coords[i][0], coords[i][1]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  _drawEdges(ctx) {
    if (!this.graphData?.edges) return;
    for (const [eid, edge] of Object.entries(this.graphData.edges)) {
      if (!edge.is_intact) {
        // Broken edge → dashed red
        ctx.strokeStyle = 'rgba(255, 30, 60, 0.5)';
        ctx.setLineDash([4, 6]);
        ctx.lineWidth = 1;
      } else {
        const show =
          (edge.type === 'logistics'    && this.showLogistics)   ||
          (edge.type === 'sensor_link'  && this.showSensorLinks) ||
          (edge.type === 'command_link' && this.showEdges);
        if (!show) continue;
        ctx.strokeStyle = EDGE_COLORS[edge.type] || 'rgba(100,100,100,0.3)';
        ctx.setLineDash([]);
        ctx.lineWidth = edge.type === 'command_link' ? 1.2 : 0.8;
      }

      const sNode = this.graphData.nodes[edge.source];
      const tNode = this.graphData.nodes[edge.target];
      if (!sNode || !tNode) continue;

      const p1 = this._proj(sNode.lat, sNode.lon);
      const p2 = this._proj(tNode.lat, tNode.lon);

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  _drawRadarRings(ctx, now) {
    if (!this.graphData?.nodes) return;
    const pulse = 0.5 + 0.5 * Math.sin(now / 1200);
    for (const [nid, node] of Object.entries(this.graphData.nodes)) {
      if (node.type !== 'radar' || !node.is_functional) continue;
      if (!node.coverage_radius_km) continue;

      const p = this._proj(node.lat, node.lon);
      // Scale radius: rough conversion from km to canvas px
      const kmPx = this._kmToPx(node.coverage_radius_km * node.effective_capacity);

      // Outer ring
      ctx.beginPath();
      ctx.arc(p.x, p.y, kmPx, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(0, 255, 136, ${0.12 + 0.06 * pulse})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([8, 12]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Scan beam
      const angle = ((now / 1000) * Math.PI * 2 / 4) % (Math.PI * 2); // 4s per rotation
      ctx.save();
      ctx.translate(p.x, p.y);
      const grad = ctx.createConicalGradient
        ? ctx.createConicalGradient(0, 0, angle)
        : null;
      // Fallback: draw a thin line beam
      ctx.rotate(angle);
      ctx.strokeStyle = `rgba(0, 255, 136, ${0.25 * node.effective_capacity})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(kmPx, 0);
      ctx.stroke();
      ctx.restore();
    }
  }

  _drawNodes(ctx, now) {
    if (!this.graphData?.nodes) return;
    for (const [nid, node] of Object.entries(this.graphData.nodes)) {
      const p     = this._proj(node.lat, node.lon);
      const vis   = NODE_VISUAL[node.type] || NODE_VISUAL.civil_infra;
      const flash = this._flashes.get(nid);
      const isFlashing = flash && flash.endTime > now;

      // Health ring
      const health = node.health || 0;
      const ringColor = health > 0.6 ? '#00ff88'
                      : health > 0.3 ? '#ffd700' : '#ff2244';
      ctx.beginPath();
      ctx.arc(p.x, p.y, vis.size / 2 + 4, -Math.PI / 2,
              -Math.PI / 2 + health * Math.PI * 2);
      ctx.strokeStyle = ringColor;
      ctx.lineWidth   = 2;
      ctx.stroke();

      // AD coverage ring
      if (node.type === 'air_defense' && node.range_km && node.is_functional) {
        const r = this._kmToPx(node.range_km * node.effective_capacity);
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.18)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([6, 10]);
        ctx.stroke();
        ctx.setLineDash([]);
        // Filled aura
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 229, 255, 0.04)';
        ctx.fill();
      }

      // Node icon
      ctx.font      = `${vis.size}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      if (isFlashing) {
        ctx.shadowColor = '#ff2244';
        ctx.shadowBlur  = 20;
        ctx.fillStyle   = '#ff8888';
      } else if (!node.is_functional) {
        ctx.fillStyle = '#334455';
        ctx.shadowBlur = 0;
      } else {
        ctx.shadowColor = vis.color;
        ctx.shadowBlur  = 6;
        ctx.fillStyle   = vis.color;
      }
      ctx.fillText(vis.symbol, p.x, p.y);
      ctx.shadowBlur = 0;

      // Node ID label
      ctx.font      = '8px JetBrains Mono, monospace';
      ctx.fillStyle = node.is_functional ? 'rgba(180,220,255,0.7)' : 'rgba(80,100,120,0.5)';
      ctx.fillText(nid, p.x, p.y + vis.size / 2 + 8);
    }
  }

  _drawThreats(ctx, now) {
    for (const threat of this.threats) {
      if (!threat.active) continue;
      const p = this._proj(threat.current_lat, threat.current_lon);

      // Threat track line toward target
      if (this.graphData?.nodes[threat.target_node_id]) {
        const tNode = this.graphData.nodes[threat.target_node_id];
        const tp    = this._proj(tNode.lat, tNode.lon);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(tp.x, tp.y);
        ctx.strokeStyle = 'rgba(255, 30, 60, 0.25)';
        ctx.lineWidth   = 0.8;
        ctx.setLineDash([3, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Threat dot (blinking)
      const blink = Math.sin(now / 300) > 0;
      if (blink) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle   = '#ff2244';
        ctx.shadowColor = '#ff2244';
        ctx.shadowBlur  = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      } else {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#ff8888';
        ctx.fill();
      }

      // ETA label
      ctx.font      = '8px JetBrains Mono, monospace';
      ctx.fillStyle = '#ff8888';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`ETA:${threat.eta}`, p.x + 6, p.y - 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
    }
  }

  _drawEngagementBeams(ctx, now) {
    for (const beam of this._beams) {
      if (beam.endTime < now) continue;
      const frac = 1 - (beam.endTime - now) / 600;
      const alpha = 1 - frac;
      ctx.beginPath();
      ctx.moveTo(beam.from.x, beam.from.y);
      ctx.lineTo(beam.to.x, beam.to.y);
      ctx.strokeStyle = beam.hit
        ? `rgba(255, 30, 60, ${alpha})`
        : `rgba(0, 229, 255, ${alpha})`;
      ctx.lineWidth = 2;
      ctx.shadowColor = beam.hit ? '#ff2244' : '#00e5ff';
      ctx.shadowBlur  = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  _cleanupExpired(now) {
    for (const [id, f] of this._flashes) {
      if (f.endTime < now) this._flashes.delete(id);
    }
    this._beams = this._beams.filter(b => b.endTime > now);
  }

  // ------------------------------------------------------------------
  // Geometry helpers
  // ------------------------------------------------------------------

  _kmToPx(km) {
    // Rough conversion: map spans ~14° lat ≈ 1560 km over canvas height
    const degSpan   = MAP_BOUNDS.latMax - MAP_BOUNDS.latMin;
    const kmPerDeg  = 111.0;
    const kmSpan    = degSpan * kmPerDeg;
    const pxPerKm   = (this.canvas.height - 96) / kmSpan;
    return km * pxPerKm;
  }

  // ------------------------------------------------------------------
  // Click hit-test
  // ------------------------------------------------------------------

  hitTest(canvasX, canvasY) {
    if (!this.graphData?.nodes) return null;
    let closest = null, minDist = 18;
    for (const [nid, node] of Object.entries(this.graphData.nodes)) {
      const p = this._proj(node.lat, node.lon);
      const d = Math.hypot(canvasX - p.x, canvasY - p.y);
      if (d < minDist) { minDist = d; closest = nid; }
    }
    return closest;
  }

  getNodeCanvasPos(nid) {
    if (!this.graphData?.nodes[nid]) return null;
    const node = this.graphData.nodes[nid];
    return this._proj(node.lat, node.lon);
  }
}

// Export as global
window.MapRenderer = MapRenderer;
