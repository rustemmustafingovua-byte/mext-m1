/* MEXT-M1 dashboard — zero dependencies, zero CDN.
 * Polls /api/snapshot and renders three canvas charts with a 240-sample ring. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const MAX_SAMPLES = 240;

  const state = { history: [], snap: null, online: false, ragBusy: false };
  // `?poll=1` forces plain polling: handy behind proxies and for headless
  // screenshots, where an open SSE stream blocks "network idle".
  const FORCE_POLL = new URLSearchParams(window.location.search).has('poll');
  // Server-sent events primary, fast polling as a fallback.
  const stream = { last: 0, pollTimer: null, source: null };

  // ---------------------------------------------------------------- format
  const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  function fmtBytes(bytes) {
    let n = Number(bytes) || 0;
    if (n <= 0) return '0 B';
    let i = 0;
    while (n >= 1024 && i < UNITS.length - 1) { n /= 1024; i++; }
    const digits = n >= 100 ? 0 : n >= 10 ? 1 : 2;
    return `${n.toFixed(digits)} ${UNITS[i]}`;
  }
  function fmtPct(x, d = 0) { return `${((Number(x) || 0) * 100).toFixed(d)}%`; }
  function fmtNum(x) {
    const n = Number(x) || 0;
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return `${Math.round(n)}`;
  }
  function fmtAgo(ts) {
    const s = Math.max(0, Date.now() / 1000 - (Number(ts) || 0));
    if (s < 60) return `${Math.round(s)}s`;
    if (s < 3600) return `${Math.round(s / 60)}m`;
    return `${(s / 3600).toFixed(1)}h`;
  }
  function hexA(hex, alpha) {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
    const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    return `rgba(${r},${g},${b},${alpha})`;
  }
  function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
  }
  function setBar(id, ratio) {
    const el = $(id);
    if (el) el.style.width = `${clamp(ratio, 0, 1) * 100}%`;
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // ---------------------------------------------------------------- canvas
  class Chart {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.pad = { l: 76, r: 14, t: 12, b: 22 };
      this.hoverX = null;
      this._resize();
      window.addEventListener('resize', () => { this._resize(); renderAll(); });
      if (typeof ResizeObserver !== 'undefined') {
        this._observer = new ResizeObserver(() => { this._resize(); renderAll(); });
        this._observer.observe(canvas);
      }
      canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        this.hoverX = e.clientX - rect.left;
        renderAll();
      });
      canvas.addEventListener('mouseleave', () => { this.hoverX = null; renderAll(); });
    }
    _resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      this.w = Math.max(20, rect.width);
      this.h = Math.max(20, rect.height);
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    frame() {
      this.ctx.clearRect(0, 0, this.w, this.h);
      return {
        x0: this.pad.l,
        x1: this.w - this.pad.r,
        y0: this.h - this.pad.b,
        y1: this.pad.t,
      };
    }
  }

  const charts = {};

  function axis(chart, maxY, fmtY, steps = 4) {
    const { ctx } = chart;
    const { x0, x1, y0, y1 } = chart.frame();
    ctx.save();
    ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const y = y1 + (y0 - y1) * t;
      ctx.strokeStyle = i === steps ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.055)';
      ctx.beginPath();
      ctx.moveTo(x0, y + 0.5);
      ctx.lineTo(x1, y + 0.5);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.34)';
      ctx.fillText(fmtY(maxY * (1 - t)), x0 - 8, y);
    }
    ctx.restore();
    return { x0, x1, y0, y1 };
  }

  function emptyNote(chart, text) {
    const { ctx } = chart;
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.28)';
    ctx.font = '11.5px -apple-system, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, chart.w / 2, chart.h / 2);
    ctx.restore();
  }

  /* ---------- chart 1: physical RAM stack + virtual volume ---------- */
  function renderRam() {
    const chart = charts.ram;
    if (!chart) return;
    const hist = state.history;
    const snap = state.snap;
    if (!snap || hist.length === 0) { emptyNote(chart, 'waiting for samples…'); return; }

    const installed = snap.memory.total || hist[0].total || 1;
    let usedMax = 0;
    let virtualMax = 0;
    for (const p of hist) {
      usedMax = Math.max(usedMax, p.mext + p.app + p.wired + p.comp);
      virtualMax = Math.max(virtualMax, p.virtual);
    }
    // Zoom on the data: showing all 8 GiB of an idle machine would flatten the
    // composition to a 3 px band. Installed/used totals are in the panel header.
    const maxY = Math.max(usedMax, virtualMax, 2 * 1024 * 1024) * 1.28;

    const { x0, x1, y0, y1 } = axis(chart, maxY, fmtBytes);
    const { ctx } = chart;
    const n = hist.length;
    const X = (i) => (n <= 1 ? x1 : x0 + ((x1 - x0) * i) / (n - 1));
    const Y = (v) => y0 - (y0 - y1) * clamp(v / maxY, 0, 1);

    const layers = [
      { key: 'mext', color: '#34d399' },
      { key: 'app', color: '#60a5fa' },
      { key: 'wired', color: '#a78bfa' },
      { key: 'comp', color: '#f472b6' },
    ];

    let lower = new Array(n).fill(0);
    for (const layer of layers) {
      const upper = hist.map((p, i) => lower[i] + (p[layer.key] || 0));
      const grad = ctx.createLinearGradient(0, y1, 0, y0);
      grad.addColorStop(0, hexA(layer.color, 0.42));
      grad.addColorStop(1, hexA(layer.color, 0.12));

      ctx.beginPath();
      ctx.moveTo(X(0), Y(upper[0]));
      for (let i = 1; i < n; i++) ctx.lineTo(X(i), Y(upper[i]));
      for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(i), Y(lower[i]));
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(X(0), Y(upper[0]));
      for (let i = 1; i < n; i++) ctx.lineTo(X(i), Y(upper[i]));
      ctx.strokeStyle = hexA(layer.color, 0.75);
      ctx.lineWidth = 1.2;
      ctx.stroke();
      lower = upper;
    }

    // Virtual volume (compressed pool logical bytes) — dashed amber.
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(X(0), Y(hist[0].virtual));
    for (let i = 1; i < n; i++) ctx.lineTo(X(i), Y(hist[i].virtual));
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = 'rgba(251,191,36,0.45)';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.restore();

    // Physics ceiling, when it fits the zoomed axis.
    if (installed <= maxY) {
      ctx.save();
      ctx.setLineDash([2, 5]);
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.beginPath();
      ctx.moveTo(x0, Y(installed));
      ctx.lineTo(x1, Y(installed));
      ctx.stroke();
      ctx.restore();
    }

    // Hover crosshair + tooltip.
    const tip = $('ram-tip');
    if (chart.hoverX != null && chart.hoverX >= x0 && chart.hoverX <= x1) {
      const idx = clamp(Math.round(((chart.hoverX - x0) / (x1 - x0)) * (n - 1)), 0, n - 1);
      const p = hist[idx];
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.22)';
      ctx.beginPath();
      ctx.moveTo(X(idx), y1);
      ctx.lineTo(X(idx), y0);
      ctx.stroke();
      ctx.fillStyle = '#fbbf24';
      ctx.beginPath();
      ctx.arc(X(idx), Y(p.virtual), 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      if (tip) {
        const physical = p.mext + p.app + p.wired + p.comp;
        tip.innerHTML =
          `<b>${fmtBytes(physical)}</b> <span class="k">physical</span><br>` +
          `<span class="k">mext</span> ${fmtBytes(p.mext)} · <span class="k">app</span> ${fmtBytes(p.app)}<br>` +
          `<span class="k">wired</span> ${fmtBytes(p.wired)} · <span class="k">compressor</span> ${fmtBytes(p.comp)}<br>` +
          `<span class="k">virtual</span> ${fmtBytes(p.virtual)} · ${fmtAgo(p.t)} ago`;
        tip.style.left = `${clamp(X(idx), 90, chart.w - 90)}px`;
        tip.style.top = `${Math.max(56, Y(physical))}px`;
        tip.classList.add('on');
      }
    } else if (tip) {
      tip.classList.remove('on');
    }
  }

  /* ------------------- chart 2: prediction hit rate ------------------- */
  function smooth(ctx, pts) {
    if (pts.length < 2) return;
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      const [px, py] = pts[i - 1];
      const [cx, cy] = pts[i];
      const mx = (px + cx) / 2;
      ctx.bezierCurveTo(mx, py, mx, cy, cx, cy);
    }
  }

  function renderHit() {
    const chart = charts.hit;
    if (!chart) return;
    const hist = state.history;
    if (!hist.length) { emptyNote(chart, 'learning…'); return; }

    const { x0, x1, y0, y1 } = axis(chart, 1, (v) => `${Math.round(v * 100)}%`);
    const { ctx } = chart;
    const n = hist.length;
    const X = (i) => (n <= 1 ? x1 : x0 + ((x1 - x0) * i) / (n - 1));
    const Y = (v) => y0 - (y0 - y1) * clamp(v, 0, 1);

    // 60% target line
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = 'rgba(110,231,183,0.3)';
    ctx.beginPath();
    ctx.moveTo(x0, Y(0.6));
    ctx.lineTo(x1, Y(0.6));
    ctx.stroke();
    ctx.restore();

    const pts = hist.map((p, i) => [X(i), Y(p.hit)]);
    const grad = ctx.createLinearGradient(0, y1, 0, y0);
    grad.addColorStop(0, 'rgba(52,211,153,0.38)');
    grad.addColorStop(1, 'rgba(52,211,153,0.02)');

    ctx.beginPath();
    smooth(ctx, pts);
    ctx.lineTo(X(n - 1), y0);
    ctx.lineTo(X(0), y0);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    smooth(ctx, pts);
    ctx.strokeStyle = '#6ee7b7';
    ctx.lineWidth = 1.7;
    ctx.shadowColor = 'rgba(52,211,153,0.5)';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    const last = hist[n - 1];
    ctx.beginPath();
    ctx.arc(X(n - 1), Y(last.hit), 3, 0, Math.PI * 2);
    ctx.fillStyle = '#6ee7b7';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(X(n - 1), Y(last.hit), 7, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(110,231,183,0.18)';
    ctx.fill();
  }

  /* ---------------------- chart 3: SSD throughput ---------------------- */
  function renderSsd() {
    const chart = charts.ssd;
    if (!chart) return;
    const hist = state.history;
    if (!hist.length) { emptyNote(chart, 'idle — no SSD traffic yet'); return; }

    let peak = 0;
    for (const p of hist) peak = Math.max(peak, p.ssd);
    // Floor of 4 MB/s keeps the axis stable and avoids duplicate integer labels
    // when the machine is idle.
    const maxY = Math.max(4, peak * 1.24);

    const { x0, x1, y0, y1 } = axis(chart, maxY, (v) =>
      v >= 100 ? `${Math.round(v)}` : v >= 10 ? v.toFixed(0) : v.toFixed(1)
    );
    const { ctx } = chart;
    const n = hist.length;
    const step = (x1 - x0) / Math.max(1, n - 1);
    const barW = Math.max(1.6, Math.min(6, step * 0.62));
    const Y = (v) => y0 - (y0 - y1) * clamp(v / maxY, 0, 1);

    const grad = ctx.createLinearGradient(0, y0, 0, y1);
    grad.addColorStop(0, 'rgba(56,189,248,0.95)');
    grad.addColorStop(1, 'rgba(56,189,248,0.18)');

    for (let i = 0; i < n; i++) {
      const v = hist[i].ssd;
      if (v <= 0.05) continue;
      const x = x0 + i * step;
      const h = Math.max(1.5, (y0 - Y(v)));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(x - barW / 2, y0 - h, barW, h, 1.5);
      ctx.fill();
    }

    // Trend line over the bars (moving average).
    if (n > 2) {
      const pts = hist.map((p, i) => [x0 + i * step, Y(p.ssd)]);
      ctx.beginPath();
      smooth(ctx, pts);
      ctx.strokeStyle = 'rgba(125,211,252,0.85)';
      ctx.lineWidth = 1.1;
      ctx.stroke();
    }

    ctx.save();
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(125,211,252,0.8)';
    ctx.textAlign = 'right';
    ctx.fillText(`peak ${maxY.toFixed(0)} MB/s`, x1, y1 + 10);
    ctx.restore();
  }

  function renderAll() {
    renderRam();
    renderHit();
    renderSsd();
  }

  // ---------------------------------------------------------------- cards
  function updateCards(s) {
    const m = s.memory || {}, p = s.pool || {}, pred = s.predictor || {}, act = s.activity || {};
    const pressure = s.pressure || {};

    setText('pool-budget', `cap ${fmtBytes(p.budget_bytes)}`);
    setText('pool-resident', fmtBytes(p.resident_bytes));
    setText('pool-ratio', p.ratio ? `${p.ratio.toFixed(2)}× lz4` : '—');
    setBar('pool-bar', (p.resident_bytes || 0) / Math.max(1, p.budget_bytes || s.effective_budget || 1));
    setText('pool-files', `${fmtNum(p.files)} files`);
    setText('pool-evict', `${fmtNum(p.evictions)} evicted`);
    setText(
      'pool-saved',
      `${p.arena_pages ? `arena ${p.arena_pages}p · ` : ''}saved ${fmtBytes(p.saved_bytes)}`
    );

    setText('virtual-bytes', fmtBytes(p.logical_bytes));
    setText('virtual-saved', `${p.ratio ? p.ratio.toFixed(1) : '—'}× vs raw`);
    setBar('virtual-bar', (p.logical_bytes || 0) / Math.max(1, m.total || 1));
    setText('virtual-decomp', `${fmtNum(p.decompressions)} decompressions`);
    setText('virtual-released', `${fmtBytes(p.released_bytes)} released`);

    setText('hit-rate', fmtPct(pred.hit_rate, 1));
    setText('hit-count', `${fmtNum(pred.hits)} / ${fmtNum(pred.attempts)} hits`);
    setBar('hit-bar', pred.hit_rate || 0);
    setText('hit-edges', `${fmtNum(pred.edges)} edges`);
    setText('hit-obs', `${fmtNum(pred.observations)} observations`);
    setText('hit-chip', fmtPct(pred.hit_rate));

    setText('ssd-mbps', `${(p.ssd_mbps || 0).toFixed(1)} MB/s`);
    const lat = s.latency || {};
    const warmP95 = Number(lat.warm_p95_ms) || 0;
    setText(
      'ssd-warm',
      `${(act.warm_avg_ms || 0).toFixed(1)} ms avg · p95 ${warmP95 < 10 ? warmP95.toFixed(1) : warmP95.toFixed(0)} ms`
    );
    setBar('ssd-bar', (p.ssd_mbps || 0) / Math.max(40, Math.max(...state.history.map((x) => x.ssd), 1)));
    setText('ssd-warms', `${fmtNum(act.warms)} warms`);
    setText('swap-used', `swap ${fmtBytes(pressure.swap_used)}`);

    const level = (pressure.level || 'unknown').toLowerCase();
    const pill = $('pressure-pill');
    if (pill) {
      pill.classList.remove('good', 'warn', 'bad');
      pill.classList.add(level === 'green' ? 'good' : level === 'yellow' ? 'warn' : level === 'red' ? 'bad' : '');
    }
    setText('pressure-text', `pressure ${level} · ${fmtBytes(m.available)} free`);

    const ai = s.ai || {};
    const neuralOk = !!(s.neural && s.neural.available);
    const ragOk = !!(s.rag && s.rag.available);
    const aiPill = $('ai-pill');
    if (aiPill) {
      aiPill.classList.remove('good', 'warn', 'bad');
      aiPill.classList.add(neuralOk && ragOk ? 'good' : ai.reachable || neuralOk || ragOk ? 'warn' : 'bad');
    }
    setText('ai-text', `ai ${neuralOk || ragOk ? 'online' : ai.reachable ? 'starting' : 'offline'}`);

    setText('workspace', s.workspace || '—');
    setText(
      'footer-note',
      `MEXT-M1 · profile ${s.profile || 'default'} · one CLI, three predictive layers`
    );
    const up = Number(s.uptime_s) || 0;
    setText(
      'uptime',
      up < 60
        ? `up ${Math.round(up)}s`
        : `up ${Math.floor(up / 3600)}h ${String(Math.floor((up % 3600) / 60)).padStart(2, '0')}m`
    );
    setText('version', `v${s.version || '—'}`);
    setText('event-count', `${(s.events || []).length} recent`);
    const used = (m.app || 0) + (m.wired || 0) + (m.compressor_occupied || 0) + (p.resident_bytes || 0);
    setText(
      'ram-hint',
      `installed ${fmtBytes(m.total)} · used ${fmtBytes(used)} · MEXT cap ${fmtBytes(p.budget_bytes)} · virtual ${fmtBytes(p.logical_bytes)}`
    );
  }

  // ---------------------------------------------------------------- neural
  function updateNeural(s) {
    const n = s.neural || {};
    const chip = $('neural-chip');
    if (!n.available) {
      if (chip) { chip.textContent = 'offline'; chip.className = 'chip amber'; }
      setText('n-model', s.model || '—');
      setText('n-note', n.error || (s.ai && s.ai.error) || 'AI engine not running — start with ./setup.sh && mext start');
      setBar('n-bar', 0);
      return;
    }
    if (chip) { chip.textContent = n.mode || 'mlx'; chip.className = 'chip green'; }
    setText('n-model', n.model || s.model || '—');
    setText('n-hot', fmtBytes(n.hot_bytes));
    setText('n-cold', fmtBytes(n.cold_bytes));
    setText('n-resident', fmtBytes(n.resident_bytes));
    setText('n-streamed', fmtBytes(n.streamed_bytes));
    const total = (n.prefetch_hits || 0) + (n.prefetch_misses || 0);
    setText('n-prefetch', total ? fmtPct(n.prefetch_hits / total) : '—');
    setText('n-tps', n.tokens_per_s ? `${n.tokens_per_s.toFixed(1)} tok/s` : 'idle');
    setText('n-lookahead', `${n.lookahead ?? '—'} layers · ${n.layer_index ?? 0}/${n.layers ?? '—'}`);
    setBar('n-bar', (n.resident_bytes || 0) / Math.max(1, n.cache_target_bytes || 1));
    const parts = [];
    if (n.cache_target_bytes) parts.push(`cache target ${fmtBytes(n.cache_target_bytes)}`);
    if (n.action) parts.push(`policy ${n.action}`);
    if (n.metal_active_bytes) parts.push(`metal ${fmtBytes(n.metal_active_bytes)}`);
    if (n.evictions) parts.push(`${fmtNum(n.evictions)} evicted`);
    if (n.experts_total) {
      parts.push(
        `experts ${n.experts_resident}/${n.experts_total} · ${fmtPct(n.expert_hit_rate)} routed`
      );
    }
    setText('n-note', parts.join(' · ') || 'hot weights resident, cold weights mmap-streamed');
  }

  // ---------------------------------------------------------------- rag
  function updateRagPanel(s) {
    const r = s.rag || {};
    const chip = $('rag-chip');
    if (!r.available) {
      if (chip) { chip.textContent = 'offline'; chip.className = 'chip amber'; }
      return;
    }
    if (chip) { chip.textContent = `${fmtNum(r.docs || 0)} docs`; chip.className = 'chip sky'; }
    setText('r-chunks', fmtNum(r.chunks));
    setText('r-clusters', fmtNum(r.clusters));
    setText('r-hit', `${fmtPct(r.prefetch_hit_rate || 0)}${r.prefetches ? ` · ${fmtNum(r.prefetches)}` : ''}`);
    setText('r-lat', `${(r.latency_ms || 0).toFixed(1)} ms`);
  }

  function renderResults(results, meta) {
    const list = $('rag-results');
    if (!list) return;
    if (!results || !results.length) {
      list.innerHTML = `<li class="empty">${esc(meta || 'no matches')}</li>`;
      return;
    }
    list.innerHTML = results
      .map((r) => {
        const score = (Number(r.score) || 0).toFixed(3);
        const path = esc(r.path || '?');
        const chunk = r.chunk ?? 0;
        const text = esc((r.text || '').replace(/\s+/g, ' ').slice(0, 220));
        return `<li><div class="r-top"><span class="mono">${path} <span class="dim">#${chunk}</span></span><b>${score}</b></div><div class="r-text">${text}</div></li>`;
      })
      .join('');
  }

  function bindRag() {
    const input = $('rag-input');
    const ray = $('rag-ray');
    if (!input) return;
    let timer = null;
    let seq = 0;

    input.addEventListener('input', () => {
      const value = input.value.trim();
      clearTimeout(timer);
      if (value.length < 3) {
        if (ray) { ray.textContent = 'idle'; ray.classList.remove('on'); }
        return;
      }
      timer = setTimeout(async () => {
        const mine = ++seq;
        const started = performance.now();
        try {
          const res = await fetch(`/api/rag/type?prefix=${encodeURIComponent(value)}`);
          const data = await res.json();
          if (mine !== seq) return; // stale
          if (!data || data.available === false) {
            if (ray) { ray.textContent = 'rag offline'; ray.classList.remove('on'); }
            return;
          }
          const ms = performance.now() - started;
          const clusters = (data.clusters || []).length;
          if (ray) {
            ray.textContent = `prefetched ${clusters} cluster${clusters === 1 ? '' : 's'} · ${fmtBytes(data.warm_bytes)} · ${ms.toFixed(1)} ms`;
            ray.classList.add('on');
          }
          if (Array.isArray(data.docs) && data.docs.length) {
            renderResults(
              data.docs.map((d) => ({ path: d.path, chunk: d.chunk, score: d.score ?? 0, text: d.text || '' })),
              'prefetched candidates'
            );
          }
        } catch (e) {
          if (ray) { ray.textContent = 'error'; ray.classList.remove('on'); }
        }
      }, 130);
    });

    input.addEventListener('keydown', async (e) => {
      if (e.key !== 'Enter') return;
      const q = input.value.trim();
      if (!q || state.ragBusy) return;
      state.ragBusy = true;
      if (ray) { ray.textContent = 'querying…'; ray.classList.add('on'); }
      try {
        const res = await fetch(`/api/rag/query?q=${encodeURIComponent(q)}&k=6`);
        const data = await res.json();
        if (!data || data.available === false) {
          renderResults([], data && data.error ? data.error : 'RAG engine unavailable');
          if (ray) ray.textContent = 'offline';
          return;
        }
        renderResults(data.results || [], 'no matches');
        if (ray) {
          ray.textContent = `${(data.latency_ms || 0).toFixed(1)} ms · prefetch ${data.prefetch_hit ? 'hit' : 'miss'}`;
          ray.classList.toggle('on', !!data.prefetch_hit);
        }
      } catch (err) {
        if (ray) ray.textContent = 'error';
      } finally {
        state.ragBusy = false;
      }
    });
  }

  // ---------------------------------------------------------------- feed
  const KIND_CLASS = (kind) => {
    const k = String(kind || '').toLowerCase();
    if (k.includes('warm')) return 'warm';
    if (k.includes('open') || k.includes('create') || k.includes('modify')) return 'open';
    if (k.includes('pred')) return 'prediction';
    if (k.includes('budget') || k.includes('release')) return 'budget';
    if (k.includes('error')) return 'error';
    if (k === 'ai') return 'ai';
    return '';
  };

  function updateFeed(s) {
    const feed = $('feed');
    if (!feed) return;
    const events = (s.events || []).slice().reverse().slice(0, 40);
    if (!events.length) {
      feed.innerHTML = '<li class="empty">no activity yet — open a file or run with --demo</li>';
      return;
    }
    feed.innerHTML = events
      .map((e) => {
        const cls = KIND_CLASS(e.kind);
        const detail = esc(e.detail || '');
        return `<li><span class="k">${fmtAgo(e.ts)}</span><span class="kind ${cls}">${esc(e.kind || '')}</span><span style="min-width:0"><span class="path">${esc(e.path || '—')}</span> <span class="detail">${detail}</span></span></li>`;
      })
      .join('');
  }

  function updatePool(s) {
    const list = $('pool-list');
    if (!list) return;
    const workspace = (state.snap && state.snap.workspace) || '';
    const entries = ((s.pool && s.pool.entries) || []).slice(0, 24);
    if (!entries.length) {
      list.innerHTML = '<li class="empty">pool is empty — waiting for file activity</li>';
      return;
    }
    const max = Math.max(...entries.map((e) => e.logical_bytes || 0), 1);
    list.innerHTML = entries
      .map((e) => {
        const raw = String(e.path || '');
        const path = esc(workspace && raw.startsWith(workspace) ? raw.slice(workspace.length + 1) : raw);
        const ratio = e.ratio ? `${e.ratio.toFixed(1)}×` : '';
        const storage = e.storage === 'packed' ? ' · packed' : '';
        return `<li><div class="row"><span class="path" title="${path}">${path}</span><span class="sz">${fmtBytes(e.logical_bytes)} · ${ratio}${storage}</span></div><div class="mini"><i style="width:${((e.logical_bytes || 0) / max) * 100}%"></i></div></li>`;
      })
      .join('');
  }

  // ---------------------------------------------------------------- main
  function apply(s) {
    state.snap = s;
    const m = s.memory || {}, p = s.pool || {}, pred = s.predictor || {};
    state.history.push({
      t: s.ts,
      mext: p.resident_bytes || 0,
      app: m.app || 0,
      wired: m.wired || 0,
      comp: m.compressor_occupied || 0,
      virtual: p.logical_bytes || 0,
      hit: pred.hit_rate || 0,
      ssd: p.ssd_mbps || 0,
      total: m.total || 1,
    });
    if (state.history.length > MAX_SAMPLES) state.history.shift();
    updateCards(s);
    updateNeural(s);
    updateRagPanel(s);
    updateFeed(s);
    updatePool(s);
    renderAll();
  }

  async function poll() {
    try {
      const res = await fetch('/api/snapshot', { cache: 'no-store' });
      if (!res.ok) throw new Error(`http ${res.status}`);
      const snap = await res.json();
      state.online = true;
      stream.last = Date.now();
      apply(snap);
    } catch (err) {
      if (state.online) {
        state.online = false;
        setText('ai-text', 'core offline');
        const pill = $('ai-pill');
        if (pill) { pill.classList.remove('good', 'warn'); pill.classList.add('bad'); }
        setText('footer-note', 'core offline — restart with `mext start`');
      }
    }
  }

  function startPolling() {
    if (stream.pollTimer) return;
    stream.pollTimer = setInterval(poll, 900);
  }

  function stopPolling() {
    if (stream.pollTimer) {
      clearInterval(stream.pollTimer);
      stream.pollTimer = null;
    }
  }

  function connectStream() {
    if (typeof EventSource === 'undefined') {
      startPolling();
      return;
    }
    try {
      const source = new EventSource('/api/stream');
      stream.source = source;
      source.onmessage = (event) => {
        try {
          const snap = JSON.parse(event.data);
          stream.last = Date.now();
          state.online = true;
          apply(snap);
          stopPolling();
        } catch (err) {
          /* malformed frame: keep the last good one */
        }
      };
      // EventSource reconnects on its own; the watchdog below decides when to
      // fall back to polling for good.
      source.onerror = () => {};
    } catch (err) {
      startPolling();
    }
  }

  function boot() {
    charts.ram = new Chart($('ram-chart'));
    charts.hit = new Chart($('hit-chart'));
    charts.ssd = new Chart($('ssd-chart'));
    bindRag();
    requestAnimationFrame(renderAll);
    poll();
    if (FORCE_POLL) startPolling();
    else connectStream();
    setInterval(() => {
      if (Date.now() - stream.last > 3200) startPolling();
    }, 1000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
