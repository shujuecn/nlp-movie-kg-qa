"""周星驰电影知识图谱问答系统的 Gradio 工作台（v2 界面）。

先决条件：
    1. kg/ 下已有三元组与别名表（python main.py build-kg；成品随仓库分发）；
    2. data/index/ 下已有 FAISS 索引（python main.py index；缺失时 Retriever 会
       尝试自动重建，需要 SILICONFLOW_API_KEY）；
    3. 可选：根目录 .env 配置 SILRA_API_KEY（无 key 自动进入降级模式）。

用法：
    python webapp.py
    python webapp.py --config configs/kg_qa.yaml --port 7860 --host 127.0.0.1

界面（mockup 三轮定稿方案）：
    头部   左标题 + 右状态签行（● 运行中 / ◐ 仅直答 / 模型名 / 73 部 · N 三元组
           +「图谱浏览」开关）；组件加载失败时状态签变 ❌ 加载失败，详情收进
           默认折叠的「加载详情」Accordion；
    主面板 左对话区（gr.Chatbot messages + 回答模式下拉 + 输入框 + 发送/清空），
           右证据面板：问题理解 JSON、命中三元组 + 检索片段（单组件 HTML，含
           score 徽章 / 强度条 / 类型胶囊 / 来源角标 / 引用角标联动）、本次用量；
    抽屉   点「图谱浏览」开关从右侧滑出（组件内 fixed 覆盖层 + JS 控制显隐）：
           左三元组数据表（搜索即过滤、行点击聚焦），右 d3 力导向邻域图
           （≤40 节点、拖拽 / 缩放 / 点击高亮邻居、图例）；首次打开才渲染，
           收起时停止力仿真；
    弹窗   点胶囊打开「聚焦：实体（邻域 N 节点）」弹窗，盖在抽屉之上；
           Esc 先关弹窗再关抽屉。

说明：
    - 单轮问答：回答只依赖图谱与检索，不依赖历史对话，左侧会话历史仅作展示；
    - 「仅直答」= answer(llm=None)，等同无 key 降级，状态签切琥珀色；
    - 分级加载：图谱 / 别名 / 向量检索 / LLM 任一失败不白屏，能在缺角情况下
      继续运行（retriever/llm 缺失只降级，图谱缺失才整体报错）；
    - 问答编排复用 kg_qa.answer（契约 §6），webapp 只做界面与证据展示；
      图谱可视化数据由 kg_viz.py 序列化（邻域提取 / JSON 组装）。
"""

from __future__ import annotations

import argparse
import html
import re
from typing import Any

import gradio as gr

from kgqa.config import Config, load_config
from kgqa.graph import load_graph
from kgqa.kg_linker import load_aliases
from kgqa.kg_viz import (
    drawer_payload,
    error_payload,
    payloads_for_entities,
    viz_json,
)
from kgqa.llm_client import create_client, has_api_key, load_env_file
from kgqa.qa.answering import answer
from kgqa.qa.retriever import Retriever

# answer() 返回的 mode → 界面显示名（契约 §6）
# 回答模式 → (是否使用 LLM, 强制路线)；教学对照：同一问题四种证据组装方式
MODE_CONFIG: dict[str, tuple[bool, str | None]] = {
    "自动": (True, None),
    "仅直答": (False, "fact"),
    "仅 RAG": (True, "rag"),
    "仅检索": (False, "evidence"),
}

MODE_LABELS: dict[str, str] = {
    "direct": "模板直答",
    "rag": "RAG 生成",
    "fallback": "超范围兜底",
    "degraded": "降级模式（无 key / 仅直答）",
}
USAGE_PLACEHOLDER = "暂无调用（提问后显示本次 API 用量）"
EVI_PLACEHOLDER = ""

# serve 启动时一次性加载的运行时组件；分级容错：
#   graph 缺失 → error（整体不可用）；retriever / llm 缺失 → 降级仍可问答
_state: dict[str, Any] = {
    "error": None,
    "load_errors": [],
    "cfg": None,
    "graph": None,
    "aliases": None,
    "retriever": None,
    "llm": None,
    "n_movies": None,
    "n_triples": None,
}

# lucide「share-2」内联 SVG（抽屉开关图标，跟随文字颜色）
SHARE2_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/>'
    '<circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>'
    '<line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>'
    '<line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>'
)

# 抽屉 / 弹窗 / 力导向图的浏览器端运行时：d3 懒加载、邻域提取（与 kg_viz 同口径）、
# 表格渲染与搜索过滤、抽屉显隐（transform + aria-pressed）、Esc 先关弹窗再关抽屉。
KGVIZ_JS = r"""
window.KGViz = (function () {
  const D3_URL = 'https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js';
  let d3Promise = null;

  function loadD3() {
    if (window.d3) return Promise.resolve(window.d3);
    if (!d3Promise) {
      d3Promise = new Promise(function (resolve, reject) {
        if (!document.querySelector('script[src*="d3.min.js"]')) {
          const s = document.createElement('script');
          s.src = D3_URL;
          document.head.appendChild(s);
        }
        const t0 = Date.now();
        (function poll() {
          if (window.d3) return resolve(window.d3);
          if (Date.now() - t0 > 15000) return reject(new Error('加载超时'));
          setTimeout(poll, 90);
        })();
      });
    }
    return d3Promise;
  }

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function fmt(n) { return Number(n || 0).toLocaleString('en-US'); }
  function typeOf(data, name) {
    return (data && data.types && data.types[name]) || '字面值';
  }

  const S = {
    root: null, data: null, drawerEl: null, modalEl: null,
    drawerSvg: null, modalSvg: null,
    drawerOpen: false, drawerRendered: false, escHandler: null,
  };

  /* 与 kg_viz.neighborhood 同口径的邻域提取：1 跳含字面值（排序靠后），
     2 跳跳过字面值；实体优先、度数降序、名字升序；上限 data.max_nodes（40）。 */
  function neighborhood(data, focus, maxNodes) {
    if (!data || !data.edges || !data.edges.length) return null;
    const cap = maxNodes || data.max_nodes || 40;
    const hops = data.hops || 2;
    const deg = {};
    data.edges.forEach(function (e) {
      deg[e[0]] = (deg[e[0]] || 0) + 1;
      deg[e[2]] = (deg[e[2]] || 0) + 1;
    });
    if (!deg[focus]) focus = data.focus;
    if (!deg[focus]) return null;
    const chosen = [focus];
    const seen = {}; seen[focus] = 1;
    let frontier = [focus];
    for (let depth = 0; depth < hops && chosen.length < cap; depth++) {
      const cand = {};
      frontier.forEach(function (n) {
        data.edges.forEach(function (e) {
          if (e[0] === n && !seen[e[2]]) cand[e[2]] = 1;
          if (e[2] === n && !seen[e[0]]) cand[e[0]] = 1;
        });
      });
      let list = Object.keys(cand);
      if (depth > 0) {
        list = list.filter(function (n) { return typeOf(data, n) !== '字面值'; });
      }
      list.sort(function (a, b) {
        const la = typeOf(data, a) === '字面值' ? 1 : 0;
        const lb = typeOf(data, b) === '字面值' ? 1 : 0;
        if (la !== lb) return la - lb;
        if ((deg[b] || 0) !== (deg[a] || 0)) return (deg[b] || 0) - (deg[a] || 0);
        return a < b ? -1 : 1;
      });
      const nxt = [];
      for (let i = 0; i < list.length && chosen.length < cap; i++) {
        seen[list[i]] = 1; chosen.push(list[i]); nxt.push(list[i]);
      }
      frontier = nxt;
    }
    const inSet = {};
    chosen.forEach(function (n) { inSet[n] = 1; });
    const links = []; const lk = {};
    data.edges.forEach(function (e) {
      if (!inSet[e[0]] || !inSet[e[2]]) return;
      const key = e[0] + '|' + e[2] + '|' + e[1];
      if (lk[key]) return;
      lk[key] = 1;
      links.push({ source: e[0], target: e[2], relation: e[1], src: e[3] });
    });
    const nodes = chosen.map(function (id) {
      return { id: id, type: typeOf(data, id), deg: deg[id] || 0 };
    });
    const legend = {};
    nodes.forEach(function (n) { legend[n.type] = (legend[n.type] || 0) + 1; });
    return {
      focus: focus, nodes: nodes, links: links, colors: (data && data.colors) || {},
      legend: Object.keys(legend).map(function (t) { return { type: t, count: legend[t] }; })
        .sort(function (a, b) { return b.count - a.count; }),
    };
  }

  function legendHtml(payload, fallbackData) {
    let legend = payload && payload.legend;
    const colors = (payload && payload.colors) || (fallbackData && fallbackData.colors) || {};
    if ((!legend || !legend.length) && fallbackData && fallbackData.types) {
      const c = {};
      Object.keys(fallbackData.types).forEach(function (k) {
        const t = fallbackData.types[k];
        c[t] = (c[t] || 0) + 1;
      });
      legend = Object.keys(c).map(function (t) { return { type: t, count: c[t] }; })
        .sort(function (a, b) { return b.count - a.count; });
    }
    return (legend || []).map(function (item) {
      return '<span class="lg"><i class="lg-dot" style="background:'
        + (colors[item.type] || '#9ca3af') + '"></i>' + esc(item.type)
        + ' ×' + fmt(item.count) + '</span>';
    }).join('');
  }

  function renderGraph(svgEl, payload) {
    if (!svgEl || !payload) return;
    loadD3().then(function (d3) {
      try { draw(d3, svgEl, payload); }
      catch (e) { console.error('KGViz 渲染失败', e); }
    }).catch(function (err) {
      const wrap = svgEl.parentElement;
      if (wrap) wrap.innerHTML = '<div class="kg-err">d3 加载失败：'
        + esc(err && err.message) + '（请检查网络或 CSP）</div>';
    });
  }

  function draw(d3, svgEl, payload) {
    if (svgEl.__sim) svgEl.__sim.stop();
    const wrap = svgEl.parentElement;
    const w = Math.max((wrap && wrap.clientWidth) || 620, 280);
    const h = Math.max((wrap && wrap.clientHeight) || 430, 260);
    const svg = d3.select(svgEl).attr('viewBox', '0 0 ' + w + ' ' + h);
    svg.selectAll('*').remove();

    const colors = payload.colors || {};
    const nodes = payload.nodes.map(function (n) { return Object.assign({}, n); });
    const links = payload.links.map(function (l) {
      return { source: l.source, target: l.target, rel: l.relation, src: l.src };
    });
    const byId = {};
    nodes.forEach(function (n) { byId[n.id] = n; });
    const focus = byId[payload.focus] || null;
    if (focus) { focus.fx = w / 2; focus.fy = h / 2; }

    const adj = {};
    links.forEach(function (l) {
      (adj[l.source] = adj[l.source] || []).push(l.target);
      (adj[l.target] = adj[l.target] || []).push(l.source);
    });

    function radius(d) { return d.type === '电影' ? 17 : 11; }

    const g = svg.append('g');
    const linkSel = g.append('g').selectAll('line').data(links).join('line')
      .attr('class', function (d) {
        return 'kg-edge' + (d.src === 'llm-offline' ? ' kg-edge-llm' : '');
      })
      .append('title')
      .text(function (d) {
        return d.source + ' —' + d.rel + '→ ' + d.target + (d.src ? '（' + d.src + '）' : '');
      })
      .select(function () { return this.parentNode; });
    const elab = g.append('g').selectAll('text').data(links).join('text')
      .attr('class', 'kg-elab').attr('text-anchor', 'middle')
      .text(function (d) { return d.rel; });
    let ring = null;
    if (focus) {
      ring = g.append('circle').attr('class', 'kg-ring')
        .attr('r', radius(focus) * (focus.type === '电影' ? 1 : 1) + 6);
    }
    const node = g.append('g').selectAll('g').data(nodes).join('g')
      .attr('class', 'kg-node').style('cursor', 'pointer');
    node.append('circle')
      .attr('r', function (d) { return radius(d) * (d.id === payload.focus ? 1.2 : 1); })
      .attr('fill', function (d) { return colors[d.type] || '#9ca3af'; })
      .attr('fill-opacity', 0.92)
      .append('title')
      .text(function (d) { return d.id + '（' + d.type + '）'; });
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', function (d) { return (d.type === '电影' ? 30 : 22); })
      .text(function (d) { return d.id.length > 9 ? d.id.slice(0, 9) + '…' : d.id; });

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(function (d) { return d.id; }).distance(88))
      .force('charge', d3.forceManyBody().strength(-170))
      .force('center', d3.forceCenter(w / 2, h / 2))
      .force('collide', d3.forceCollide().radius(function (d) { return radius(d) + 6; }));
    function paintPositions() {
      linkSel.attr('x1', function (d) { return d.source.x; })
        .attr('y1', function (d) { return d.source.y; })
        .attr('x2', function (d) { return d.target.x; })
        .attr('y2', function (d) { return d.target.y; });
      elab.attr('x', function (d) { return (d.source.x + d.target.x) / 2; })
        .attr('y', function (d) { return (d.source.y + d.target.y) / 2 - 3; });
      node.attr('transform', function (d) { return 'translate(' + d.x + ',' + d.y + ')'; });
      if (ring && focus) ring.attr('cx', focus.x).attr('cy', focus.y);
    }
    sim.on('tick', paintPositions);
    /* 不依赖 rAF 定时器：停掉内置计时器后手动同步跑完 240 个 tick，
       得到确定性布局——避免抽屉刚打开就关闭 / 后台标签页 rAF 节流
       导致节点全部停在 (0,0)。拖拽时再通过 restart() 恢复实时仿真。 */
    sim.stop();
    for (let i = 0; i < 240; i++) sim.tick();
    paintPositions();
    svgEl.__sim = sim;

    let selected = null;
    function applyHighlight(id) {
      if (!id) {
        node.classed('kg-dim', false);
        linkSel.classed('kg-dim', false);
        elab.classed('kg-elab-on', false);
        return;
      }
      const keep = new Set([id].concat(adj[id] || []));
      node.classed('kg-dim', function (d) { return !keep.has(d.id); });
      linkSel.classed('kg-dim', function (d) { return d.source.id !== id && d.target.id !== id; });
      elab.classed('kg-elab-on', function (d) { return d.source.id === id || d.target.id === id; });
    }
    node.on('click', function (ev, d) {
      ev.stopPropagation();
      selected = (selected === d.id) ? null : d.id;
      applyHighlight(selected);
    });
    svg.on('click', function () { selected = null; applyHighlight(null); });
    node.call(d3.drag()
      .on('start', function (ev, d) {
        if (!ev.active) sim.alphaTarget(0.25).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', function (ev, d) { d.fx = ev.x; d.fy = ev.y; })
      .on('end', function (ev, d) {
        if (!ev.active) sim.alphaTarget(0);
        if (d !== focus) { d.fx = null; d.fy = null; }
      }));
    svg.call(d3.zoom().scaleExtent([0.5, 3]).on('zoom', function (ev) {
      g.attr('transform', ev.transform);
    }));
  }

  function buildTable() {
    const tbody = S.drawerEl.querySelector('.kg-tbody');
    if (!tbody || !S.data) return;
    tbody.innerHTML = S.data.edges.map(function (e) {
      const cls = e[3] === 'llm-offline' ? 's-llm' : 's-douban';
      return '<tr class="kg-row" data-h="' + esc(e[0]) + '" data-t="' + esc(e[2]) + '"><td>'
        + esc(e[0]) + '</td><td><span class="kg-rel kg-rel-static">' + esc(e[1])
        + '</span></td><td>' + esc(e[2]) + '</td><td><span class="kg-src ' + cls + '">'
        + esc(e[3]) + '</span></td></tr>';
    }).join('');
  }

  function filterTable(q) {
    const tbody = S.drawerEl.querySelector('.kg-tbody');
    if (!tbody) return;
    const needle = String(q || '').trim().toLowerCase();
    let visible = 0;
    for (let i = 0; i < tbody.rows.length; i++) {
      const tr = tbody.rows[i];
      const hit = !needle || tr.textContent.toLowerCase().indexOf(needle) >= 0;
      tr.style.display = hit ? '' : 'none';
      if (hit) visible++;
    }
    const stats = S.drawerEl.querySelector('.kg-table-stats');
    if (stats && S.data && S.data.stats) {
      stats.textContent = needle
        ? '命中 ' + fmt(visible) + ' / ' + fmt(S.data.stats.triples) + ' 条'
        : '共 ' + fmt(S.data.stats.triples) + ' 条 · 关系 ' + S.data.stats.relations + ' 种';
    }
  }

  function focusEntity(name) {
    const payload = S.data ? neighborhood(S.data, name) : null;
    if (!payload) return;
    S.focusName = payload.focus;
    const headStats = S.drawerEl.querySelector('.kg-drawer-stats');
    if (headStats && S.data && S.data.stats) {
      headStats.textContent = '聚焦：' + payload.focus + ' · ' + payload.nodes.length
        + ' 节点 · ' + payload.links.length + ' 边｜全库 ' + fmt(S.data.stats.triples) + ' 三元组';
    }
    renderGraph(S.drawerSvg, payload);
  }

  function init(root, data) {
    S.root = root;
    S.data = data;
    S.drawerEl = root.querySelector('.kg-drawer');
    S.modalEl = root.querySelector('.kg-modal');
    if (!S.drawerEl || !S.modalEl) return;
    S.drawerSvg = S.drawerEl.querySelector('svg.kg-svg');
    S.modalSvg = S.modalEl.querySelector('svg.kg-svg');

    const stats = S.drawerEl.querySelector('.kg-table-stats');
    if (stats && data && data.stats) {
      stats.textContent = '共 ' + fmt(data.stats.triples) + ' 条 · 关系 '
        + data.stats.relations + ' 种 · 实体 ' + fmt(data.stats.entities);
    }
    if (!data || !data.edges || !data.edges.length) {
      const wrap = S.drawerSvg && S.drawerSvg.parentElement;
      if (wrap) {
        wrap.innerHTML = '<div class="kg-err">'
          + esc((data && data.error) || '图谱数据为空：请先运行 python main.py build-kg')
          + '</div>';
      }
      if (stats) stats.textContent = '图谱未加载';
    } else {
      buildTable();
      const lg = S.drawerEl.querySelector('.kg-legend');
      if (lg) lg.innerHTML = legendHtml(null, data);
    }

    const search = S.drawerEl.querySelector('.kg-search');
    if (search) {
      search.addEventListener('input', function () { filterTable(search.value); });
    }
    const tbody = S.drawerEl.querySelector('.kg-tbody');
    if (tbody) {
      tbody.addEventListener('click', function (ev) {
        const tr = ev.target.closest('tr.kg-row');
        if (!tr) return;
        tbody.querySelectorAll('tr.kg-row-focus').forEach(function (row) {
          row.classList.remove('kg-row-focus');
        });
        tr.classList.add('kg-row-focus');
        const cell = ev.target.closest('td');
        let name = tr.getAttribute('data-h') || '';
        if (cell && cell.cellIndex === 2) {
          const t = tr.getAttribute('data-t') || '';
          if (t && typeOf(S.data, t) !== '字面值') name = t;
        }
        if (name) focusEntity(name);
      });
    }
    const dc = S.drawerEl.querySelector('[data-kgclose-drawer]');
    if (dc) dc.addEventListener('click', function () { closeDrawer(); });
    const mc = S.modalEl.querySelector('[data-kgclose-modal]');
    if (mc) mc.addEventListener('click', function () { closeModal(); });
    S.modalEl.addEventListener('click', function (ev) {
      if (ev.target === S.modalEl) closeModal();
    });

    if (S.escHandler) document.removeEventListener('keydown', S.escHandler);
    S.escHandler = function (ev) {
      if (ev.key !== 'Escape') return;
      if (S.modalEl && !S.modalEl.hidden) { closeModal(); return; }
      if (S.drawerOpen) closeDrawer();
    };
    document.addEventListener('keydown', S.escHandler);
    syncToggle();
  }

  function openDrawer() {
    if (!S.drawerEl) return;
    S.drawerEl.classList.add('open');
    S.drawerOpen = true;
    syncToggle();
    if (!S.drawerRendered) {
      S.drawerRendered = true;
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { focusEntity(S.data && S.data.focus); });
      });
    } else {
      /* 再次打开时重绘：修复上次布局被中途停止（拖拽未结束即关闭）的状态 */
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { focusEntity(S.focusName || (S.data && S.data.focus)); });
      });
    }
  }

  function closeDrawer() {
    if (!S.drawerEl) return;
    S.drawerEl.classList.remove('open');
    S.drawerOpen = false;
    syncToggle();
    if (S.drawerSvg && S.drawerSvg.__sim) S.drawerSvg.__sim.stop();
  }

  function toggleDrawer() {
    if (S.drawerOpen) closeDrawer(); else openDrawer();
  }

  function openFocus(name, payload) {
    if (!S.modalEl) return;
    payload = payload || (S.data ? neighborhood(S.data, name) : null);
    if (!payload || !payload.nodes || !payload.nodes.length) return;
    const label = S.modalEl.querySelector('.kg-focus-name');
    if (label) {
      label.textContent = name + '（邻域 ' + payload.nodes.length + ' 节点 · '
        + payload.links.length + ' 边）';
    }
    const lg = S.modalEl.querySelector('.kg-legend');
    if (lg) lg.innerHTML = legendHtml(payload, null);
    S.modalEl.hidden = false;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { renderGraph(S.modalSvg, payload); });
    });
  }

  function closeModal() {
    if (!S.modalEl) return;
    S.modalEl.hidden = true;
    if (S.modalSvg && S.modalSvg.__sim) S.modalSvg.__sim.stop();
  }

  function syncToggle() {
    const btn = document.querySelector('[data-kgtoggle]');
    if (btn) btn.setAttribute('aria-pressed', S.drawerOpen ? 'true' : 'false');
  }

  return {
    init: init,
    toggleDrawer: toggleDrawer,
    openFocus: openFocus,
    closeModal: closeModal,
    closeDrawer: closeDrawer,
    syncToggle: syncToggle,
    neighborhood: neighborhood,
    renderGraph: renderGraph,
  };
})();
"""

KG_CSS = """
:root{--kg-acc:#f97316;--kg-text:#1a1c20;--kg-muted:#5b6472;--kg-border:#e3e5ea;
--kg-surface:#ffffff;--kg-panel:#f6f7f9;--kg-barbg:#e8eaf1;--kg-edge:#cbd5e1;
--kg-chip-fg:#5b6472;--kg-chip-bd:#dfe2e9;
--kg-ok-fg:#067a57;--kg-ok-bd:#b8ecd6;--kg-ok-bg:#effcf6;
--kg-warn-fg:#b45309;--kg-warn-bd:#fde68a;--kg-warn-bg:#fff8ea;
--kg-err-fg:#b91c1c;--kg-err-bd:#fecaca;--kg-err-bg:#fef2f2}
.dark{--kg-text:#e8eaee;--kg-muted:#a7adb9;--kg-border:#2c2e35;--kg-surface:#202228;
--kg-panel:#17181c;--kg-barbg:#26282f;--kg-edge:#475569;
--kg-chip-fg:#a7adb9;--kg-chip-bd:#2c2e35;
--kg-ok-fg:#5fd4a4;--kg-ok-bd:#17553c;--kg-ok-bg:#0f2318;
--kg-warn-fg:#fbbf24;--kg-warn-bd:#57430f;--kg-warn-bg:#241a05;
--kg-err-fg:#f87171;--kg-err-bd:#7f1d1d;--kg-err-bg:#2a1212}

/* ── 头部：标题与状态签 ── */
.kg-title{font-size:18px;font-weight:600;color:var(--kg-text);letter-spacing:.01em}
.kg-chips{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;align-items:center}
.kg-chip{font-size:13px;padding:3px 10px;border-radius:999px;border:1px solid var(--kg-chip-bd);
color:var(--kg-chip-fg);white-space:nowrap;background:transparent;display:inline-flex;
align-items:center;gap:5px;font-variant-numeric:tabular-nums;font-family:inherit;line-height:1.5}
.kg-chip.ok{color:var(--kg-ok-fg);border-color:var(--kg-ok-bd);background:var(--kg-ok-bg)}
.kg-chip.warn{color:var(--kg-warn-fg);border-color:var(--kg-warn-bd);background:var(--kg-warn-bg)}
.kg-chip.err{color:var(--kg-err-fg);border-color:var(--kg-err-bd);background:var(--kg-err-bg)}
.kg-toggle{cursor:pointer;transition:all .15s}
.kg-toggle:hover{border-color:var(--kg-acc);color:var(--kg-acc)}
.kg-toggle[aria-pressed="true"]{background:var(--kg-acc);border-color:var(--kg-acc);color:#fff}
.kg-toggle[aria-pressed="true"]:hover{color:#fff}

/* ── 证据面板：四字小节标题 / 三元组胶囊 / 来源角标 ── */
.kg-evi-root{min-height:120px;max-height:640px;overflow-y:auto;padding-right:2px}
.kg-h4{font-size:13px;margin:12px 0 6px;color:var(--kg-muted);font-weight:600;letter-spacing:.02em}
.kg-empty{font-size:13px;color:var(--kg-muted);padding:18px 6px;text-align:center}
.kg-tri{display:flex;align-items:center;gap:6px;padding:5px 0;font-size:13px;flex-wrap:wrap}
.kg-tri + .kg-tri{border-top:1px dashed var(--kg-border)}
.kg-ent{padding:2px 9px;border-radius:6px;font-size:13px;white-space:nowrap;color:#fff;display:inline-block}
.kg-ent.t-电影{background:#3b82f6}.kg-ent.t-人物{background:#22c55e}
.kg-ent.t-角色{background:#a855f7}.kg-ent.t-类型{background:#f59e0b}
.kg-ent.t-地区{background:#06b6d4}.kg-ent.t-语言{background:#64748b}
.kg-ent.t-字面值{background:#9ca3af}.kg-ent.t-主题{background:#ec4899}
.kg-link{cursor:pointer}
.kg-link:hover{outline:2px solid var(--kg-acc);outline-offset:1px}
.kg-rel{font-size:13px;padding:2px 8px;border-radius:999px;border:1px solid var(--kg-acc);
color:var(--kg-acc);white-space:nowrap;background:transparent}
.kg-rel-static{cursor:default;outline:none}
.kg-src{margin-left:auto;font-size:13px;padding:1px 7px;border-radius:5px;white-space:nowrap}
.kg-src.s-douban{background:var(--kg-panel);color:var(--kg-muted);border:1px solid var(--kg-border)}
.kg-src.s-llm{background:#f4ebff;color:#8b3fd1;border:1px solid #e2d0fb}
.dark .kg-src.s-llm{background:#33214d;color:#c9a6f7;border-color:#4b3566}

/* ── 检索片段：编号 + score 徽章 + 强度条 ── */
.kg-pass{border:1px solid var(--kg-border);border-radius:9px;padding:9px 11px;margin:7px 0;background:var(--kg-panel)}
.kg-pass-top{display:flex;align-items:center;gap:8px}
.kg-cnum{font-size:13px;font-weight:600;color:var(--kg-acc)}
.kg-pass-title{font-size:13px;color:var(--kg-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kg-score{margin-left:auto;display:inline-flex;align-items:center;gap:7px;font-size:13px;font-weight:600;
color:var(--kg-acc);font-variant-numeric:tabular-nums}
.kg-bar{width:56px;height:5px;border-radius:3px;background:var(--kg-barbg);overflow:hidden;display:inline-block}
.kg-bar i{display:block;height:100%;background:var(--kg-acc)}
.kg-pass p{margin:6px 0 0;font-size:13px;line-height:1.7;color:var(--kg-text);word-break:break-word;display:-webkit-box;-webkit-line-clamp:7;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer}.kg-pass.kg-open p{display:block;-webkit-line-clamp:unset;overflow:visible}.gradio-container a[href^="#kg-cite-"]{color:var(--kg-acc,#f97316);font-weight:500;text-decoration:none;border-bottom:1px dashed var(--kg-acc,#f97316)}#kg-mode-radio .wrap,#kg-mode-radio [class*="container"]{display:flex;flex-direction:row;align-items:center;gap:10px;flex-wrap:nowrap;border:none;background:transparent;box-shadow:none;padding:6px 0;min-height:0}#kg-mode-radio label{display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-size:13px;color:var(--kg-muted,#6b7280);margin:0;border:1px solid var(--kg-border,#e0e3ea);border-radius:999px;padding:5px 14px;white-space:nowrap}#kg-mode-radio label:has(input:checked){color:var(--kg-acc,#f97316);border-color:var(--kg-acc,#f97316);font-weight:500}#kg-mode-radio input{accent-color:var(--kg-acc,#f97316);margin:0}
.kg-flash{outline:2px solid var(--kg-acc);outline-offset:2px;border-radius:9px}

/* ── 抽屉层与图谱浏览 ── */
.kg-layer{position:fixed;inset:0;z-index:90;pointer-events:none;font-family:inherit}
.kg-drawer{position:absolute;top:10px;right:10px;bottom:10px;width:80%;pointer-events:auto;
display:flex;flex-direction:column;gap:8px;padding:12px;overflow:hidden;
background:var(--kg-surface);border:1px solid var(--kg-border);border-radius:14px;
transform:translateX(103%);transition:transform .28s ease;box-shadow:-12px 0 32px rgba(8,10,14,.18)}
.kg-drawer.open{transform:translateX(0)}
@media (prefers-reduced-motion:reduce){.kg-drawer{transition:none}}
.kg-drawer-head{display:flex;align-items:center;gap:10px;font-size:14px;font-weight:600;color:var(--kg-text)}
.kg-drawer-ico{color:var(--kg-acc);display:inline-flex}
.kg-drawer-stats,.kg-table-stats,.kg-stats{font-size:13px;color:var(--kg-muted);
font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kg-drawer-head .kg-drawer-stats{margin-left:auto}
.kg-close{border:1px solid var(--kg-border);background:transparent;color:var(--kg-text);
border-radius:8px;width:28px;height:28px;font-size:15px;cursor:pointer;line-height:1;flex:none}
.kg-close:hover{border-color:var(--kg-acc);color:var(--kg-acc)}
.kg-panel{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:10px;flex:1;min-height:0}
.kg-left,.kg-right{display:flex;flex-direction:column;min-height:0;border:1px solid var(--kg-border);border-radius:10px;overflow:hidden}
.kg-toolbar{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--kg-border);flex-wrap:wrap}
.kg-search{margin-left:auto;height:30px;width:170px;border:1px solid var(--kg-border);border-radius:8px;
background:transparent;color:var(--kg-text);padding:0 10px;font-size:13px;font-family:inherit}
.kg-search:focus{outline:1px solid var(--kg-acc)}
.kg-tablewrap{flex:1;overflow-y:auto;min-height:0}
.kg-table{width:100%;border-collapse:collapse;font-size:12px}
.kg-table th{position:sticky;top:0;background:var(--kg-surface);text-align:left;font-weight:500;
font-size:13px;color:var(--kg-muted);padding:7px 10px;border-bottom:1px solid var(--kg-border);z-index:1}
.kg-table td{padding:6px 10px;border-bottom:1px dashed var(--kg-border);color:var(--kg-text)}
.kg-tbody tr{cursor:pointer}
.kg-tbody tr:hover td{background:rgba(148,163,184,.12)}
.kg-tbody tr.kg-row-focus td{background:rgba(249,115,22,.16)}
.kg-svgwrap{flex:1;min-height:0;position:relative;overflow:hidden}
.kg-svg{width:100%;height:100%;display:block}
.kg-svg text{fill:var(--kg-text);font-size:13px;pointer-events:none}
.kg-svg .kg-elab{fill:var(--kg-muted);font-size:11.5px;opacity:0}
.kg-svg .kg-elab-on{opacity:1}
.kg-svg line{stroke:var(--kg-edge);stroke-width:1.3}
.kg-svg line.kg-edge-llm{stroke:#a855f7;stroke-dasharray:4 3}
.kg-svg .kg-dim{opacity:.13}
.kg-ring{fill:none;stroke:var(--kg-acc);stroke-width:2.5}
.kg-legend{display:grid;grid-template-columns:repeat(4,auto);justify-content:start;gap:8px 18px;padding:8px 12px 2px;font-size:14px;color:var(--kg-muted)}
.kg-legend .lg{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.lg-dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.kg-err{color:var(--kg-err-fg);font-size:13px;padding:16px;line-height:1.7;white-space:pre-wrap}

/* ── 聚焦实体弹窗（盖在抽屉之上） ── */
.kg-modal{position:absolute;inset:0;z-index:95;pointer-events:auto;background:rgba(8,10,14,.55);
display:grid;place-items:center}
.kg-modal[hidden]{display:none}
.kg-modal-panel{width:min(92%,860px);height:min(88%,700px);background:var(--kg-surface);
border:1px solid var(--kg-border);border-radius:12px;display:flex;flex-direction:column;
padding:12px;gap:8px}
.kg-modal-head{display:flex;justify-content:space-between;align-items:center;font-size:14px;
font-weight:600;color:var(--kg-text)}
.kg-modal-head b{color:var(--kg-acc);font-weight:600}

/* ── 页脚 ── */
.kg-foot{font-size:13px;color:var(--kg-muted);text-align:center;padding:2px 0 0}
#kg-drawer-comp{padding:0 !important;margin:0 !important;border:none !important;background:none !important;min-height:0 !important;height:0;overflow:visible}

@media (max-width:860px){
  .kg-panel{grid-template-columns:1fr}
  .kg-drawer{width:100%;top:0;right:0;bottom:0;border-radius:0}
}
"""

# d3 CDN + KGViz 运行时（抽屉 / 弹窗 / 力导向图渲染），随 launch(head=) 注入一次。
KG_HEAD = (
    '<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>\n'
    "<script>\n" + KGVIZ_JS + "\n</script>"
)

KG_DRAWER_TEMPLATE = (
    '<div class="kg-layer">'
    '<div class="kg-drawer" aria-label="图谱浏览抽屉">'
    '<div class="kg-drawer-head">'
    '<span class="kg-drawer-ico">' + SHARE2_SVG + "</span>"
    "<span>图谱浏览</span>"
    '<span class="kg-drawer-stats"></span>'
    '<button class="kg-close" type="button" data-kgclose-drawer aria-label="收起抽屉">×</button>'
    "</div>"
    '<div class="kg-panel">'
    '<div class="kg-left">'
    '<div class="kg-toolbar"><span class="kg-table-stats"></span>'
    '<input class="kg-search" placeholder="搜索实体 / 关系…" aria-label="搜索三元组"></div>'
    '<div class="kg-tablewrap"><table class="kg-table"><thead><tr>'
    "<th>头实体</th><th>关系</th><th>尾实体</th><th>来源</th>"
    '</tr></thead><tbody class="kg-tbody"></tbody></table></div>'
    "</div>"
    '<div class="kg-right">'
    '<div class="kg-svgwrap"><svg class="kg-svg" role="img" aria-label="知识图谱邻域网络图"></svg></div>'
    '<div class="kg-legend"></div>'
    "</div>"
    "</div>"
    "</div>"
    '<div class="kg-modal" hidden>'
    '<div class="kg-modal-panel">'
    '<div class="kg-modal-head"><span>聚焦：<b class="kg-focus-name"></b></span>'
    '<button class="kg-close" type="button" data-kgclose-modal aria-label="关闭">×</button></div>'
    '<div class="kg-svgwrap"><svg class="kg-svg" role="img" aria-label="聚焦实体邻域网络图"></svg></div>'
    '<div class="kg-legend"></div>'
    "</div>"
    "</div>"
    '<script type="application/json" class="kg-data">${value}</script>'
    "</div>"
)

KG_DRAWER_JS = r"""
(function () {
  let data = null;
  try {
    const el = element.querySelector('.kg-data');
    data = el ? JSON.parse(el.textContent) : null;
  } catch (e) { data = null; }
  if (window.KGViz) window.KGViz.init(element, data);
})();
"""

KG_EVIDENCE_TEMPLATE = '<div class="kg-evi-root">${value}</div>'

KG_EVIDENCE_JS = r"""
(function () {
  function readData() {
    const el = element.querySelector('.kg-evi-data');
    if (!el) return {};
    try { return JSON.parse(el.textContent) || {}; } catch (e) { return {}; }
  }
  function flash(target) {
    if (!target) return;
    target.classList.remove('kg-flash');
    void target.offsetWidth;
    target.classList.add('kg-flash');
    target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    setTimeout(function () { target.classList.remove('kg-flash'); }, 1600);
  }
  if (element.__kgEviBound) return;
  element.__kgEviBound = true;
  /* 组件内委托：类型胶囊 / 关系胶囊 → 聚焦弹窗；检索片段编号角标 → 高亮 */
  element.addEventListener('click', function (ev) {
    const cap = ev.target.closest('[data-ent]');
    if (cap) {
      const name = cap.getAttribute('data-ent');
      const nb = (readData().neighborhoods) || {};
      if (window.KGViz && name) window.KGViz.openFocus(name, nb[name]);
      return;
    }
    const cite = ev.target.closest('.kg-cite');
    if (cite) flash(element.querySelector('#kg-idx-' + cite.getAttribute('data-n')));
    const pass = ev.target.closest('.kg-pass');
    if (pass) { pass.classList.toggle('kg-open'); return; }
  });
  /* 全局委托：聊天回答里的 [n] 引用是 markdown 链接（#kg-cite-n），
     拦截跳转并高亮证据面板中对应编号的行 / 卡片 */
  document.addEventListener('click', function (ev) {
    const a = ev.target.closest ? ev.target.closest('a[href^="#kg-cite-"]') : null;
    if (!a) return;
    const m = (a.getAttribute('href') || '').match(/kg-cite-(\d+)/);
    if (!m) return;
    ev.preventDefault();
    flash(element.querySelector('#kg-idx-' + m[1]));
  });
})();
"""

KG_CHIPS_JS = r"""
element.addEventListener('click', function (ev) {
  const btn = ev.target.closest('[data-kgtoggle]');
  if (btn && window.KGViz) window.KGViz.toggleDrawer();
});
watch('value', function () { if (window.KGViz) window.KGViz.syncToggle(); });
"""


def _load_state(cfg: Config) -> None:
    """分级加载图谱、别名表、FAISS 索引与（可选）LLM 客户端。

    - 图谱加载失败 → _state["error"]（页面状态签 ❌，问答不可用，服务仍启动）；
    - 别名 / 检索 / LLM 失败 → 记入 load_errors，对应能力降级但界面照常；
    - 无 key 时 llm=None，kg_qa 走规则降级（契约 §6）。
    """
    print(
        "正在加载知识图谱、别名表与 FAISS 索引（Retriever 将连接 Embedding API，请稍候）…"
    )
    errors: list[str] = []
    graph = aliases = retriever = llm = None

    try:
        graph = load_graph(str(cfg.kg_dir))
    except Exception as exc:
        errors.append(f"知识图谱加载失败：{exc}")

    try:
        aliases = load_aliases(str(cfg.aliases_file))
        print(f"  别名表 {len(aliases)} 行。")
    except Exception as exc:
        errors.append(f"别名表加载失败：{exc}（实体链接将退化为子串匹配）")

    try:
        retriever = Retriever(cfg, cfg.embedding)
    except Exception as exc:
        errors.append(f"向量检索初始化失败：{exc}（概述 / 生成题将进入降级模式）")

    if has_api_key(cfg.llm):
        try:
            llm = create_client(cfg.llm)
            print(f"silra API 客户端已就绪（{cfg.llm.model}）：理解与生成走 API。")
        except Exception as exc:
            errors.append(f"LLM 客户端创建失败：{exc}（按无 key 降级运行）")
    else:
        print(
            "⚠️ 未配置 SILRA_API_KEY，当前为降级模式：事实题直答可用，生成题返回检索证据。"
        )

    n_movies: int | None = None
    try:
        movies_file = cfg.root / cfg.douban.movies_file
        if movies_file.exists():
            with movies_file.open(encoding="utf-8") as fh:
                n_movies = sum(1 for line in fh if line.strip())
    except Exception:
        n_movies = None

    n_triples = graph.number_of_edges() if graph is not None else None
    _state.update(
        cfg=cfg,
        graph=graph,
        aliases=aliases,
        retriever=retriever,
        llm=llm,
        n_movies=n_movies,
        n_triples=n_triples,
        load_errors=errors,
        error=None,
    )
    if graph is None:
        _state["error"] = "\n".join(errors or ["知识图谱未能加载。"]) + (
            "\n\n请确认以下步骤已在仓库根目录完成：\n"
            "1. `python main.py ingest`——原始数据入库（data/movies.jsonl）；\n"
            "2. `python main.py build-kg`——生成 kg/ 三元组与别名表；\n"
            "3. `python main.py index`——生成 data/index/ 下的 FAISS 索引。"
        )
        print(f"组件加载失败，服务仍将启动并在页面状态签展示提示：{errors}")
    else:
        print(f"加载完成：图谱 {graph.number_of_nodes()} 节点 / {n_triples} 条边。")
        for message in errors:
            print(f"⚠️ {message}")


def _short_model(model: str | None) -> str:
    """去掉机构前缀的短模型名（Qwen/Qwen3-Embedding-8B → Qwen3-Embedding-8B）。"""
    if not model:
        return "—"
    return str(model).split("/")[-1]


def _chips_html(mode: str = "自动") -> str:
    """头部状态签行：状态 + 模型名 + 语料规模 + 图谱浏览开关。

    状态随加载结果与回答模式下拉联动：加载失败 ❌（红）＞ 仅直答 ◐（琥珀）＞
    降级运行 ◐（琥珀）＞ 运行中 ●（绿）。
    """
    if _state.get("error"):
        status_cls, status_text = "err", "❌ 加载失败"
    elif mode in ("仅直答", "仅检索"):
        status_cls, status_text = "warn", f"◐ {mode}"
    elif _state.get("llm") is None or _state.get("retriever") is None:
        status_cls, status_text = "warn", "◐ 降级运行"
    else:
        status_cls, status_text = "ok", "● 运行中"

    cfg: Config | None = _state.get("cfg")
    llm_name = _short_model(cfg.llm.model) if cfg else "—"
    embed_name = _short_model(cfg.embedding.model) if cfg else "—"
    n_movies = _state.get("n_movies")
    n_triples = _state.get("n_triples")
    corpus = (
        f"{n_movies if n_movies else '—'} 部 · "
        f"{n_triples if n_triples is not None else '—'} 三元组"
    )

    chips = [
        f'<span class="kg-chip {status_cls}">{status_text}</span>',
        f'<span class="kg-chip">{html.escape(llm_name)}</span>',
        f'<span class="kg-chip">{html.escape(embed_name)}</span>',
        f'<span class="kg-chip">{corpus}</span>',
        '<button class="kg-chip kg-toggle" type="button" data-kgtoggle aria-pressed="false" '
        f'title="打开图谱浏览抽屉">{SHARE2_SVG}图谱浏览</button>',
    ]
    return '<div class="kg-chips">' + "".join(chips) + "</div>"


def _load_details_markdown() -> str:
    """「加载详情」Accordion 内容：正常时一行说明，异常时逐项列出。"""
    if _state.get("error"):
        return str(_state["error"])
    errors: list[str] = _state.get("load_errors") or []
    if errors:
        return "\n".join(f"- {message}" for message in errors)
    return "全部组件加载正常（图谱 / 别名表 / FAISS 索引 / LLM 客户端）。"


def _usage_line(result: dict[str, Any]) -> str:
    """「本次用量」：最近一问的 API 调用次数与 token 用量。

    kg_qa 只保证 api_usage 含 prompt_tokens / completion_tokens（契约 §3/§6）：
    - api_usage 为 None → 本地模板 / 规则 / 检索，0 次调用；
    - 理解 dict 带 degraded_reason → 生成未发生，usage 来自理解调用（N=1）；
    - 否则为 RAG 生成调用（N=2：理解 1 + 生成 1）。
    """
    mode_raw = str(result.get("mode") or "?")
    mode = MODE_LABELS.get(mode_raw, mode_raw)
    evidence = result.get("evidence") or {}
    usage = evidence.get("api_usage")
    if not usage:
        return f"{mode}｜本次 API 调用：无（本地模板 / 规则 / 检索）"
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    degraded = "degraded_reason" in (evidence.get("understanding") or {})
    calls, understand, generate = (1, 1, 0) if degraded else (2, 1, 1)
    return (
        f"{mode}｜{calls} 次调用（理解 {understand} + 生成 {generate}）"
        f"｜prompt {prompt:,}｜completion {completion:,}"
    )


def _pill(entity: str, entity_type: str, neighborhoods: dict[str, Any]) -> str:
    """实体胶囊：按类型着色；邻域数据存在的实体可点击打开聚焦弹窗。"""
    cls = "kg-ent t-" + (entity_type or "字面值")
    if entity in neighborhoods:
        cls += " kg-link"
    attrs = ""
    if entity in neighborhoods:
        attrs = f' data-ent="{html.escape(entity, quote=True)}" title="查看 {html.escape(entity)} 的邻域"'
    return f'<span class="{cls}"{attrs}>{html.escape(entity)}</span>'


def _triple_rows_html(
    triples: list[dict[str, Any]], neighborhoods: dict[str, Any]
) -> str:
    """命中三元组：头实体胶囊 + 关系描边胶囊 + 尾实体胶囊 + 来源角标。

    关系胶囊点击聚焦该行的头实体；行 id 与证据编号（契约：三元组 [1..T]）对齐。
    """
    rows: list[str] = []
    for index, row in enumerate(triples, 1):
        head = str(row.get("head") or "")
        head_type = str(row.get("head_type") or "字面值")
        relation = str(row.get("relation") or "")
        tail = str(row.get("tail") or "")
        tail_type = str(row.get("tail_type") or "字面值")
        source = str(row.get("source") or "")
        src_cls = "s-llm" if source == "llm-offline" else "s-douban"
        rel_attrs = ""
        if head in neighborhoods:
            rel_attrs = f' data-ent="{html.escape(head, quote=True)}" title="查看 {html.escape(head)} 的邻域"'
        rows.append(
            f'<div class="kg-tri" id="kg-idx-{index}">'
            f"{_pill(head, head_type, neighborhoods)}"
            f'<span class="kg-rel{"" if rel_attrs else " kg-rel-static"}'
            f'{" kg-link" if rel_attrs else ""}"{rel_attrs}>{html.escape(relation)}</span>'
            f"{_pill(tail, tail_type, neighborhoods)}"
            f'<span class="kg-src {src_cls}">{html.escape(source or "graph")}</span>'
            "</div>"
        )
    if not rows:
        return '<div class="kg-empty">本次未命中三元组（问题理解见左侧 JSON）。</div>'
    return "".join(rows)


def _passages_html(passages: list[dict[str, Any]], offset: int) -> str:
    """检索片段卡片：证据编号 [n] + score 徽章（大字 + 迷你强度条）+ 片段正文。

    编号沿用 kg_qa._serialize_evidence 的口径：简介片段接在三元组之后（[T+1..]），
    与回答里的 [n] 引用一一对应。
    """
    cards: list[str] = []
    for index, passage in enumerate(passages, offset + 1):
        try:
            score = float(passage.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        pct = round(max(0.0, min(1.0, score)) * 100)
        title = html.escape(str(passage.get("title") or ""))
        chunk = html.escape(str(passage.get("chunk") or ""))
        title_html = f'<span class="kg-pass-title">《{title}》</span>' if title else ""
        cards.append(
            f'<div class="kg-pass" id="kg-idx-{index}">'
            '<div class="kg-pass-top">'
            f'<span class="kg-cnum">[{index}]</span>{title_html}'
            '<span class="kg-score">'
            f'<span class="kg-bar"><i style="width:{pct}%"></i></span>{score:.2f}'
            "</span></div>"
            f"<p>{chunk}</p>"
            "</div>"
        )
    if not cards:
        return '<div class="kg-empty">本次没有检索到简介片段（未配置 Embedding key 或证据不足）。</div>'
    return "".join(cards)


def _evidence_html(result: dict[str, Any]) -> str:
    """把命中三元组与检索片段渲染进同一个 gr.HTML（引用角标 / 胶囊交互都在组件内）。

    附带 <script type="application/json"> 数据块：命中三元组涉及实体的邻域
    （kg_viz.payloads_for_entities 预计算，≤10 个），供弹窗与 d3 渲染。
    """
    graph = _state.get("graph")
    evidence = result.get("evidence") or {}
    triples = evidence.get("triples") or []
    passages = evidence.get("passages") or []

    entities: list[str] = []
    for row in triples:
        for key, type_key in (("head", "head_type"), ("tail", "tail_type")):
            name = str(row.get(key) or "")
            if not name or str(row.get(type_key) or "字面值") == "字面值":
                continue
            if name not in entities:
                entities.append(name)
    neighborhoods = payloads_for_entities(graph, entities) if graph is not None else {}

    parts = [
        f'<h4 class="kg-h4">命中三元组（{len(triples)}）</h4>',
        _triple_rows_html(triples, neighborhoods),
        f'<h4 class="kg-h4">检索片段（{len(passages)}）</h4>',
        _passages_html(passages, len(triples)),
        '<script type="application/json" class="kg-evi-data">'
        + viz_json({"neighborhoods": neighborhoods})
        + "</script>",
    ]
    return "".join(parts)


def _link_citations(answer_text: str, n_evidence: int) -> str:
    """把回答里的 [n] 引用替换为 markdown 内链 #kg-cite-n（n 为有效证据编号时）。"""
    pattern = re.compile(r"\[(\d{1,2})\](?!\()")

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= n_evidence:
            return f"[\\[{number}\\]](#kg-cite-{number})"
        return match.group(0)

    return pattern.sub(replace, answer_text)


def _answer_once(question: str, mode: str) -> dict[str, Any]:
    """单轮问答入口：把界面状态交给 kg_qa.answer（契约 §6）。"""
    if _state.get("error"):
        raise RuntimeError(str(_state["error"]))
    use_llm, route = MODE_CONFIG.get(mode, (True, None))
    llm = _state["llm"] if use_llm else None
    return answer(
        question,
        graph=_state["graph"],
        aliases=_state["aliases"],
        retriever=_state["retriever"],
        llm=llm,
        cfg=_state["cfg"],
        force_route=route,
    )


def respond(
    message: str, history: list[dict[str, Any]], mode: str
) -> tuple[str, list[dict[str, Any]], Any, str, str]:
    """Gradio 回调：单轮问答 + 证据面板回填。

    会话历史仅前端展示；answer() 本身无状态，每次都从图谱与检索取证据。
    """
    message = message.strip()
    if not message:
        return "", history, None, EVI_PLACEHOLDER, USAGE_PLACEHOLDER

    history = [*history, {"role": "user", "content": message}]
    understanding: Any = None
    evidence_html = EVI_PLACEHOLDER
    usage_text = USAGE_PLACEHOLDER
    try:
        result = _answer_once(message, mode)
        evidence = result.get("evidence") or {}
        reply = str(result.get("answer") or "（没有生成回答）")
        n_evidence = len(evidence.get("triples") or []) + len(
            evidence.get("passages") or []
        )
        reply = _link_citations(reply, n_evidence)
        understanding = evidence.get("understanding")
        evidence_html = _evidence_html(result)
        usage_text = _usage_line(result)
    except Exception as exc:  # 出错也写进对话区，避免界面无响应
        reply = f"出错了：{exc}"
        usage_text = f"出错｜本次 API 调用：无（{type(exc).__name__}）"
    history = [*history, {"role": "assistant", "content": reply}]
    return "", history, understanding, evidence_html, usage_text


def _clear() -> tuple[list[dict[str, Any]], str, None, str, str]:
    """清空对话与证据面板（运行时组件保持不变）。"""
    return [], "", None, EVI_PLACEHOLDER, USAGE_PLACEHOLDER


def _drawer_value() -> str:
    """图谱浏览抽屉的全量 JSON（viz_json 转义后注入 data 块）。"""
    graph = _state.get("graph")
    if graph is None:
        return viz_json(error_payload(str(_state.get("error") or "知识图谱未加载。")))
    return viz_json(drawer_payload(graph))


def build_demo() -> gr.Blocks:
    """组装 Blocks 界面（mockup 定稿：头部状态签 + 问答主面板 + 图谱抽屉）。"""
    with gr.Blocks(title="周星驰电影知识图谱问答系统") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(
                    '<div class="kg-title">周星驰电影知识图谱问答系统</div>',
                    elem_id="kg-title-wrap",
                    apply_default_css=False,
                )
            with gr.Column(scale=3):
                chips_html = gr.HTML(
                    _chips_html(),
                    elem_id="kg-chips",
                    apply_default_css=False,
                    js_on_load=KG_CHIPS_JS,
                )
        with gr.Accordion("加载详情", open=False):
            gr.Markdown(_load_details_markdown())

        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    label="对话（单轮问答）",
                    height=460,
                    placeholder="例如：紫霞仙子是谁演的？／介绍一下《大话西游之大圣娶亲》",
                )
                msg = gr.Textbox(label="你的问题", placeholder="输入问题")
                mode_dd = gr.Radio(
                    choices=["自动", "仅直答", "仅 RAG", "仅检索"],
                    value="自动",
                    label="回答模式",
                    elem_id="kg-mode-radio",
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空")
            with gr.Column(scale=4):
                understanding_json = gr.JSON(label="问题理解")
                evidence_html = gr.HTML(
                    EVI_PLACEHOLDER,
                    label="命中三元组 / 检索片段",
                    show_label=False,
                    apply_default_css=False,
                    elem_id="kg-evi",
                    html_template=KG_EVIDENCE_TEMPLATE,
                    js_on_load=KG_EVIDENCE_JS,
                )
                usage_tb = gr.Textbox(
                    label="本次用量",
                    value=USAGE_PLACEHOLDER,
                    interactive=False,
                    max_lines=2,
                )

        gr.HTML(
            _drawer_value(),
            label=None,
            show_label=False,
            apply_default_css=False,
            elem_id="kg-drawer-comp",
            html_template=KG_DRAWER_TEMPLATE,
            js_on_load=KG_DRAWER_JS,
        )

        panel_outputs = [msg, chatbot, understanding_json, evidence_html, usage_tb]
        msg.submit(respond, [msg, chatbot, mode_dd], panel_outputs)
        send_btn.click(respond, [msg, chatbot, mode_dd], panel_outputs)
        clear_btn.click(_clear, None, panel_outputs)

        # 服务打开后自动演示一条例子（直答零 API 成本），并在输入框预填下一问
        bootstrap_question = "紫霞仙子是谁演的？"
        prefill_question = "介绍一下《大话西游之大圣娶亲》"

        def _bootstrap_example() -> tuple[Any, ...]:
            result = list(respond(bootstrap_question, [], "自动"))
            result[0] = prefill_question  # 输入框预填充
            return tuple(result)

        demo.load(_bootstrap_example, None, panel_outputs)
        mode_dd.change(_chips_html, [mode_dd], [chips_html])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="周星驰电影知识图谱问答工作台（Gradio Blocks）"
    )
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    parser.add_argument(
        "--host", default=None, help="监听地址，默认取配置 gradio.host（127.0.0.1）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="监听端口，默认取配置 gradio.port（7860）",
    )
    args = parser.parse_args()

    load_env_file()  # 读根目录 .env（提供 SILRA_API_KEY 时使用）
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"配置加载失败：{exc}")
        raise SystemExit(1) from None

    # 兼容 gradio 段的两种字段命名（server_name/server_port 与 host/port）
    host = (
        args.host
        or getattr(cfg.gradio, "server_name", None)
        or getattr(cfg.gradio, "host", None)
        or "127.0.0.1"
    )
    port = (
        args.port
        or getattr(cfg.gradio, "server_port", None)
        or getattr(cfg.gradio, "port", None)
        or 7860
    )

    try:
        _load_state(cfg)
    except Exception as exc:  # _load_state 内部已分级容错，这里兜底保启动
        _state["error"] = (
            f"{exc}\n\n请确认已在仓库根目录完成 ingest / build-kg / index 三步。"
        )
        print(f"组件加载失败，服务仍将启动并在页面状态签展示中文提示：{exc}")

    demo = build_demo()
    print(f"工作台地址：http://{host}:{port}（Ctrl+C 退出）")
    try:
        demo.launch(
            server_name=str(host),
            server_port=int(port),
            css=KG_CSS,
            head=KG_HEAD,
        )
    except OSError as exc:
        print(
            f"端口 {port} 启动失败（{exc}）。常见原因是端口已被其他进程占用，可用 --port 换一个端口。"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
