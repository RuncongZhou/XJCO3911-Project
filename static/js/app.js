(function () {
    'use strict';

    const state = {
        activeModel: 'AlexNet',
        layerData: [],
        topologyData: null,
        simResult: null,
        compareResult: null
    };

    const PRESETS = {
        paper: { deviceNum: 5, bwMin: 21, bwMax: 31, perfMin: 41, perfMax: 60 },
        lowbw: { deviceNum: 5, bwMin: 10, bwMax: 20, perfMin: 41, perfMax: 60 },
        highbw: { deviceNum: 5, bwMin: 40, bwMax: 60, perfMin: 41, perfMax: 60 },
        lowperf: { deviceNum: 5, bwMin: 21, bwMax: 31, perfMin: 20, perfMax: 40 },
        highperf: { deviceNum: 5, bwMin: 21, bwMax: 31, perfMin: 60, perfMax: 100 }
    };

    const HISTORY_KEY = 'collab_infer_history_v1';
    const HISTORY_MAX = 5;

    function getRandomSeed() {
        const v = parseInt(document.getElementById('randomSeed')?.value, 10);
        return Number.isFinite(v) ? v : 1;
    }

    /** Effective bandwidth factor 0.05–1.0 for simulation (congestion / loss abstraction). */
    function getLinkEfficiency() {
        const el = document.getElementById('linkEfficiency');
        if (!el) return 1;
        const v = parseInt(el.value, 10);
        if (!Number.isFinite(v)) return 1;
        return Math.max(0.05, Math.min(1, v / 100));
    }

    function syncLinkEfficiencyLabel() {
        const el = document.getElementById('linkEfficiency');
        const lab = document.getElementById('linkEfficiencyLabel');
        if (el && lab) lab.textContent = el.value + '%';
    }

    function applyPreset(name) {
        const p = PRESETS[name];
        if (!p) return;
        document.getElementById('deviceNum').value = p.deviceNum;
        document.getElementById('bandwidthMin').value = p.bwMin;
        document.getElementById('bandwidthMax').value = p.bwMax;
        document.getElementById('performanceMin').value = p.perfMin;
        document.getElementById('performanceMax').value = p.perfMax;
        syncDeviceOrder();
        buildPartitionDesign();
        buildTopology();
        appendLog(`Preset "${name}" applied`, 'info');
    }

    function boot() {
        syncDeviceOrder();
        buildPartitionDesign();
        buildTopology();
        bindEvents();
        fetchLayers(document.getElementById('modelSelect').value);
        renderHistory();
    }

    function buildPartitionDesign() {
        const n = parseInt(document.getElementById('deviceNum').value);
        const totalLayers = state.layerData.length || 13;
        const container = document.getElementById('partitionDesignContainer');
        container.innerHTML = '';
        const hint = document.createElement('p');
        hint.className = 'partition-hint';
        hint.textContent = `Model has ${totalLayers} layers (L1–L${totalLayers}). Specify layer range per device.`;
        container.appendChild(hint);
        const layersPerDev = Math.max(1, Math.ceil(totalLayers / n));
        for (let i = 0; i < n; i++) {
            const row = document.createElement('div');
            row.className = 'partition-row';
            const startVal = Math.min(i * layersPerDev + 1, totalLayers);
            const endVal = Math.min((i + 1) * layersPerDev, totalLayers);
            row.innerHTML = `
                <label>Device ${i}:</label>
                <input type="number" class="partition-start" data-device="${i}" min="1" max="${totalLayers}" value="${startVal}" placeholder="Start">
                <span>–</span>
                <input type="number" class="partition-end" data-device="${i}" min="1" max="${totalLayers}" value="${endVal}" placeholder="End">
            `;
            container.appendChild(row);
        }
    }

    function readPartitionDesign() {
        const starts = [];
        const ends = [];
        document.querySelectorAll('.partition-start').forEach((inp) => {
            const idx = parseInt(inp.dataset.device);
            starts[idx] = parseInt(inp.value) || 1;
        });
        document.querySelectorAll('.partition-end').forEach((inp) => {
            const idx = parseInt(inp.dataset.device);
            ends[idx] = parseInt(inp.value) || 1;
        });
        const n = parseInt(document.getElementById('deviceNum').value);
        return {
            startLayers: Array.from({ length: n }, (_, i) => starts[i] ?? 1),
            endLayers: Array.from({ length: n }, (_, i) => ends[i] ?? 1)
        };
    }

    function applyAlgoPartitionToForm() {
        if (!state.simResult || !state.simResult.partitionScheme) return;
        const ps = state.simResult.partitionScheme;
        document.querySelectorAll('.partition-start').forEach((inp, i) => {
            if (ps.startLayers && ps.startLayers[i] !== undefined) inp.value = ps.startLayers[i];
        });
        document.querySelectorAll('.partition-end').forEach((inp, i) => {
            if (ps.endLayers && ps.endLayers[i] !== undefined) inp.value = ps.endLayers[i];
        });
        document.getElementById('useCustomPartition').checked = true;
        appendLog('Applied partition from last run. Edit if needed and run simulation.', 'info');
    }

    function bindEvents() {
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            state.activeModel = e.target.value;
            fetchLayers(state.activeModel);
            appendLog(`Model switched to: ${state.activeModel}`, 'info');
        });

        document.getElementById('deviceNum').addEventListener('change', () => {
            syncDeviceOrder();
            buildPartitionDesign();
            buildTopology();
        });

        document.getElementById('generateTopologyBtn').addEventListener('click', buildTopology);
        document.getElementById('randomOrderBtn').addEventListener('click', shuffleDeviceOrder);
        document.getElementById('simulateBtn').addEventListener('click', executeSimulation);
        document.getElementById('compareBtn').addEventListener('click', runComparison);
        document.getElementById('exportCompareBtn')?.addEventListener('click', exportCompareToCsv);
        document.getElementById('exportCompareJsonBtn')?.addEventListener('click', exportCompareJson);
        document.getElementById('exportSimJsonBtn')?.addEventListener('click', exportSimJson);
        document.getElementById('applyAlgoPartitionBtn').addEventListener('click', applyAlgoPartitionToForm);

        document.querySelectorAll('.preset-btn').forEach((btn) => {
            btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
        });

        document.querySelectorAll('.tab-btn').forEach((btn) => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                activateTab(tab);
        });
    });

        document.getElementById('clearLogBtn').addEventListener('click', () => {
        document.getElementById('logContainer').innerHTML = '';
    });

        const linkEff = document.getElementById('linkEfficiency');
        if (linkEff) {
            linkEff.addEventListener('input', syncLinkEfficiencyLabel);
            syncLinkEfficiencyLabel();
        }
    }

    function activateTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach((b) => {
            b.classList.toggle('active', b.dataset.tab === tabId);
        });
        document.querySelectorAll('.tab-pane').forEach((p) => {
            p.classList.remove('active');
        });
        document.getElementById(tabId + 'Tab').classList.add('active');

        if (tabId === 'model' && state.layerData.length > 0) {
            renderModelViz();
        } else if (tabId === 'topology' && state.topologyData) {
            renderTopologyViz();
        } else if (tabId === 'partition' && state.simResult) {
            renderPartitionViz();
        } else if (tabId === 'metrics' && state.simResult) {
            renderMetrics();
        } else if (tabId === 'compare' && state.compareResult) {
            renderCompareTab();
        }
    }

    async function fetchLayers(modelId) {
        try {
            appendLog(`Loading model: ${modelId}`, 'info');
            const resp = await fetch(`/api/model/${modelId}/layers`);
            const json = await resp.json();
            if (json.error) throw new Error(json.error);
            state.layerData = json.layers;
            appendLog(`Loaded ${json.totalLayers} layers`, 'success');
            buildPartitionDesign();
            renderModelViz();
        } catch (err) {
            appendLog(`Failed to load model: ${err.message}`, 'error');
        }
    }

    function renderModelViz() {
        const box = d3.select('#modelVisualization');
        box.selectAll('*').remove();

        if (state.layerData.length === 0) {
            box.append('p').text('No model data');
        return;
    }

        const w = box.node().offsetWidth - 40;
        const h = Math.max(600, state.layerData.length * 40);
        const svg = box.append('svg').attr('width', w).attr('height', h);

        const layerH = h / state.layerData.length;
        const nodeW = 150;
        const nodeH = 30;

        const groups = svg.selectAll('.layer-node')
            .data(state.layerData)
        .enter()
        .append('g')
        .attr('class', 'layer-node')
            .attr('transform', (d, i) => `translate(${w / 2 - nodeW / 2}, ${i * layerH + layerH / 2 - nodeH / 2})`);

        groups.append('rect')
            .attr('width', nodeW)
            .attr('height', nodeH)
        .attr('rx', 5)
            .attr('fill', (d) => layerTypeColor(d.type))
        .attr('stroke', '#333')
        .attr('stroke-width', 2);

        groups.append('text')
        .attr('class', 'layer-label')
            .attr('x', nodeW / 2)
            .attr('y', nodeH / 2)
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
            .text((d) => d.name || `Layer ${d.id}`);

    svg.selectAll('.layer-connection')
            .data(state.layerData.slice(0, -1))
        .enter()
        .append('line')
        .attr('class', 'layer-connection')
            .attr('x1', w / 2)
            .attr('y1', (d, i) => (i + 1) * layerH)
            .attr('x2', w / 2)
            .attr('y2', (d, i) => (i + 1) * layerH + 10)
        .attr('stroke', '#666')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)');

    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 5)
        .attr('refY', 5)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', '#666');

        groups.append('title')
            .text((d) => `${d.name}\nType: ${d.type}\nFLOPs: ${d.flops.toExponential(2)}\nData: ${d.dataSize.toFixed(2)} MB`);
    }

    function layerTypeColor(type) {
        const map = { '3': '#4dabf7', '2': '#51cf66', '1': '#ffd43b', '0': '#ff8787' };
        return map[type] || '#868e96';
    }

    function syncDeviceOrder() {
        const n = parseInt(document.getElementById('deviceNum').value);
    const container = document.getElementById('deviceOrderContainer');
    container.innerHTML = '';
        const order = Array.from({ length: n }, (_, i) => i);
        order.forEach((id) => {
        const tag = document.createElement('div');
        tag.className = 'device-tag';
            tag.textContent = `Device ${id}`;
            tag.dataset.deviceId = id;
        tag.draggable = true;
        container.appendChild(tag);
    });
        initDragDrop();
    }

    let draggedNode = null;

    function initDragDrop() {
        document.querySelectorAll('.device-tag').forEach((tag) => {
            tag.addEventListener('dragstart', (e) => {
                draggedNode = tag;
                tag.style.opacity = '0.5';
            });
            tag.addEventListener('dragover', (e) => e.preventDefault());
            tag.addEventListener('drop', (e) => {
    e.preventDefault();
                if (draggedNode !== tag) {
                    const parent = tag.parentNode;
                    const list = Array.from(parent.children);
                    const fromIdx = list.indexOf(draggedNode);
                    const toIdx = list.indexOf(tag);
                    if (fromIdx < toIdx) {
                        parent.insertBefore(draggedNode, tag.nextSibling);
        } else {
                        parent.insertBefore(draggedNode, tag);
                    }
                }
            });
            tag.addEventListener('dragend', () => {
                tag.style.opacity = '1';
                draggedNode = null;
            });
        });
    }

    function shuffleDeviceOrder() {
        const n = parseInt(document.getElementById('deviceNum').value);
        const order = Array.from({ length: n }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
    }
    const container = document.getElementById('deviceOrderContainer');
    container.innerHTML = '';
        order.forEach((id) => {
        const tag = document.createElement('div');
        tag.className = 'device-tag';
            tag.textContent = `Device ${id}`;
            tag.dataset.deviceId = id;
        tag.draggable = true;
        container.appendChild(tag);
    });
        initDragDrop();
        appendLog('Device order shuffled', 'info');
    }

    function readDeviceOrder() {
        return Array.from(document.querySelectorAll('.device-tag'))
            .map((t) => parseInt(t.dataset.deviceId));
    }

    async function buildTopology() {
        try {
            const n = parseInt(document.getElementById('deviceNum').value);
            const bwMin = parseInt(document.getElementById('bandwidthMin').value);
            const bwMax = parseInt(document.getElementById('bandwidthMax').value);
            const perfMin = parseInt(document.getElementById('performanceMin').value);
            const perfMax = parseInt(document.getElementById('performanceMax').value);

            appendLog('Generating topology...', 'info');
            const resp = await fetch('/api/device-topology', {
            method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                    deviceNum: n,
                    bandwidthRange: [bwMin, bwMax],
                    performanceRange: [perfMin, perfMax],
                    randomSeed: getRandomSeed()
            })
        });
            const json = await resp.json();
            if (!json.success) throw new Error(json.error);
            state.topologyData = json;
            appendLog('Topology generated', 'success');
            renderTopologyViz();
        } catch (err) {
            appendLog(`Topology failed: ${err.message}`, 'error');
        }
    }

    function renderTopologyViz() {
        if (!state.topologyData) return;

    const container = document.getElementById('topologyVisualization');
    container.innerHTML = '';

        const nodes = state.topologyData.nodes.map((node) => ({
        id: node.id,
        label: `${node.label}\n${node.performance.toFixed(1)} GFlops/s`,
        color: {
                background: '#6366f1',
                border: '#4f46e5',
                highlight: { background: '#8b5cf6', border: '#6366f1' }
            },
            font: { size: 14, color: 'white' },
        x: node.x,
        y: node.y,
        fixed: true
    }));

        const edges = state.topologyData.edges.map((e) => ({
            from: e.from,
            to: e.to,
            label: e.label,
            color: { color: '#666', highlight: '#6366f1' },
        width: 2,
            smooth: { type: 'continuous' }
        }));

        const net = new vis.Network(container, { nodes, edges }, {
            physics: { enabled: false },
            interaction: { dragNodes: true, dragView: true, zoomView: true },
            nodes: { shape: 'box', margin: 10 },
            edges: { arrows: { to: { enabled: true, scaleFactor: 1 } } }
        });
    }

    async function executeSimulation() {
        const throughputEl = document.getElementById('throughputValue');
        const inferenceEl = document.getElementById('inferenceTimeValue');
        const utilizationEl = document.getElementById('utilizationValue');
        const commEl = document.getElementById('commVolumeValue');
        const energyEl = document.getElementById('energyProxyValue');
        const tableBox = document.getElementById('deviceMetricsTable');
        if (throughputEl) throughputEl.textContent = '-';
        if (inferenceEl) inferenceEl.textContent = '-';
        if (utilizationEl) utilizationEl.textContent = '-';
        if (commEl) commEl.textContent = '-';
        if (energyEl) energyEl.textContent = '-';
        if (tableBox) tableBox.innerHTML = '';
        try {
            const model = document.getElementById('modelSelect').value;
            const n = parseInt(document.getElementById('deviceNum').value);
            const algo = document.getElementById('algorithmSelect').value;
            const order = readDeviceOrder();
            if (order.some((id) => !Number.isFinite(id))) {
                appendLog('Device order invalid (check Device order panel)', 'error');
                return;
            }
            const bwMin = parseInt(document.getElementById('bandwidthMin').value);
            const bwMax = parseInt(document.getElementById('bandwidthMax').value);
            const perfMin = parseInt(document.getElementById('performanceMin').value);
            const perfMax = parseInt(document.getElementById('performanceMax').value);
            const useCustom = document.getElementById('useCustomPartition').checked;

            const body = {
                model,
                deviceNum: n,
                algorithm: algo,
                deviceOrder: order,
                bandwidthRange: [bwMin, bwMax],
                performanceRange: [perfMin, perfMax],
                randomSeed: getRandomSeed(),
                linkEfficiency: getLinkEfficiency()
            };
            appendLog(`Device order: [${order.join(', ')}]`, 'info');
            if (useCustom) {
                const scheme = readPartitionDesign();
                body.partitionScheme = scheme;
                appendLog('Running simulation with custom partition', 'info');
                appendLog(`Sent partition: start=${scheme.startLayers.join(',')} end=${scheme.endLayers.join(',')}`, 'info');
            } else {
                appendLog(`Running simulation: ${algo}`, 'info');
            }

            const resp = await fetch('/api/simulate', {
            method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                cache: 'no-store'
            });
            const json = await resp.json();
            if (!json.success) throw new Error(json.error);

            state.simResult = json;
            saveHistoryEntry('simulate', body, json);
            appendLog('Simulation done', 'success');
            if (json.usedCustomPartition) appendLog('(Used your custom partition)', 'info');
            else appendLog('(Used algorithm partition)', 'info');
            appendLog(`Throughput: ${json.throughput.toFixed(4)} batches/s`, 'info');
            const infTime = (json.inferenceTime != null && Number.isFinite(json.inferenceTime)) ? json.inferenceTime.toFixed(4) : '-';
            appendLog(`Inference time: ${infTime} s`, 'info');

            activateTab('metrics');
            requestAnimationFrame(function () {
                renderMetrics(json);
                renderPartitionViz(json);
                const metricsSection = document.getElementById('metricsTab');
                if (metricsSection) metricsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        } catch (err) {
            appendLog(`Simulation failed: ${err.message}`, 'error');
        }
    }

    function renderPartitionViz(data) {
        const r = data || state.simResult;
        if (!r || !r.partitionScheme) return;

    const container = document.getElementById('partitionVisualization');
        if (!container) return;
    container.innerHTML = '';

        const div = document.createElement('div');
        div.className = 'partition-visualization';

        r.deviceMetrics.forEach((dev) => {
            const block = document.createElement('div');
            block.className = 'device-partition';

        const header = document.createElement('div');
        header.className = 'device-partition-header';
            header.textContent = `Device ${dev.deviceId}`;
            block.appendChild(header);

            const list = document.createElement('div');
            list.className = 'layers-list';

            if (dev.startLayer > 0 && dev.endLayer >= dev.startLayer) {
                for (let L = dev.startLayer; L <= dev.endLayer; L++) {
                const chip = document.createElement('div');
                chip.className = 'layer-chip active';
                    chip.textContent = `L${L}`;
                    list.appendChild(chip);
                }
            }
            block.appendChild(list);
            div.appendChild(block);
        });
        container.appendChild(div);
    }

    function renderBandwidthHeatmap(matrix) {
        const wrap = document.getElementById('bandwidthHeatmapWrap');
        if (!wrap || !matrix || !matrix.length) {
            if (wrap) wrap.innerHTML = '<p class="tab-hint">No matrix</p>';
            return;
        }
        const n = matrix.length;
        wrap.innerHTML = '';
        const tbl = document.createElement('table');
        tbl.className = 'heatmap-table';
        const trh = document.createElement('tr');
        trh.appendChild(document.createElement('th'));
        for (let j = 0; j < n; j++) {
            const th = document.createElement('th');
            th.textContent = 'D' + j;
            trh.appendChild(th);
        }
        tbl.appendChild(trh);
        let mx = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const v = Number(matrix[i][j]) || 0;
                if (v > mx) mx = v;
            }
        }
        for (let i = 0; i < n; i++) {
            const tr = document.createElement('tr');
            const th = document.createElement('th');
            th.textContent = 'D' + i;
            tr.appendChild(th);
            for (let j = 0; j < n; j++) {
                const td = document.createElement('td');
                const v = Number(matrix[i][j]) || 0;
                const a = mx > 0 ? 0.15 + 0.85 * (v / mx) : 0.3;
                td.textContent = i === j ? '—' : v.toFixed(1);
                td.style.background = i === j ? '#e2e8f0' : `rgba(31, 78, 121, ${a})`;
                td.style.color = i === j ? '#64748b' : '#fff';
                tr.appendChild(td);
            }
            tbl.appendChild(tr);
        }
        wrap.appendChild(tbl);
    }

    function renderPerfBars(perfList, deviceOrder) {
        const wrap = document.getElementById('perfBarsWrap');
        if (!wrap || !perfList) {
            if (wrap) wrap.innerHTML = '';
            return;
        }
        wrap.innerHTML = '';
        const mx = Math.max(...perfList, 1);
        const order = deviceOrder || perfList.map((_, i) => i);
        order.forEach((did, idx) => {
            const v = perfList[did] !== undefined ? perfList[did] : perfList[idx];
            const row = document.createElement('div');
            row.className = 'perf-bar-row';
            const label = document.createElement('span');
            label.className = 'perf-bar-label';
            label.textContent = 'Device ' + did;
            const bar = document.createElement('div');
            bar.className = 'perf-bar-track';
            const fill = document.createElement('div');
            fill.className = 'perf-bar-fill';
            fill.style.width = (100 * Number(v) / mx) + '%';
            fill.title = Number(v).toFixed(2) + ' GFlops/s';
            bar.appendChild(fill);
            const val = document.createElement('span');
            val.className = 'perf-bar-val';
            val.textContent = Number(v).toFixed(2);
            row.appendChild(label);
            row.appendChild(bar);
            row.appendChild(val);
            wrap.appendChild(row);
        });
    }

    function renderPipelineStages(r) {
        const wrap = document.getElementById('pipelineStageWrap');
        if (!wrap) return;
        wrap.innerHTML = '';
        if (!r.stageTimes || !r.stageTimes.length) {
            wrap.innerHTML = '<p class="tab-hint">No stage times</p>';
            return;
        }
        const mx = Math.max(...r.stageTimes.map((x) => (x == null ? 0 : x)), 1e-9);
        const order = r.deviceOrder || [];
        r.stageTimes.forEach((t, i) => {
            if (t == null) return;
            const row = document.createElement('div');
            row.className = 'perf-bar-row';
            const label = document.createElement('span');
            label.className = 'perf-bar-label';
            const devId = order[i] !== undefined ? order[i] : i;
            label.textContent = 'Stage ' + i + ' (device ' + devId + ')';
            const bar = document.createElement('div');
            bar.className = 'perf-bar-track';
            const fill = document.createElement('div');
            fill.className = 'perf-bar-fill perf-bar-fill-stage';
            if (r.bottleneckStageIndex === i) fill.classList.add('is-bottleneck');
            fill.style.width = (100 * t / mx) + '%';
            bar.appendChild(fill);
            const val = document.createElement('span');
            val.className = 'perf-bar-val';
            val.textContent = t.toFixed(6) + ' s';
            row.appendChild(label);
            row.appendChild(bar);
            row.appendChild(val);
            wrap.appendChild(row);
        });
        if (r.bottleneckDeviceId != null) {
            const p = document.createElement('p');
            p.className = 'bottleneck-note';
            p.textContent = 'Throughput bottleneck: pipeline stage ' + r.bottleneckStageIndex +
                ' (device ' + r.bottleneckDeviceId + ') — max stage time in pipeline model.';
            wrap.appendChild(p);
        }
    }

    function renderMetrics(data) {
        const r = data || state.simResult;
        const throughputEl = document.getElementById('throughputValue');
        const inferenceEl = document.getElementById('inferenceTimeValue');
        const utilizationEl = document.getElementById('utilizationValue');
        const tableBox = document.getElementById('deviceMetricsTable');
        const seedEcho = document.getElementById('seedEcho');
        const exportSim = document.getElementById('exportSimJsonBtn');
        if (!throughputEl || !inferenceEl || !utilizationEl || !tableBox) return;
        if (!r) return;
        throughputEl.textContent =
            r.throughput > 0 ? r.throughput.toFixed(4) : '-';
        inferenceEl.textContent =
            (r.inferenceTime != null && Number.isFinite(r.inferenceTime)) ? r.inferenceTime.toFixed(4) : '-';

        const util = r.devicePerformance
            ? (r.devicePerformance.reduce((a, b) => a + b, 0) / r.devicePerformance.length /
               Math.max(...r.devicePerformance)) * 100
            : null;
        utilizationEl.textContent =
            util !== null ? util.toFixed(2) : '-';

        if (seedEcho) {
            const le = r.linkEfficiency != null ? r.linkEfficiency : getLinkEfficiency();
            seedEcho.textContent = 'Random seed: ' + (r.randomSeed != null ? r.randomSeed : getRandomSeed()) +
                ' · Link efficiency: ' + (typeof le === 'number' ? (le * 100).toFixed(0) + '%' : '-') +
                ' (applied to bandwidth in evaluation).';
        }
        const commEl = document.getElementById('commVolumeValue');
        const energyEl = document.getElementById('energyProxyValue');
        const hintEl = document.getElementById('energyProxyHint');
        const ps = r.pipelineStats;
        if (commEl) {
            commEl.textContent = (ps && ps.totalCommunicationMB != null)
                ? ps.totalCommunicationMB.toFixed(4)
                : '-';
        }
        if (energyEl) {
            energyEl.textContent = (ps && ps.energyProxyJoules != null)
                ? ps.energyProxyJoules.toFixed(4)
                : '-';
        }
        if (hintEl) {
            if (ps && ps.energyProxyNote) {
                hintEl.style.display = 'block';
                hintEl.textContent = ps.energyProxyNote;
            } else {
                hintEl.style.display = 'none';
                hintEl.textContent = '';
            }
        }
        if (exportSim) exportSim.style.display = 'inline-block';

        tableBox.innerHTML = '';

        if (r.deviceMetrics) {
    const table = document.createElement('table');
            table.innerHTML = `
                <thead><tr>
                    <th>Device ID</th>
                    <th>Performance (GFlops/s)</th>
                    <th>Start layer</th>
                    <th>End layer</th>
                    <th>Layers</th>
                </tr></thead>
                <tbody></tbody>
            `;
            const tbody = table.querySelector('tbody');
            r.deviceMetrics.forEach((dev) => {
        const row = document.createElement('tr');
                const layerCount = dev.endLayer >= dev.startLayer ? dev.endLayer - dev.startLayer + 1 : 0;
        row.innerHTML = `
                    <td>${dev.deviceId}</td>
                    <td>${dev.performance.toFixed(2)}</td>
                    <td>${dev.startLayer}</td>
                    <td>${dev.endLayer}</td>
                    <td>${layerCount}</td>
        `;
        tbody.appendChild(row);
    });
            tableBox.appendChild(table);
        }

        renderBandwidthHeatmap(r.bandwidthMatrix);
        renderPerfBars(r.devicePerformance, r.deviceOrder);
        renderPipelineStages(r);
    }

    function renderCompareCharts(results) {
        const wrap = document.getElementById('compareChartsWrap');
        if (!wrap || !results) {
            if (wrap) wrap.innerHTML = '';
            return;
        }
        wrap.innerHTML = '';
        const algos = Object.keys(results);
        const tpMax = Math.max(...algos.map((a) => results[a].throughput || 0), 1e-9);
        const tiVals = algos.map((a) => results[a].inferenceTime).filter((x) => x != null && Number.isFinite(x));
        const tiMax = tiVals.length ? Math.max(...tiVals) : 1e-9;

        const h4a = document.createElement('h4');
        h4a.className = 'metrics-subtitle';
        h4a.textContent = 'Throughput';
        wrap.appendChild(h4a);
        algos.forEach((algo) => {
            const tp = results[algo].throughput || 0;
            const row = document.createElement('div');
            row.className = 'perf-bar-row';
            const label = document.createElement('span');
            label.className = 'perf-bar-label';
            label.textContent = algo;
            const bar = document.createElement('div');
            bar.className = 'perf-bar-track';
            const fill = document.createElement('div');
            fill.className = 'perf-bar-fill compare-tp';
            fill.style.width = (100 * tp / tpMax) + '%';
            bar.appendChild(fill);
            const val = document.createElement('span');
            val.className = 'perf-bar-val';
            val.textContent = tp.toFixed(4);
            row.appendChild(label);
            row.appendChild(bar);
            row.appendChild(val);
            wrap.appendChild(row);
        });

        const h4b = document.createElement('h4');
        h4b.className = 'metrics-subtitle';
        h4b.textContent = 'Inference time (lower is better)';
        wrap.appendChild(h4b);
        algos.forEach((algo) => {
            const ti = results[algo].inferenceTime;
            const row = document.createElement('div');
            row.className = 'perf-bar-row';
            const label = document.createElement('span');
            label.className = 'perf-bar-label';
            label.textContent = algo;
            const bar = document.createElement('div');
            bar.className = 'perf-bar-track';
            const fill = document.createElement('div');
            fill.className = 'perf-bar-fill compare-ti';
            const tnum = (ti != null && Number.isFinite(ti)) ? ti : 0;
            fill.style.width = (100 * tnum / tiMax) + '%';
            bar.appendChild(fill);
            const val = document.createElement('span');
            val.className = 'perf-bar-val';
            val.textContent = (ti != null && Number.isFinite(ti)) ? ti.toFixed(4) : '-';
            row.appendChild(label);
            row.appendChild(bar);
            row.appendChild(val);
            wrap.appendChild(row);
        });

        const hasPs = algos.some((a) => results[a].pipelineStats && results[a].pipelineStats.totalCommunicationMB != null);
        if (hasPs) {
            const commMax = Math.max(
                ...algos.map((a) => (results[a].pipelineStats && results[a].pipelineStats.totalCommunicationMB) || 0),
                1e-9
            );
            const h4c = document.createElement('h4');
            h4c.className = 'metrics-subtitle';
            h4c.textContent = 'Activation traffic (MB, sum per stage)';
            wrap.appendChild(h4c);
            algos.forEach((algo) => {
                const ps = results[algo].pipelineStats;
                const mb = (ps && ps.totalCommunicationMB != null) ? ps.totalCommunicationMB : 0;
                const row = document.createElement('div');
                row.className = 'perf-bar-row';
                const label = document.createElement('span');
                label.className = 'perf-bar-label';
                label.textContent = algo;
                const bar = document.createElement('div');
                bar.className = 'perf-bar-track';
                const fill = document.createElement('div');
                fill.className = 'perf-bar-fill compare-comm';
                fill.style.width = (100 * mb / commMax) + '%';
                bar.appendChild(fill);
                const val = document.createElement('span');
                val.className = 'perf-bar-val';
                val.textContent = mb.toFixed(4);
                row.appendChild(label);
                row.appendChild(bar);
                row.appendChild(val);
                wrap.appendChild(row);
            });

            const enMax = Math.max(
                ...algos.map((a) => (results[a].pipelineStats && results[a].pipelineStats.energyProxyJoules) || 0),
                1e-9
            );
            const h4e = document.createElement('h4');
            h4e.className = 'metrics-subtitle';
            h4e.textContent = 'Energy proxy J (illustrative)';
            wrap.appendChild(h4e);
            algos.forEach((algo) => {
                const ps = results[algo].pipelineStats;
                const ej = (ps && ps.energyProxyJoules != null) ? ps.energyProxyJoules : 0;
                const row = document.createElement('div');
                row.className = 'perf-bar-row';
                const label = document.createElement('span');
                label.className = 'perf-bar-label';
                label.textContent = algo;
                const bar = document.createElement('div');
                bar.className = 'perf-bar-track';
                const fill = document.createElement('div');
                fill.className = 'perf-bar-fill compare-energy';
                bar.appendChild(fill);
                fill.style.width = (100 * ej / enMax) + '%';
                const val = document.createElement('span');
                val.className = 'perf-bar-val';
                val.textContent = ej.toFixed(4);
                row.appendChild(label);
                row.appendChild(bar);
                row.appendChild(val);
                wrap.appendChild(row);
            });
        }
    }

    function renderCompareTab(data) {
        const container = document.getElementById('compareTableContainer');
        const exportBtn = document.getElementById('exportCompareBtn');
        const exportJsonBtn = document.getElementById('exportCompareJsonBtn');
        if (!container) return;
        container.innerHTML = '';
        const r = data || state.compareResult;
        if (!r || !r.results) {
            if (exportBtn) exportBtn.style.display = 'none';
            if (exportJsonBtn) exportJsonBtn.style.display = 'none';
            const cw = document.getElementById('compareChartsWrap');
            if (cw) cw.innerHTML = '';
            return;
        }
        if (exportBtn) exportBtn.style.display = 'inline-block';
        if (exportJsonBtn) exportJsonBtn.style.display = 'inline-block';

        renderCompareCharts(r.results);

        const results = r.results;
        const table = document.createElement('table');
        table.innerHTML = `
            <thead><tr>
                <th>Algorithm</th>
                <th>Throughput (batches/s)</th>
                <th>Inference time (s)</th>
                <th>Comm (MB)</th>
                <th>Energy proxy (J)</th>
            </tr></thead>
            <tbody></tbody>
        `;
        const tbody = table.querySelector('tbody');
        Object.entries(results).forEach(([algo, res]) => {
            const row = document.createElement('tr');
            const tp = res.throughput !== undefined ? res.throughput.toFixed(4) : '-';
            const ti = (res.inferenceTime != null && Number.isFinite(res.inferenceTime)) ? res.inferenceTime.toFixed(4) : '-';
            const ps = res.pipelineStats;
            const cm = (ps && ps.totalCommunicationMB != null) ? ps.totalCommunicationMB.toFixed(4) : '-';
            const ej = (ps && ps.energyProxyJoules != null) ? ps.energyProxyJoules.toFixed(4) : '-';
            row.innerHTML = `<td>${algo}</td><td>${tp}</td><td>${ti}</td><td>${cm}</td><td>${ej}</td>`;
            tbody.appendChild(row);
        });
        container.appendChild(table);
        const seedNote = document.createElement('p');
        seedNote.className = 'tab-hint';
        const le = r.linkEfficiency != null ? r.linkEfficiency : getLinkEfficiency();
        seedNote.textContent = 'Random seed: ' + (r.randomSeed != null ? r.randomSeed : getRandomSeed()) +
            ' · Link efficiency: ' + (typeof le === 'number' ? (le * 100).toFixed(0) + '%' : '-');
        container.appendChild(seedNote);
    }

    function exportSimJson() {
        const r = state.simResult;
        if (!r) return;
        const payload = {
            exportedAt: new Date().toISOString(),
            config: {
                model: document.getElementById('modelSelect').value,
                deviceNum: parseInt(document.getElementById('deviceNum').value, 10),
                algorithm: document.getElementById('algorithmSelect').value,
                bandwidthRange: [
                    parseInt(document.getElementById('bandwidthMin').value, 10),
                    parseInt(document.getElementById('bandwidthMax').value, 10)
                ],
                performanceRange: [
                    parseInt(document.getElementById('performanceMin').value, 10),
                    parseInt(document.getElementById('performanceMax').value, 10)
                ],
                randomSeed: getRandomSeed(),
                linkEfficiency: getLinkEfficiency(),
                deviceOrder: readDeviceOrder()
            },
            result: r
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'simulation_' + payload.config.model + '_' + Date.now() + '.json';
        a.click();
        URL.revokeObjectURL(a.href);
    }

    function exportCompareJson() {
        const r = state.compareResult;
        if (!r || !r.results) return;
        const payload = {
            exportedAt: new Date().toISOString(),
            config: {
                model: document.getElementById('modelSelect').value,
                deviceNum: parseInt(document.getElementById('deviceNum').value, 10),
                bandwidthRange: [
                    parseInt(document.getElementById('bandwidthMin').value, 10),
                    parseInt(document.getElementById('bandwidthMax').value, 10)
                ],
                performanceRange: [
                    parseInt(document.getElementById('performanceMin').value, 10),
                    parseInt(document.getElementById('performanceMax').value, 10)
                ],
                randomSeed: getRandomSeed(),
                linkEfficiency: getLinkEfficiency(),
                deviceOrder: readDeviceOrder()
            },
            result: r
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'compare_' + payload.config.model + '_' + Date.now() + '.json';
        a.click();
        URL.revokeObjectURL(a.href);
    }

    function saveHistoryEntry(kind, requestBody, responseJson) {
        try {
            let h = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
            const entry = {
                t: Date.now(),
                kind,
                label: kind === 'compare'
                    ? 'Compare: HM ' + (responseJson.results?.HiveMind?.throughput?.toFixed(2) || '?') +
                      ' / EP ' + (responseJson.results?.EdgePipe?.throughput?.toFixed(2) || '?')
                    : 'Sim: ' + (responseJson.throughput != null ? responseJson.throughput.toFixed(2) : '?') + ' batches/s'
            };
            h.unshift(entry);
            h = h.slice(0, HISTORY_MAX);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
            renderHistory();
        } catch (e) { /* ignore */ }
    }

    function renderHistory() {
        const wrap = document.getElementById('historyWrap');
        if (!wrap) return;
        try {
            const h = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
            wrap.innerHTML = '';
            if (!h.length) {
                wrap.innerHTML = '<p class="tab-hint">No saved entries yet.</p>';
                return;
            }
            h.forEach((item) => {
                const div = document.createElement('div');
                div.className = 'history-item';
                const time = new Date(item.t).toLocaleString();
                div.textContent = time + ' — ' + item.label + ' (' + item.kind + ')';
                wrap.appendChild(div);
            });
        } catch (e) {
            wrap.innerHTML = '';
        }
    }

    function exportCompareToCsv() {
        const r = state.compareResult;
        if (!r || !r.results) return;
        const model = document.getElementById('modelSelect')?.value || 'AlexNet';
        const n = parseInt(document.getElementById('deviceNum')?.value || 5);
        const bwMin = parseInt(document.getElementById('bandwidthMin')?.value || 21);
        const bwMax = parseInt(document.getElementById('bandwidthMax')?.value || 31);
        const perfMin = parseInt(document.getElementById('performanceMin')?.value || 41);
        const perfMax = parseInt(document.getElementById('performanceMax')?.value || 60);
        const le = r.linkEfficiency != null ? r.linkEfficiency : getLinkEfficiency();
        const rows = [['model', 'deviceNum', 'bandwidthMin', 'bandwidthMax', 'perfMin', 'perfMax', 'linkEfficiency', 'algorithm', 'throughput', 'inferenceTime', 'commMB', 'energyProxyJ']];
        Object.entries(r.results).forEach(([algo, res]) => {
            const ps = res.pipelineStats;
            const cm = (ps && ps.totalCommunicationMB != null) ? ps.totalCommunicationMB : '';
            const ej = (ps && ps.energyProxyJoules != null) ? ps.energyProxyJoules : '';
            rows.push([model, n, bwMin, bwMax, perfMin, perfMax, le, algo, res.throughput ?? '', res.inferenceTime ?? '', cm, ej]);
        });
        const csv = rows.map(r => r.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'compare_' + model + '_' + n + 'dev_' + Date.now() + '.csv';
        a.click();
        URL.revokeObjectURL(a.href);
    }

    async function runComparison() {
        const container = document.getElementById('compareTableContainer');
        if (container) container.innerHTML = '';
        try {
            const model = document.getElementById('modelSelect').value;
            const n = parseInt(document.getElementById('deviceNum').value);

            appendLog('Comparing algorithms...', 'info');

            const bwMin = parseInt(document.getElementById('bandwidthMin').value);
            const bwMax = parseInt(document.getElementById('bandwidthMax').value);
            const perfMin = parseInt(document.getElementById('performanceMin').value);
            const perfMax = parseInt(document.getElementById('performanceMax').value);

            const order = readDeviceOrder();
            const compareBody = {
                model,
                deviceNum: n,
                deviceOrder: order,
                algorithms: ['HiveMind', 'EdgePipe'],
                bandwidthRange: [bwMin, bwMax],
                performanceRange: [perfMin, perfMax],
                randomSeed: getRandomSeed(),
                linkEfficiency: getLinkEfficiency()
            };
            const resp = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(compareBody),
                cache: 'no-store'
            });
            const json = await resp.json();
            if (!json.success) throw new Error(json.error);

            state.compareResult = json;
            saveHistoryEntry('compare', compareBody, json);
            appendLog('Comparison done', 'success');
            Object.entries(json.results).forEach(([algo, res]) => {
                const it = (res.inferenceTime != null && Number.isFinite(res.inferenceTime)) ? res.inferenceTime.toFixed(4) : '-';
                appendLog(`${algo}: throughput=${res.throughput.toFixed(4)}, inference time=${it}`, 'info');
            });
            if (container) container.innerHTML = '';
            activateTab('compare');
            renderCompareTab(json);
        } catch (err) {
            appendLog(`Comparison failed: ${err.message}`, 'error');
        }
    }

    function appendLog(msg, level = 'info') {
        const box = document.getElementById('logContainer');
        const entry = document.createElement('div');
        entry.className = `log-entry ${level}`;
        entry.innerHTML = `<span class="log-time">[${new Date().toLocaleTimeString()}]</span>${msg}`;
        box.appendChild(entry);
        box.scrollTop = box.scrollHeight;
    }

    document.addEventListener('DOMContentLoaded', boot);
})();
