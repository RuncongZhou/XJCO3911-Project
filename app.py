"""
Multi-Device DNN Collaborative Inference Visualization - Backend Service
"""
from pathlib import Path
from functools import wraps
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys

# Resolve project root and extend module path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Model -> (HiveMind_module, EdgePipe_module, csv_filename)
_MODEL_ALGO_MAP = {
    'AlexNet': ('HiveMind_AlexNet', 'EdgePipe_AlexNet', 'AlexNet'),
    'Vgg19': ('HiveMind_Vgg19', 'EdgePipe_Vgg19', 'vgg19'),
    'YOLONet': ('HiveMind_YOLONet', 'EdgePipe_YOLONet', 'YOLONet'),
    'SqueezeNet': ('HiveMind_AlexNet', 'EdgePipe_AlexNet', 'SqueezeNet'),
}
# Fallback for Resnet50, Resnet101, SqueezeNet
_DEFAULT_ALGO = ('HiveMind_AlexNet', 'EdgePipe_AlexNet', 'AlexNet')

# Illustrative energy proxy for multi-objective discussion (not hardware-calibrated).
_ENERGY_J_PER_COMP_S = 18.0
_ENERGY_J_PER_COMM_MB = 0.2
_ENERGY_PROXY_NOTE = (
    'Linear proxy E = alpha*sum(T_comp) + beta*M_comm_MB '
    '(alpha=%.0f J/s, beta=%.2f J/MB); illustrative, not hardware-calibrated.'
    % (_ENERGY_J_PER_COMP_S, _ENERGY_J_PER_COMM_MB)
)


def _get_engine_classes(model_name: str, algo: str):
    """Load HiveMind/Pipeline from model-specific module."""
    entry = _MODEL_ALGO_MAP.get(model_name, _DEFAULT_ALGO)
    hm_mod, ep_mod, _ = entry
    if algo == 'HiveMind':
        mod = __import__(hm_mod, fromlist=['HiveMind'])
        return mod.HiveMind
    elif algo == 'EdgePipe':
        mod = __import__(ep_mod, fromlist=['Pipeline'])
        return mod.Pipeline
    return None


# Flask application factory pattern
app = Flask(__name__)
CORS(app)


class RuntimeConfig:
    """Holds active simulation parameters."""
    __slots__ = ('model', 'device_num', 'algorithm', 'device_order', 'partition_scheme')

    def __init__(self):
        self.model = 'AlexNet'
        self.device_num = 5
        self.algorithm = 'HiveMind'
        self.device_order = None
        self.partition_scheme = None


_runtime_config = RuntimeConfig()


def _handle_api_errors(f):
    """Decorator for uniform API error handling."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            import traceback
            return jsonify(success=False, error=str(e), traceback=traceback.format_exc()), 500
    return wrapper


def _resolve_data_path(model_name: str) -> Path:
    """Build path to model CSV data file. Prefer model-specific CSV when it exists."""
    model_csv = _PROJECT_ROOT / "data" / f"{model_name}.csv"
    if model_csv.exists():
        return model_csv
    entry = _MODEL_ALGO_MAP.get(model_name, _DEFAULT_ALGO)
    csv_name = entry[2]
    return _PROJECT_ROOT / "data" / f"{csv_name}.csv"


def _parse_layer_records(df: pd.DataFrame) -> list:
    """Convert DataFrame rows into layer descriptor dicts."""
    return [
        {
                'id': idx,
                'name': row.get('module name', f'Layer {idx}'),
                'type': row.get('type', 'Unknown'),
                'flops': float(row.get('Flops', 0)),
                'dataSize': float(row.get('DataSize', 0)),
                'inputShape': str(row.get('input shape', '')),
                'outputShape': str(row.get('output shape', '')),
                'params': int(row.get('params', 0))
            }
        for idx, row in df.iterrows()
    ]


def _to_numeric_series(ser):
    """Convert series to float, handling string values with commas (e.g. '168,805,248.00')."""
    return pd.to_numeric(ser.astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)


def _inject_model_layer_data(engine, model_name: str, algo: str):
    """Load model's CSV and inject layer_flops, layer_data_sizes, LAYER_COUNT into engine.
    Ensures algorithm uses the actual model's layer count (e.g. 21 for SqueezeNet) instead of hardcoded 13."""
    path = _resolve_data_path(model_name)
    if not path.exists():
        return
    df = pd.read_csv(path)
    flops = _to_numeric_series(df['Flops']).values.astype(np.float64)
    data_sizes = _to_numeric_series(df['DataSize']).values.astype(np.float64)
    L = len(flops)
    engine.LAYER_COUNT = L
    # HiveMind/EdgePipe may access data_sizes[L]; ensure it exists (use last layer's output size)
    if len(data_sizes) == L:
        data_sizes = np.append(data_sizes, data_sizes[-1] if L > 0 else 0)
    if algo == 'HiveMind':
        engine.layer_flops = flops
        engine.layer_data_sizes = data_sizes
    else:
        engine.flops_per_layer = flops
        engine.data_per_layer = data_sizes


def _engine_layer_arrays(engine):
    """HiveMind uses layer_flops/layer_data_sizes; EdgePipe uses flops_per_layer/data_per_layer."""
    if getattr(engine, 'layer_flops', None) is not None:
        return engine.layer_flops, engine.layer_data_sizes
    return engine.flops_per_layer, engine.data_per_layer


def _populate_bandwidth_matrix(engine, device_count: int, bw_min: int, bw_max: int):
    """Fill symmetric bandwidth matrix from random values."""
    n_pairs = device_count * (device_count - 1) // 2
    vals = np.random.randint(bw_min, bw_max, size=n_pairs)
    engine.band = np.zeros((device_count, device_count))
    offset = 0
    for i in range(device_count - 1):
        for j in range(i + 1, device_count):
            engine.band[i, j] = engine.band[j, i] = vals[offset]
            offset += 1


def _inject_device_context(device_order_list, device_count: int, model_name: str, algo: str, bw_range=None):
    """Set module-level device order and first-hop bandwidth for algorithm modules.
    If bw_range is given, first_hop uses that range so the first hop is not a fixed bottleneck."""
    entry = _MODEL_ALGO_MAP.get(model_name, _DEFAULT_ALGO)
    hm_mod_name, ep_mod_name, _ = entry
    hm_mod = __import__(hm_mod_name, fromlist=['HiveMind'])
    ep_mod = __import__(ep_mod_name, fromlist=['Pipeline'])

    setattr(hm_mod, 's', np.array(device_order_list))
    setattr(ep_mod, 's', np.array(device_order_list))

    if bw_range and len(bw_range) >= 2:
        low, high = max(1, bw_range[0]), max(2, bw_range[1]) + 1
        first_hop = np.random.randint(low, high, size=device_count)
    else:
        first_hop = np.random.randint(1, 50, size=device_count)
        mask = np.random.randint(device_count - 1, size=device_count // 2)
        first_hop[mask] = 0.01
    setattr(hm_mod, 'first_band', first_hop.copy())
    setattr(ep_mod, 'first_band', first_hop.copy())


def _redistribute_partition_to_avoid_relay(start_layers: list, end_layers: list,
                                           total_layers: int, device_count: int) -> tuple:
    """When relay devices (end_layer=-1) exist and total_layers >= device_count,
    redistribute layers uniformly so each device gets at least 1 layer."""
    relay_count = sum(1 for e in end_layers if e == -1)
    if relay_count == 0 or total_layers < device_count:
        return start_layers, end_layers

    layers_per_dev = total_layers // device_count
    remainder = total_layers % device_count
    ls, le = [], []
    idx = 1
    for i in range(device_count):
        count = layers_per_dev + (1 if i < remainder else 0)
        ls.append(idx)
        le.append(idx + count - 1)
        idx += count
    return ls, le


def _extract_partition_hivemind(engine, device_count: int) -> dict:
    """Use HiveMind Ls/Le when set by enhancedDijkstra*, else derive from cost/npi."""
    if hasattr(engine, 'Ls') and hasattr(engine, 'Le') and len(engine.Ls) == device_count:
        return {'startLayers': list(map(int, engine.Ls)), 'endLayers': list(map(int, engine.Le))}

    ls = [0] * device_count
    le = [0] * device_count
    if not (hasattr(engine, 'npi') and hasattr(engine, 'cost')):
        return {'startLayers': ls, 'endLayers': le}
    min_cost, idx = float('inf'), 0
    for j in range(1, engine.L + 1):
        if engine.cost[1, j] < min_cost:
            min_cost, idx = engine.cost[1, j], j
    ls[0], le[0] = 1, idx
    for i in range(1, device_count):
        le[i] = int(engine.npi[i, idx])
        ls[i] = idx + 1
        if le[i] != -1:
            idx = int(le[i])
    return {'startLayers': ls, 'endLayers': le}


def _evaluate_partition(
    start_layers: list, end_layers: list,
    device_order: list, band: np.ndarray, perf: np.ndarray, first_band: np.ndarray,
    layer_flops: np.ndarray, layer_data_sizes: np.ndarray
) -> tuple:
    """
    Evaluate throughput and inference time for a given partition (pipeline model).
    start_layers, end_layers: 1-indexed inclusive, e.g. [1,4,7], [3,6,13] means dev0: L1-L3, dev1: L4-L6, dev2: L7-L13
    """
    nd = len(device_order)
    s = np.array(device_order)
    stage_times = []
    infer_parts = []

    stage_comm_bytes = []
    stage_comp_seconds = []
    for i in range(nd):
        ls, le = start_layers[i], end_layers[i]
        if ls <= 0 or le < ls:
            stage_times.append(float('inf'))
            infer_parts.append(float('inf'))
            stage_comm_bytes.append(0.0)
            stage_comp_seconds.append(0.0)
            continue
        comp = np.sum(layer_flops[ls - 1:le] / perf[s[i]] / 1e9)
        stage_comp_seconds.append(float(comp))
        if i == 0:
            nbytes = float(layer_data_sizes[0])
            xfer_in = nbytes / first_band[s[0]]
            stage_t = max(xfer_in, comp)
            infer_parts.append(xfer_in + comp)
        else:
            data_idx = ls - 1
            nbytes = float(layer_data_sizes[data_idx])
            xfer_in = nbytes / band[s[i - 1], s[i]]
            stage_t = max(xfer_in, comp)
            infer_parts.append(xfer_in + comp)
        stage_comm_bytes.append(nbytes)
        stage_times.append(stage_t)

    throughput = 1 / max(stage_times) if stage_times and max(stage_times) > 0 else 0
    inference_time = sum(infer_parts) if all(p != float('inf') for p in infer_parts) else float('inf')
    if stage_times:
        vals = [float(t) if np.isfinite(t) else -1.0 for t in stage_times]
        bottleneck_stage_index = int(np.argmax(vals))
    else:
        bottleneck_stage_index = -1
    total_comm = float(sum(stage_comm_bytes))
    total_comp = float(sum(stage_comp_seconds))
    energy_proxy = _ENERGY_J_PER_COMP_S * total_comp + _ENERGY_J_PER_COMM_MB * (total_comm / 1e6)
    pipeline_stats = {
        'totalCommunicationBytes': total_comm,
        'totalCommunicationMB': total_comm / 1e6,
        'stageCommunicationBytes': stage_comm_bytes,
        'totalComputeSeconds': total_comp,
        'stageComputeSeconds': stage_comp_seconds,
        'energyProxyJoules': energy_proxy,
    }
    return throughput, inference_time, stage_times, bottleneck_stage_index, pipeline_stats


def _layer_arrays_for_model(model_name: str):
    """Layer flops and data sizes (with trailing size for pipeline eval)."""
    path = _resolve_data_path(model_name)
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    layer_flops = _to_numeric_series(df['Flops']).values.astype(np.float64)
    layer_data_sizes = _to_numeric_series(df['DataSize']).values.astype(np.float64)
    L = len(layer_flops)
    if len(layer_data_sizes) == L:
        layer_data_sizes = np.append(layer_data_sizes, layer_data_sizes[-1] if L > 0 else 0.0)
    return layer_flops, layer_data_sizes


def _first_hop_for_stages(device_count: int, model_name: str, bw_range: list) -> np.ndarray:
    entry = _MODEL_ALGO_MAP.get(model_name, _DEFAULT_ALGO)
    hm_mod = __import__(entry[0], fromlist=['HiveMind'])
    fh = getattr(hm_mod, 'first_band', None)
    if fh is not None:
        a = np.asarray(fh, dtype=np.float64).reshape(-1)[:device_count]
        if a.size == device_count:
            return a
    mid = max(1.0, float(bw_range[0] + bw_range[1]) / 2.0)
    return np.ones(device_count, dtype=np.float64) * mid


def _extract_partition_edgepipe(engine, device_count: int) -> dict:
    """Derive layer partition from EdgePipe Ls/Le (set by dynamic_planning), else uniform fallback."""
    if hasattr(engine, 'Ls') and hasattr(engine, 'Le') and len(engine.Ls) == device_count:
        return {'startLayers': list(map(int, engine.Ls)), 'endLayers': list(map(int, engine.Le))}

    ls, le = [0] * device_count, [0] * device_count
    if not (hasattr(engine, 'L')):
        return {'startLayers': ls, 'endLayers': le}

    layers_per_dev = engine.L // device_count
    remainder = engine.L % device_count
    layer_idx = 0
    for i in range(device_count):
        ls[i] = layer_idx + 1
        count = layers_per_dev + (1 if i < remainder else 0)
        layer_idx += count
        le[i] = layer_idx
    return {'startLayers': ls, 'endLayers': le}


def _serialize_pipeline_stats(stats: dict) -> dict:
    """JSON-safe pipeline stats for API responses."""
    if not stats:
        return None
    return {
        'totalCommunicationMB': round(stats['totalCommunicationMB'], 6),
        'energyProxyJoules': round(stats['energyProxyJoules'], 6),
        'totalComputeSeconds': round(stats['totalComputeSeconds'], 6),
        'stageCommunicationMB': [round(b / 1e6, 8) for b in stats['stageCommunicationBytes']],
        'energyProxyNote': _ENERGY_PROXY_NOTE,
    }


def _pipeline_stats_from_engine(engine, model_name: str, algo: str, device_count: int,
                                device_order: list, bw_range: list, link_efficiency: float) -> dict:
    """Same partition + comm/energy stats as /api/simulate (for compare tab)."""
    if algo == 'HiveMind':
        partition = _extract_partition_hivemind(engine, device_count)
        L = getattr(engine, 'LAYER_COUNT', getattr(engine, 'L', 13))
        sl, el = _redistribute_partition_to_avoid_relay(
            partition['startLayers'], partition['endLayers'], L, device_count
        )
        if (sl, el) != (partition['startLayers'], partition['endLayers']):
            partition = {'startLayers': sl, 'endLayers': el}
            entry = _MODEL_ALGO_MAP.get(model_name, _DEFAULT_ALGO)
            hm_mod = __import__(entry[0], fromlist=['HiveMind'])
            first_hop = getattr(hm_mod, 'first_band', np.ones(device_count) * 25)
        else:
            first_hop = _first_hop_for_stages(device_count, model_name, bw_range)
        lf, ld = engine.layer_flops, engine.layer_data_sizes
    else:
        partition = _extract_partition_edgepipe(engine, device_count)
        first_hop = _first_hop_for_stages(device_count, model_name, bw_range)
        lf, ld = _engine_layer_arrays(engine)
    band_eval = np.asarray(engine.band, dtype=np.float64) * link_efficiency
    perf = np.asarray(engine.fn, dtype=np.float64)
    fh = np.asarray(first_hop, dtype=np.float64).reshape(-1)[:device_count] * link_efficiency
    *_, stats = _evaluate_partition(
        partition['startLayers'], partition['endLayers'], list(device_order),
        band_eval, perf, fh, lf, ld
    )
    return stats


@app.route('/')
def serve_index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def api_models():
    """Return available model identifiers."""
    return jsonify(['AlexNet', 'Vgg19', 'YOLONet', 'Resnet50', 'Resnet101', 'SqueezeNet'])


@app.route('/api/model/<model_name>/layers', methods=['GET'])
def api_model_layers(model_name):
    """Return layer metadata for given model."""
    path = _resolve_data_path(model_name)
    if not path.exists():
        return jsonify({'error': f'Model {model_name} not found'}), 404
    df = pd.read_csv(path)
    layers = _parse_layer_records(df)
    return jsonify(
        model=model_name,
        layers=layers,
        totalLayers=len(layers)
    )


@app.route('/api/simulate', methods=['POST'])
@_handle_api_errors
def api_simulate():
    """Run distributed inference simulation. Supports custom partition design."""
    payload = request.json or {}
    model_name = payload.get('model', 'AlexNet')
    device_count = int(payload.get('deviceNum', 5))
    algo = payload.get('algorithm', 'HiveMind')
    device_order = payload.get('deviceOrder', list(range(device_count)))
    bw_range = payload.get('bandwidthRange', [21, 31])
    perf_range = payload.get('performanceRange', [41, 60])
    custom_partition = payload.get('partitionScheme')

    try:
        seed = int(payload.get('randomSeed', 1))
    except (TypeError, ValueError):
        seed = 1
    np.random.seed(seed)

    try:
        link_eff = float(payload.get('linkEfficiency', 1.0))
    except (TypeError, ValueError):
        link_eff = 1.0
    link_eff = max(0.05, min(1.0, link_eff))

    stage_times_json = None
    bottleneck_stage_index = -1
    pipeline_stats = None

    used_custom = False
    if custom_partition and 'startLayers' in custom_partition and 'endLayers' in custom_partition:
        start_layers = [int(x) for x in custom_partition['startLayers']]
        end_layers = [int(x) for x in custom_partition['endLayers']]
        if len(start_layers) != device_count or len(end_layers) != device_count:
            return jsonify({'error': f'partitionScheme must have {device_count} start/end layers'}), 400
        start_layers = [max(1, int(x)) for x in start_layers]
        end_layers = [max(start_layers[i], int(end_layers[i])) for i in range(device_count)]
        used_custom = True

        path = _resolve_data_path(model_name)
        if not path.exists():
            return jsonify({'error': f'Model {model_name} not found'}), 404
        df = pd.read_csv(path)
        layer_flops = _to_numeric_series(df['Flops']).values.astype(np.float64)
        layer_data_sizes = _to_numeric_series(df['DataSize']).values.astype(np.float64)
        _L = len(layer_flops)
        if len(layer_data_sizes) == _L:
            layer_data_sizes = np.append(layer_data_sizes, layer_data_sizes[-1] if _L > 0 else 0.0)

        entry = _MODEL_ALGO_MAP.get(model_name, _DEFAULT_ALGO)
        hm_mod = __import__(entry[0], fromlist=['HiveMind'])
        ep_mod = __import__(entry[1], fromlist=['Pipeline'])
        setattr(hm_mod, 's', np.array(device_order))
        setattr(ep_mod, 's', np.array(device_order))
        # Use same bandwidth range as inter-device band so first-hop is not a fixed bottleneck;
        # then partition changes (comp + inter-device transfer) actually affect the result.
        first_hop = np.random.randint(max(1, bw_range[0]), max(2, bw_range[1]) + 1, size=device_count)
        setattr(hm_mod, 'first_band', first_hop.copy())
        setattr(ep_mod, 'first_band', first_hop.copy())

        n_pairs = device_count * (device_count - 1) // 2
        bw_vals = np.random.randint(bw_range[0], bw_range[1], size=n_pairs)
        band = np.zeros((device_count, device_count))
        off = 0
        for i in range(device_count - 1):
            for j in range(i + 1, device_count):
                band[i, j] = band[j, i] = bw_vals[off]
                off += 1
        perf = np.linspace(perf_range[0], perf_range[1], num=device_count)

        throughput, inference_time, stage_times, bottleneck_stage_index, pipeline_stats = _evaluate_partition(
            start_layers, end_layers, device_order, band * link_eff, perf, first_hop * link_eff,
            layer_flops, layer_data_sizes
        )
        stage_times_json = [float(x) if np.isfinite(x) else None for x in stage_times]
        partition = {'startLayers': start_layers, 'endLayers': end_layers}
    else:
        EngineClass = _get_engine_classes(model_name, algo)
        if EngineClass is None:
            return jsonify({'error': f'Unknown algorithm: {algo}'}), 400

        engine = EngineClass(device_count)
        engine.assignment()
        _inject_model_layer_data(engine, model_name, algo)
        if bw_range:
            _populate_bandwidth_matrix(engine, device_count, bw_range[0], bw_range[1])
        if perf_range:
            engine.fn = np.linspace(perf_range[0], perf_range[1], num=device_count)
        _inject_device_context(device_order, device_count, model_name, algo, bw_range=bw_range)

        if algo == 'HiveMind':
            throughput, inference_time = engine.enhancedDijkstraTime()
            partition = _extract_partition_hivemind(engine, device_count)
            # Redistribute to reduce relay devices; re-evaluate partition
            L = getattr(engine, 'LAYER_COUNT', getattr(engine, 'L', 13))
            sl, el = _redistribute_partition_to_avoid_relay(
                partition['startLayers'], partition['endLayers'], L, device_count
            )
            if (sl, el) != (partition['startLayers'], partition['endLayers']):
                entry = _MODEL_ALGO_MAP.get(model_name, _DEFAULT_ALGO)
                hm_mod = __import__(entry[0], fromlist=['HiveMind'])
                first_hop = getattr(hm_mod, 'first_band', np.ones(device_count) * 25)
                partition = {'startLayers': sl, 'endLayers': el}
                band_eval = np.asarray(engine.band, dtype=np.float64) * link_eff
                fh = np.asarray(first_hop, dtype=np.float64).reshape(-1)[:device_count] * link_eff
                throughput, inference_time, stage_times, bottleneck_stage_index, pipeline_stats = _evaluate_partition(
                    sl, el, device_order, band_eval, engine.fn, fh,
                    engine.layer_flops, engine.layer_data_sizes
                )
                stage_times_json = [float(x) if np.isfinite(x) else None for x in stage_times]
            else:
                band_eval = np.asarray(engine.band, dtype=np.float64) * link_eff
                first_hop = _first_hop_for_stages(device_count, model_name, bw_range)
                fh = np.asarray(first_hop, dtype=np.float64).reshape(-1)[:device_count] * link_eff
                _, _, stage_times, bottleneck_stage_index, pipeline_stats = _evaluate_partition(
                    partition['startLayers'], partition['endLayers'], device_order,
                    band_eval, engine.fn, fh,
                    engine.layer_flops, engine.layer_data_sizes
                )
                stage_times_json = [float(x) if np.isfinite(x) else None for x in stage_times]
        else:
            throughput, inference_time = engine.dynamic_planning()
            partition = _extract_partition_edgepipe(engine, device_count)
            band_eval = np.asarray(engine.band, dtype=np.float64) * link_eff
            first_hop = _first_hop_for_stages(device_count, model_name, bw_range)
            lf, ld = _engine_layer_arrays(engine)
            fh = np.asarray(first_hop, dtype=np.float64).reshape(-1)[:device_count] * link_eff
            _, _, stage_times, bottleneck_stage_index, pipeline_stats = _evaluate_partition(
                partition['startLayers'], partition['endLayers'], device_order,
                band_eval, engine.fn, fh,
                lf, ld
            )
            stage_times_json = [float(x) if np.isfinite(x) else None for x in stage_times]

        band = engine.band
        perf = engine.fn

    device_metrics = [
        {
            'deviceId': device_order[i],
            'performance': float(perf[device_order[i]]),
            'startLayer': partition['startLayers'][i] if i < len(partition['startLayers']) else 0,
            'endLayer': partition['endLayers'][i] if i < len(partition['endLayers']) else 0
        }
        for i in range(device_count)
    ]

    band_matrix = band.tolist() if hasattr(band, 'tolist') else band
    perf_list = perf.tolist() if hasattr(perf, 'tolist') else perf.tolist()

    bottleneck_device_id = None
    if bottleneck_stage_index >= 0 and bottleneck_stage_index < len(device_order):
        bottleneck_device_id = int(device_order[bottleneck_stage_index])

    return jsonify(
        success=True,
        usedCustomPartition=used_custom,
        randomSeed=seed,
        linkEfficiency=link_eff,
        throughput=float(throughput) if throughput != float('inf') else 0,
        inferenceTime=float(inference_time) if inference_time != float('inf') else None,
        partitionScheme=partition,
        deviceMetrics=device_metrics,
        bandwidthMatrix=band_matrix,
        devicePerformance=perf_list,
        deviceOrder=device_order,
        stageTimes=stage_times_json,
        bottleneckStageIndex=bottleneck_stage_index,
        bottleneckDeviceId=bottleneck_device_id,
        pipelineStats=_serialize_pipeline_stats(pipeline_stats),
    )


@app.route('/api/compare', methods=['POST'])
@_handle_api_errors
def api_compare():
    """Compare multiple algorithms with same device topology for fair comparison."""
    payload = request.json or {}
    model_name = payload.get('model', 'AlexNet')
    device_count = int(payload.get('deviceNum', 5))
    algorithms = payload.get('algorithms', ['HiveMind', 'EdgePipe'])
    bw_range = payload.get('bandwidthRange', [21, 31])
    perf_range = payload.get('performanceRange', [41, 60])

    try:
        seed = int(payload.get('randomSeed', 1))
    except (TypeError, ValueError):
        seed = 1
    np.random.seed(seed)

    try:
        link_eff = float(payload.get('linkEfficiency', 1.0))
    except (TypeError, ValueError):
        link_eff = 1.0
    link_eff = max(0.05, min(1.0, link_eff))
        
        results = {}
    device_order = payload.get('deviceOrder', list(range(device_count)))
        
        for algo in algorithms:
        EngineClass = _get_engine_classes(model_name, algo)
        if EngineClass is None:
                continue
        engine = EngineClass(device_count)
            engine.assignment()
        _inject_model_layer_data(engine, model_name, algo)
        _populate_bandwidth_matrix(engine, device_count, bw_range[0], bw_range[1])
        engine.fn = np.linspace(perf_range[0], perf_range[1], num=device_count)
        _inject_device_context(device_order, device_count, model_name, algo, bw_range=bw_range)
            if algo == 'HiveMind':
            tp, ti = engine.enhancedDijkstraTime()
            else:
            tp, ti = engine.dynamic_planning()
        pstats = _pipeline_stats_from_engine(
            engine, model_name, algo, device_count, device_order, bw_range, link_eff
        )
            results[algo] = {
            'throughput': float(tp) if tp != float('inf') else 0,
            'inferenceTime': float(ti) if ti != float('inf') else None,
            'pipelineStats': _serialize_pipeline_stats(pstats),
            }
    return jsonify(success=True, results=results, randomSeed=seed, linkEfficiency=link_eff)
        

@app.route('/api/device-topology', methods=['POST'])
@_handle_api_errors
def api_device_topology():
    """Generate device network topology."""
    payload = request.json or {}
    device_count = int(payload.get('deviceNum', 5))
    bw_range = payload.get('bandwidthRange', [21, 31])
    perf_range = payload.get('performanceRange', [41, 60])

    try:
        seed = int(payload.get('randomSeed', 1))
    except (TypeError, ValueError):
        seed = 1
    np.random.seed(seed)
    n_pairs = device_count * (device_count - 1) // 2
    bw_vals = np.random.randint(bw_range[0], bw_range[1], size=n_pairs)
    perf = np.linspace(perf_range[0], perf_range[1], num=device_count)

    band = np.zeros((device_count, device_count))
    idx = 0
    for i in range(device_count - 1):
        for j in range(i + 1, device_count):
            band[i, j] = band[j, i] = bw_vals[idx]
            idx += 1

    nodes = [
        {
                'id': i,
                'label': f'Device {i}',
            'performance': float(perf[i]),
            'x': np.cos(2 * np.pi * i / device_count) * 200,
            'y': np.sin(2 * np.pi * i / device_count) * 200
        }
        for i in range(device_count)
    ]
    edges = [
        {'from': i, 'to': j, 'bandwidth': float(band[i, j]), 'label': f'{band[i, j]:.1f} MB/s'}
        for i in range(device_count)
        for j in range(i + 1, device_count)
        if band[i, j] > 0
    ]

    return jsonify(
        success=True,
        nodes=nodes,
        edges=edges,
        bandwidthMatrix=band.tolist(),
        performance=perf.tolist(),
        randomSeed=seed,
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
