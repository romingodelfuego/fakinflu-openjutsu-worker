#!/usr/bin/env python3
"""fakinflu — worker openjutsu : Openjutsu (SydFloyd/openjutsu @ f783a001) en serverless.

Remplace le /handler.py de runpod/worker-comfyui ; /start.sh lance ComfyUI (127.0.0.1:8188).
Handler GENERATEUR : progression et apercus via /stream (return_aggregate_stream = False).
Squelette (ComfyUI, websocket, OOM, sonde) repris du worker wananimate.

Entree (contrat transfer.py, contracts/openjutsu.input.schema.json) :
  {"run_id", "r2": bool,
   "inputs":  {"video": clip DEJA normalise par le client (16 i/s, 480x832 ou 832x480),
               "ref": image de reference (si le preset l'exige), "mask": mask.mkv (swap, replace)},
   "outputs": {"mask", "mask_preview", "video", "preview", "probe"},
   "outputs_json": {put_url, key} | null,
   "params":  {"action": "probe|mask|generate", "points": [[x, y, label]], "preset": str,
               "prompt": str, "seed": int, "max_frames": int|null, "steps": int|null,
               "cfg": number|null, "shift": number|null}}

Actions :
  probe    : VRAM, RAM du conteneur, disque, version (aucun modele charge).
  mask     : SAM 2.1 suit les clics sur tout le clip -> mask.mkv (FFV1 sans perte) + mask_preview.png.
  generate : graphe Openjutsu par fenetre sur le ComfyUI local, assemblage -> result.mp4 (sans audio).
"""
import glob, json, os, queue, shutil, subprocess, tempfile, threading, time, traceback, urllib.error, urllib.request, uuid

import runpod

from transfer import Outputs, fetch_inputs, preview_event, version_event
from openjutsu import graph, video

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
JOB_TIMEOUT = int(os.environ.get("OPENJUTSU_JOB_TIMEOUT_S", "3500"))
MODES = [m.strip() for m in os.environ.get("OJ_MODES", "recast,replace").split(",") if m.strip()]
ACTIONS = ("probe", "mask", "generate")


def _http(method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def wait_comfy(max_s=600):
    t = time.time()
    while time.time() - t < max_s:
        try:
            _http("GET", "/system_stats", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError(f"ComfyUI unreachable at {COMFY} after {max_s}s")


def _ws_progress(client_id, q, stop):
    try:
        import websocket
    except ImportError:
        return
    url = COMFY.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
    try:
        ws = websocket.create_connection(url, timeout=5)
    except Exception:
        return
    ws.settimeout(1)
    while not stop.is_set():
        try:
            msg = ws.recv()
        except Exception:
            continue
        if not isinstance(msg, str):
            continue
        try:
            m = json.loads(msg)
        except ValueError:
            continue
        if m.get("type") in ("progress", "executing"):
            q.put(m)
    ws.close()


def run_workflow(wf):
    """Generateur : soumet le graphe, emet la progression, renvoie (secondes) via StopIteration."""
    client_id = uuid.uuid4().hex
    q, stop = queue.Queue(), threading.Event()
    th = threading.Thread(target=_ws_progress, args=(client_id, q, stop), daemon=True)
    th.start()
    try:
        try:
            r = _http("POST", "/prompt", {"prompt": wf, "client_id": client_id})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the graph (HTTP {e.code}): {e.read().decode()[:1500]}")
        pid = r["prompt_id"]
        t0, last_poll = time.time(), 0.0
        names = {k: v.get("_meta", {}).get("title") or v["class_type"] for k, v in wf.items()}
        while True:
            if time.time() - t0 > JOB_TIMEOUT:
                _http("POST", "/interrupt", {})
                raise RuntimeError(f"ComfyUI timeout ({JOB_TIMEOUT}s)")
            try:
                m = q.get(timeout=1)
                d = m.get("data", {})
                if m["type"] == "executing" and d.get("node"):
                    yield {"step": names.get(d["node"], d["node"])}
                elif m["type"] == "progress" and d.get("max"):
                    yield {"step": names.get(d.get("node"), "compute"), "pct": round(100 * d["value"] / d["max"])}
            except queue.Empty:
                pass
            if time.time() - last_poll < 2:
                continue
            last_poll = time.time()
            try:
                h = _http("GET", f"/history/{pid}")
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError as e:
                raise RuntimeError(f"ComfyUI stopped mid-compute ({e.reason}): {oom_report()}") from e
            if pid not in h:
                continue
            st = h[pid].get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[-1][1] if msgs else st
                if isinstance(detail, dict):
                    detail = {k: detail.get(k) for k in ("node_id", "node_type", "exception_message")}
                raise RuntimeError(f"ComfyUI execution error: {json.dumps(detail, ensure_ascii=False)[:800]}")
            if st.get("completed"):
                return time.time() - t0
    finally:
        stop.set()


def memory_limit_bytes():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                v = f.read().strip()
        except OSError:
            continue
        if v.isdigit() and int(v) < 1 << 60:
            return int(v)
    return None


def _cgroup(name):
    for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        try:
            with open(os.path.join(base, name)) as f:
                return f.read()
        except OSError:
            continue
    return None


def oom_report():
    events = _cgroup("memory.events") or ""
    kills = next((int(l.split()[1]) for l in events.splitlines() if l.startswith("oom_kill ")), None)
    peak = _cgroup("memory.peak") or _cgroup("memory.max_usage_in_bytes")
    limit = memory_limit_bytes()
    gb = lambda b: f"{int(b) / 1024 ** 3:.1f} GB"
    mem = f"peak {gb(peak.strip())}" if peak and peak.strip().isdigit() else "peak unknown"
    mem += f", limit {gb(limit)}" if limit else ", limit unknown"
    if kills:
        return (f"killed by the kernel out of memory (oom_kill={kills}, {mem}). Check COMFY_EXTRA_ARGS "
                f"(current: {os.environ.get('COMFY_EXTRA_ARGS', '')!r})")
    return f"no oom_kill in the cgroup ({mem}): likely native crash (worker RunPod logs)"


def _gb(b):
    return round(int(b) / 1024 ** 3, 2) if b not in (None, "") and str(b).strip().isdigit() else None


def probe_report():
    stats = {}
    try:
        stats = _http("GET", "/system_stats", timeout=10)
    except Exception as e:
        stats = {"error": f"{type(e).__name__}: {e}"}
    disk = {}
    for path in ("/", COMFY_DIR):
        try:
            st = os.statvfs(path)
            disk[path] = {"free_gb": round(st.f_bavail * st.f_frsize / 1024 ** 3, 2), "total_gb": round(st.f_blocks * st.f_frsize / 1024 ** 3, 2)}
        except OSError:
            pass
    system = stats.get("system") or {}
    models = {os.path.relpath(p, COMFY_DIR): _gb(os.path.getsize(p))
              for p in sorted(glob.glob(os.path.join(COMFY_DIR, "models", "*", "*.safetensors")))}
    sam2 = os.environ.get("SAM2_CKPT", "")
    return {
        "modes": MODES,
        "cgroup": {"limit_gb": _gb(memory_limit_bytes()), "peak_gb": _gb((_cgroup("memory.peak") or "").strip())},
        "comfyui": {"version": system.get("comfyui_version"), "pytorch": system.get("pytorch_version"),
                    "argv": system.get("argv"), "ram_total_gb": _gb(system.get("ram_total")),
                    "devices": [{k: d.get(k) for k in ("name", "type", "vram_total", "vram_free")} for d in stats.get("devices") or []],
                    "error": stats.get("error")},
        "models": models,
        "sam2": {"ckpt": sam2, "gb": _gb(os.path.getsize(sam2)) if sam2 and os.path.exists(sam2) else None},
        "disk": disk,
        "env": {"COMFY_EXTRA_ARGS": os.environ.get("COMFY_EXTRA_ARGS", ""), "OJ_UNET_DTYPE": os.environ.get("OJ_UNET_DTYPE", "")},
    }


def check_video(path):
    """Le client normalise (16 i/s, 480x832 ou 832x480) : on refuse tout le reste avant le GPU."""
    info = video.probe(path)
    if (info["width"], info["height"]) not in graph.SIZES or abs(info["fps"] - video.FPS) > 0.05:
        raise ValueError(f"video not normalized: {info['width']}x{info['height']} at {info['fps']:.2f} fps "
                         f"(expected 480x832 or 832x480 at {video.FPS} fps, see make oj-prep)")
    if not info["frames"]:
        raise ValueError("video has no frames")
    return info


def check_points(points, info):
    if not isinstance(points, list) or not points:
        raise ValueError("params.points required: [[x, y, 1], ...] in frame0.jpg pixels")
    for p in points:
        if not isinstance(p, list) or len(p) not in (2, 3) or not all(isinstance(v, (int, float)) for v in p):
            raise ValueError(f"invalid point {p!r}: expected [x, y] or [x, y, label]")
        if not (0 <= p[0] < info["width"] and 0 <= p[1] < info["height"]):
            raise ValueError(f"point {p!r} outside the {info['width']}x{info['height']} frame")
        if len(p) == 3 and p[2] not in (0, 1):
            raise ValueError(f"point {p!r}: label must be 1 (subject) or 0 (exclude)")


def check_generate(params, roles):
    """Preset, mode pris en charge par l'image, entrees requises : verifies AVANT tout telechargement."""
    presets = graph.presets()
    name = params.get("preset")
    preset = presets.get(name)
    if preset is None:
        raise ValueError(f"unknown preset {name!r} (available: {', '.join(presets)})")
    if preset["mode"] not in MODES:
        raise ValueError(f"preset {name} uses mode {preset['mode']}, not served by this image (OJ_MODES={','.join(MODES)})")
    needed = ["video"]
    if preset.get("requires_reference_image"):
        needed.append("ref")
    if preset["mode"] in ("swap", "replace"):
        needed.append("mask")
    missing = [r for r in needed if r not in roles]
    if missing:
        raise ValueError(f"preset {name} ({preset['mode']}) needs inputs {missing}"
                         + (" (track the subject first: action mask)" if "mask" in missing else ""))
    for key in ("steps", "max_frames"):
        if params.get(key) is not None and (not isinstance(params[key], int) or params[key] < 1):
            raise ValueError(f"params.{key} must be a positive integer")
    return preset


def thumb(path, dst, at=0.5):
    """Image JPEG au milieu d'une video (apercu)."""
    info = video.probe(path)
    t = max(0.0, at * info["frames"] / video.FPS)
    video.ffmpeg("-ss", f"{t:.2f}", "-i", path, "-frames:v", 1, "-q:v", 5, dst)
    with open(dst, "rb") as f:
        return f.read()


class Comfy:
    """Fichiers deposes dans input/ de ComfyUI sous des noms uniques au job ; nettoyes a la fin."""

    def __init__(self, tag):
        self.tag, self.copied = tag, []

    def put(self, path, name):
        dst = os.path.join(COMFY_DIR, "input", f"oj_{self.tag}_{name}")
        shutil.copy(path, dst)
        self.copied.append(dst)
        return os.path.basename(dst)

    def cleanup(self):
        for p in self.copied:
            if os.path.exists(p):
                os.remove(p)


def do_mask(params, paths, outputs, work):
    from openjutsu import sam2_mask
    info = check_video(paths["video"])
    points = params.get("points")
    check_points(points, info)
    yield {"type": "progress", "step": f"SAM 2.1: {len(points)} point(s), {info['frames']} frames", "pct": 5}
    t = time.time()
    mask, preview = os.path.join(work, "mask.mkv"), os.path.join(work, "mask_preview.png")
    try:
        result = sam2_mask.propagate_mask(paths["video"], points, mask, preview)
    finally:
        sam2_mask.release()
    seconds = time.time() - t
    with open(preview, "rb") as f:
        yield preview_event(f.read(), "image/png")
    yield from outputs.emit("mask", mask, "mask.mkv")
    yield from outputs.emit("mask_preview", preview, "mask_preview.png")
    outputs.write_summary()
    yield {"type": "done", "action": "mask", "frames": result["frames"], "timings": {"sam2_s": round(seconds, 1)}}


def do_generate(params, preset, paths, outputs, work, prefix, comfy):
    info = check_video(paths["video"])
    mode = preset["mode"]
    size = (info["width"], info["height"])
    total = min(info["frames"], params["max_frames"]) if params.get("max_frames") else info["frames"]
    windows = graph.plan(mode, total)
    if mode in ("swap", "replace"):
        m = video.probe(paths["mask"])
        if (m["width"], m["height"]) != size or m["frames"] < total:
            raise ValueError(f"mask {m['width']}x{m['height']} {m['frames']} frames does not match the video "
                             f"{size[0]}x{size[1]} ({total} frames used): recompute the mask")
    seed = params.get("seed") if params.get("seed") is not None else 0
    prompt = (params.get("prompt") or "").strip()
    yield {"type": "progress", "pct": 2,
           "step": f"{preset['id']} ({mode}) {size[0]}x{size[1]}, {total} frames, {len(windows)} window(s)"}

    names = {}
    if paths.get("ref"):
        names["ref"] = comfy.put(paths["ref"], "ref" + os.path.splitext(paths["ref"])[1].lower())
    if mode in graph.LONG:
        names["video"] = comfy.put(paths["video"], "input.mp4")
        if mode == "replace":
            names["mask"] = comfy.put(paths["mask"], "mask.mkv")
    wait_comfy()

    chunks, timings = [], []
    for i, window in enumerate(windows):
        if mode in graph.LONG:
            if chunks:
                tail = os.path.join(work, f"continue-{i}.mkv")
                video.write_frames(video.read_frames(chunks[-1])[-5:], tail, lossless=True)
                names["continue"] = comfy.put(tail, f"continue-{i}.mkv")
        else:
            src = os.path.join(work, f"input-{i}.mp4")
            video.slice_video(paths["video"], src, window["start"], window["count"], window["length"])
            names["video"] = comfy.put(src, f"input-{i}.mp4")
            if mode == "swap":
                msk = os.path.join(work, f"mask-{i}.mkv")
                video.slice_video(paths["mask"], msk, window["start"], window["count"], window["length"], True)
                names["mask"] = comfy.put(msk, f"mask-{i}.mkv")
        wf = graph.window_graph(preset, prompt, seed, window, i, size, names, prefix,
                                params.get("steps"), params.get("cfg"), params.get("shift"))
        label = f"window {i + 1}/{len(windows)}"
        gen = run_workflow(wf)
        while True:
            try:
                ev = next(gen)
            except StopIteration as stop:
                seconds = stop.value
                break
            local = ev.get("pct", 0) / 100
            yield {"type": "progress", "step": f"{label}: {ev['step']}", "window": i,
                   "pct": round(5 + 90 * (i + local) / len(windows))}
        hits = sorted(glob.glob(os.path.join(COMFY_DIR, "output", prefix, f"chunk-{i}_*.mp4")))
        if not hits:
            raise RuntimeError(f"{label}: ComfyUI completed without a video output (node {graph.OUTPUT_NODE})")
        chunk = os.path.join(work, f"chunk-{i}.mp4")
        shutil.move(hits[-1], chunk)
        chunks.append(chunk)
        timings.append(round(seconds, 1))
        print(f"[openjutsu] {label}: {window['length']} frames in {seconds:.1f}s", flush=True)
        yield preview_event(thumb(chunk, chunk + ".jpg"), "image/jpeg", window=i)

    result = os.path.join(work, "result.mp4")
    video.assemble(chunks, windows, result, total, graph.assemble_mode(mode))
    preview = os.path.join(work, "preview.jpg")
    thumb(result, preview)
    yield from outputs.emit("video", result, "result.mp4")
    yield from outputs.emit("preview", preview, "preview.jpg")
    outputs.write_summary()
    comfy_s = sum(timings)
    yield {"type": "done", "action": "generate", "preset": preset["id"], "mode": mode, "seed": seed,
           "frames": total, "windows": len(windows),
           "timings": {"comfy_s": round(comfy_s, 1), "windows_s": timings, "s_per_frame": round(comfy_s / total, 2)}}


def handler(job):
    inp = job.get("input") or {}
    params = inp.get("params") or {}
    action = params.get("action") or "generate"
    work = tempfile.mkdtemp(prefix="openjutsu_")
    prefix = f"fakinflu/{inp.get('run_id') or job.get('id', 'run')}"
    comfy = Comfy(uuid.uuid4().hex[:8])
    t0 = time.time()
    yield version_event("openjutsu")
    try:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r} ({'|'.join(ACTIONS)})")
        outputs = Outputs(inp)
        if action == "probe":
            wait_comfy(120)
            report = probe_report()
            print(f"[openjutsu] probe: {json.dumps(report)}", flush=True)
            path = os.path.join(work, "probe.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            yield from outputs.emit("probe", path, "probe.json")
            yield {"type": "done", "action": "probe", "timings": {"total_s": round(time.time() - t0, 1)}}
            return
        roles = set(inp.get("inputs") or {})
        if "video" not in roles:
            raise ValueError("inputs.video required (clip normalized by make oj-prep)")
        preset = check_generate(params, roles) if action == "generate" else None
        paths, events = fetch_inputs(inp, os.path.join(work, "in"), outputs)
        yield from events
        if action == "mask":
            yield from do_mask(params, paths, outputs, work)
        else:
            yield from do_generate(params, preset, paths, outputs, work, prefix, comfy)
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(os.path.join(COMFY_DIR, "output", prefix), ignore_errors=True)
        comfy.cleanup()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
