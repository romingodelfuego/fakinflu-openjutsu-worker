#!/usr/bin/env python3
"""Validation 0 credit : un ComfyUI (local, poids en fichiers vides) accepte-t-il les graphes du worker ?

ComfyUI valide le graphe entier (noeuds, types, liens, fichiers) avant toute execution ; le job accepte
est aussitot retire de la file et interrompu.

Usage : python tests/validate_graphs.py <url ComfyUI> <dossier input/ de ce ComfyUI>
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openjutsu import graph  # noqa: E402

import imageio_ffmpeg  # noqa: E402

W, H, FRAMES = 480, 832, 33


def ffmpeg(*args):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", *map(str, args)], check=True)


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60)


def main(url, input_dir):
    d = Path(input_dir)
    names = {"video": "oj_validate_input.mp4", "mask": "oj_validate_mask.mkv", "ref": "oj_validate_ref.png",
             "continue": "oj_validate_continue.mkv"}
    ffmpeg("-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=16", "-frames:v", FRAMES * 3, "-pix_fmt", "yuv420p", d / names["video"])
    ffmpeg("-f", "lavfi", "-i", f"color=c=white:size={W}x{H}:rate=16", "-frames:v", FRAMES * 3, "-c:v", "ffv1", "-pix_fmt", "gray", d / names["mask"])
    ffmpeg("-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=16", "-frames:v", 5, "-c:v", "ffv1", "-pix_fmt", "bgr0", d / names["continue"])
    ffmpeg("-f", "lavfi", "-i", f"color=c=gray:size={W}x{H}", "-frames:v", 1, d / names["ref"])
    presets = graph.presets()
    failed = 0
    try:
        for preset_id in ("anime-recast", "photoreal-replace"):
            preset = presets[preset_id]
            # fenetre 1 (33 images) puis fenetre 2 d'un clip plus long (continue_motion)
            plan = graph.plan(preset["mode"], FRAMES * 3, chunk_frames=FRAMES)
            for i, window in enumerate(plan[:2]):
                n = {k: v for k, v in names.items() if k != "continue" or i}
                g = graph.window_graph(preset, "validation", 42, window, i, (W, H), n, "fakinflu/validate")
                label = f"{preset_id} window {i} ({len(g)} nodes, {window['length']} frames, offset {g['30']['inputs']['video_frame_offset']})"
                try:
                    r = json.loads(post(url, "/prompt", {"prompt": g}).read())
                    print(f"OK   {label}: prompt_id={r.get('prompt_id')}")
                except urllib.error.HTTPError as e:
                    failed += 1
                    print(f"FAIL {label}:\n{e.read().decode()[:4000]}")
                finally:
                    post(url, "/queue", {"clear": True}).close()
                    post(url, "/interrupt", {}).close()
    finally:
        for name in names.values():
            (d / name).unlink(missing_ok=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].rstrip("/"), sys.argv[2]))
