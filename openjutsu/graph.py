"""Graphes ComfyUI d'Openjutsu et decoupage en fenetres, sans FastAPI ni comfy-sdk.

Repris de SydFloyd/openjutsu app/main.py @ f783a001 (Apache-2.0) : `workflow()` et la boucle de
`generate()`, avec les modifications de docs/openjutsu/annexes/patch_openjutsu.py (fakinflu) :
  - taille reelle (480x832 ou 832x480) ecrite dans WanAnimateToVideo, WanVaceToVideo et EmptyImage ;
  - mode `replace` (Wan 2.2 Animate en remplacement : pose + decor d'origine + masque SAM 2), fenetre
    comme recast (77 images, 5 reportees par continue_motion et video_frame_offset) ;
  - `mask_expand` pour swap et replace ; OJ_UNET_DTYPE (ex. fp8_e4m3fn) pour les GPU 48 Go.
Fonctions pures (aucun fichier ecrit) : testables sur CPU.
"""
import json
import os
from pathlib import Path

from openjutsu.video import windows

ROOT = Path(__file__).resolve().parent
LONG = ("recast", "replace")  # fenetres longues, continuation par continue_motion
OUTPUT_NODE = "44"
SIZES = ((480, 832), (832, 480))


def presets():
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "presets").glob("*.json"))}


def capacity(mode):
    return 77 if mode in LONG else 81


def plan(mode, total, chunk_frames=None):
    """Fenetres couvrant `total` images (ValueError si impossible)."""
    return list(windows(total, chunk_frames or capacity(mode), 5 if mode in LONG else 9))


def workflow(preset, prompt, seed, length, steps=None, cfg=None, shift=None, size=(832, 480)):
    graph = json.loads((ROOT / "workflows" / f"{preset['workflow']}.json").read_text())
    for node in graph.values():
        if node["class_type"] in ("WanAnimateToVideo", "WanVaceToVideo", "EmptyImage"):
            node["inputs"]["width"], node["inputs"]["height"] = size
    if os.environ.get("OJ_UNET_DTYPE"):
        graph["1"]["inputs"]["weight_dtype"] = os.environ["OJ_UNET_DTYPE"]
    params = preset["params"]
    graph["5"]["inputs"]["text"] = (prompt + ", " if prompt else "") + preset["prompt"]
    graph["40"]["inputs"].update(seed=seed, steps=steps or params["steps"], cfg=params["cfg"] if cfg is None else cfg)
    graph["4"]["inputs"]["shift"] = params.get("shift", 8.0) if shift is None else shift
    graph["30"]["inputs"]["length"] = length
    if "strength" in graph["30"]["inputs"]:
        graph["30"]["inputs"]["strength"] = params["strength"]
    if preset["mode"] in ("swap", "replace"):
        graph["18"]["inputs"]["expand"] = params.get("mask_expand", 8)
    return graph


def window_graph(preset, prompt, seed, window, index, size, names, prefix, steps=None, cfg=None, shift=None):
    """Graphe d'une fenetre. `names` = noms des fichiers deja copies dans input/ de ComfyUI :
    video (clip entier pour recast/replace, tranche de la fenetre pour swap/restyle), ref (option),
    mask (replace : clip entier ; swap : tranche), continue (recast/replace a partir de la 2e fenetre :
    5 dernieres images de la fenetre precedente)."""
    mode = preset["mode"]
    graph = workflow(preset, prompt, seed, window["length"], steps, cfg, shift, size)
    graph["10"]["inputs"]["file"] = names["video"]
    if mode in LONG:
        if mode == "replace":
            graph["15"]["inputs"]["file"] = names["mask"]
        graph["30"]["inputs"]["video_frame_offset"] = window["start"] + window["overlap"]
        if names.get("continue"):
            graph["50"] = {"class_type": "LoadVideo", "inputs": {"file": names["continue"]}}
            graph["51"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}
            graph["30"]["inputs"]["continue_motion"] = ["51", 0]
    elif mode == "swap":
        graph["15"]["inputs"]["file"] = names["mask"]
    if names.get("ref"):
        graph["12"]["inputs"]["image"] = names["ref"]
    else:
        for node in ("12", "13", "14"):
            graph.pop(node, None)
        for key in ("reference_image", "clip_vision_output"):
            graph["30"]["inputs"].pop(key, None)
    graph[OUTPUT_NODE]["inputs"]["filename_prefix"] = f"{prefix}/chunk-{index}"
    return graph


def assemble_mode(mode):
    """Assemblage : replace se raccorde comme recast (images reportees, pas de fondu)."""
    return "recast" if mode == "replace" else mode
