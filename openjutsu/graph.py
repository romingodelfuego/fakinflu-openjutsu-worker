"""Graphes ComfyUI d'Openjutsu et decoupage en fenetres, sans FastAPI ni comfy-sdk.

Repris de SydFloyd/openjutsu app/main.py @ f783a001 (Apache-2.0) : `workflow()` et la boucle de
`generate()`, avec les modifications de docs/openjutsu/annexes/patch_openjutsu.py (fakinflu) :
  - taille reelle (480x832 ou 832x480 en brouillon, 720x1280 ou 1280x720 en final) ecrite dans WanAnimateToVideo, WanVaceToVideo et EmptyImage ;
  - mode `replace` (Wan 2.2 Animate en remplacement : pose + decor d'origine + masque SAM 2), fenetre
    comme recast (77 images, 5 reportees par continue_motion et video_frame_offset) ;
  - `mask_expand` pour swap ; OJ_UNET_DTYPE (ex. fp8_e4m3fn) pour les GPU 48 Go ;
  - options Wan Animate (recast, replace), alignees sur le template officiel ComfyUI
    video_wan2_2_14B_animate : LoRA relight (replace) et lightx2v (distill), face_video (DWPose visage).
    En replace, le masque arrive deja elargi et decoupe en blocs (video.replace_mask) ; le decor est
    perce en NOIR, comme frame * (1 - mask) du pretraitement de Wan 2.2.
Fonctions pures (aucun fichier ecrit) : testables sur CPU.
"""
import json
import os
from pathlib import Path

from openjutsu.video import windows

ROOT = Path(__file__).resolve().parent
LONG = ("recast", "replace")  # fenetres longues, continuation par continue_motion
OUTPUT_NODE = "44"
SIZES = ((480, 832), (832, 480), (720, 1280), (1280, 720))
ANIMATE = ("recast", "replace")
RELIGHT_LORA = "wan2.2_animate_14B_relight_lora_bf16.safetensors"
DISTILL_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
# Unet Wan2.2-Animate-14B (v1) par preset ou par job (params.unet) ; int8 = celui des presets d'origine.
# fp8 : Kijai v2 (couches visage aussi en fp8 scaled ; la v1 bruitait en grille dans ComfyUI natif).
UNETS = {"int8": "wan2.2_animate_14B_int8_convrot.safetensors",
         "fp8": "Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors"}
# Reglables par job (params), defauts du preset ; bornes verifiees par check_options.
OPTIONS = {"relight": bool, "face": bool, "distill": bool, "mask_expand": int, "mask_block": int, "unet": str}


def options(preset, params=None):
    """Options Wan Animate resolues : params du job, sinon preset, sinon desactivees (comportement d'origine)."""
    if preset["mode"] not in ANIMATE:
        return {}
    p, params = preset["params"], params or {}
    out = {k: params[k] if params.get(k) is not None else p.get(k, t()) for k, t in OPTIONS.items()}
    out["unet"] = out["unet"] or "int8"
    if preset["mode"] != "replace":
        out.update(relight=False, mask_expand=0, mask_block=0)
    return out


def check_options(params):
    for k, t in OPTIONS.items():
        v = params.get(k)
        if v is None:
            continue
        if t is str and v not in UNETS:
            raise ValueError(f"params.{k} must be one of {', '.join(UNETS)}")
        if t is bool and not isinstance(v, bool):
            raise ValueError(f"params.{k} must be a boolean")
        if t is int and (isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 64):
            raise ValueError(f"params.{k} must be an integer in [0, 64]")


def presets():
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "presets").glob("*.json"))}


def capacity(mode):
    return 77 if mode in LONG else 81


def plan(mode, total, chunk_frames=None):
    """Fenetres couvrant `total` images (ValueError si impossible)."""
    return list(windows(total, chunk_frames or capacity(mode), 5 if mode in LONG else 9))


def workflow(preset, prompt, seed, length, steps=None, cfg=None, shift=None, size=(832, 480), opts=None):
    graph = json.loads((ROOT / "workflows" / f"{preset['workflow']}.json").read_text())
    for node in graph.values():
        if node["class_type"] in ("WanAnimateToVideo", "WanVaceToVideo", "EmptyImage"):
            node["inputs"]["width"], node["inputs"]["height"] = size
    if os.environ.get("OJ_UNET_DTYPE"):
        graph["1"]["inputs"]["weight_dtype"] = os.environ["OJ_UNET_DTYPE"]
    params = preset["params"]
    graph["5"]["inputs"]["text"] = (prompt + ", " if prompt else "") + preset["prompt"]
    graph["40"]["inputs"].update(seed=seed, steps=steps or params["steps"], cfg=params["cfg"] if cfg is None else cfg)
    if params.get("sampler"):
        graph["40"]["inputs"]["sampler_name"] = params["sampler"]
    graph["4"]["inputs"]["shift"] = params.get("shift", 8.0) if shift is None else shift
    graph["30"]["inputs"]["length"] = length
    if "strength" in graph["30"]["inputs"]:
        graph["30"]["inputs"]["strength"] = params["strength"]
    if preset["mode"] == "swap":
        graph["18"]["inputs"]["expand"] = params.get("mask_expand", 8)
    opts = options(preset) if opts is None else opts
    if opts.get("unet"):
        graph["1"]["inputs"]["unet_name"] = UNETS[opts["unet"]]
    # LoRA entre l'unet (1) et ModelSamplingSD3 (4), dans l'ordre du template : relight puis lightx2v
    model = ["1", 0]
    for key, node, name in (("relight", "60", RELIGHT_LORA), ("distill", "61", DISTILL_LORA)):
        if opts.get(key):
            graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": 1.0}}
            model = [node, 0]
    graph["4"]["inputs"]["model"] = model
    # face_video : reperes du visage (DWPose visage seul) sur l'image entiere ramenee a 512x512, sans
    # recadrage (le template coupe au centre : en vertical, la tete sortirait du cadre)
    if opts.get("face"):
        graph["70"] = {"class_type": "ImageScale", "inputs": {"image": ["11", 0], "upscale_method": "lanczos", "width": 512, "height": 512, "crop": "disabled"}}
        graph["71"] = {"class_type": "DWPreprocessor", "inputs": {**graph["20"]["inputs"], "image": ["70", 0],
                                                                    "detect_hand": "disable", "detect_body": "disable", "detect_face": "enable"}}
        graph["30"]["inputs"]["face_video"] = ["71", 0]
    return graph


def window_graph(preset, prompt, seed, window, index, size, names, prefix, steps=None, cfg=None, shift=None, opts=None):
    """Graphe d'une fenetre. `names` = noms des fichiers deja copies dans input/ de ComfyUI :
    video (clip entier pour recast/replace, tranche de la fenetre pour swap/restyle), ref (option),
    mask (replace : clip entier ; swap : tranche), continue (recast/replace a partir de la 2e fenetre :
    5 dernieres images de la fenetre precedente)."""
    mode = preset["mode"]
    graph = workflow(preset, prompt, seed, window["length"], steps, cfg, shift, size, opts)
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
