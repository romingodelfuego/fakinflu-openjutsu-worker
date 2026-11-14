"""Mode « entrees fournies » (Maquette) : pose_video, face_video et mask viennent du client au lieu d'etre detectes.

Non-regression : sans ces entrees, le graphe de chaque fenetre est STRICTEMENT celui d'avant le mode fourni
(empreintes figees dans fixtures/default_graphs.json, generees par `python tests/test_provided_inputs.py --freeze`
avec le graph.py de v1.4.0, avant toute modification). Toute evolution voulue du graphe par defaut = regenerer
l'empreinte, en le disant dans la PR.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openjutsu import graph  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "default_graphs.json"
PORTRAIT = (480, 832)


def default_matrix():
    """(nom, graphe) : chaque preset x options (preset, toutes coupees, toutes au maximum) x taille x fenetres 1 et 2,
    plus chaque sampler / scheduler. Aucune entree fournie : c'est le chemin /replace d'aujourd'hui."""
    presets = graph.presets()
    variants = {"defauts": {}, "sans-options": {"relight": False, "face": False, "distill": False, "mask_expand": 0, "mask_block": 0, "unet": "int8"},
                "tout": {"relight": True, "face": True, "distill": True, "mask_expand": 64, "mask_block": 8, "unet": "fp8"}}
    for pid in sorted(presets):
        preset = presets[pid]
        mode = preset["mode"]
        windows = graph.plan(mode, 120)[:2]
        for (vname, params), size, (i, window) in itertools.product(variants.items(), graph.SIZES, enumerate(windows)):
            names = {"video": "input.mp4", "ref": "ref.png", **({"mask": "mask.mkv"} if mode in ("replace", "swap") else {}),
                     **({"continue": f"continue-{i}.mkv"} if i and mode in graph.LONG else {})}
            g = graph.window_graph(preset, "prompt", 42, window, i, size, names, "fakinflu/ci", opts=graph.options(preset, params))
            yield f"{pid}/{vname}/{size[0]}x{size[1]}/fenetre{i + 1}", g
        g = graph.window_graph(preset, "", 0, windows[0], 0, PORTRAIT, {"video": "v.mp4"} if mode == "recast" else
                               {"video": "v.mp4", "mask": "m.mkv"}, "p")
        yield f"{pid}/sans-ref", g
    preset, window = presets["photoreal-replace"], graph.plan("replace", 33)[0]
    for sampler, scheduler in itertools.zip_longest(graph.SAMPLERS, graph.SCHEDULERS, fillvalue="simple"):
        p = graph.styled(preset, {"style": "2D anime character", "negative": "photorealistic", "sampler": sampler, "scheduler": scheduler})
        yield f"style/{sampler}-{scheduler}", graph.window_graph(p, "prompt", 42, window, 0, PORTRAIT,
                                                                 {"video": "input.mp4", "ref": "ref.png", "mask": "mask.mkv"}, "fakinflu/ci")


def digest(g):
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()


def links(g):
    for node in g.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                yield value[0]


# --- non-regression -------------------------------------------------------------------------------------

def test_default_graphs_unchanged(monkeypatch):
    monkeypatch.delenv("OJ_UNET_DTYPE", raising=False)  # empreintes figees sans dtype impose
    frozen = json.loads(FIXTURE.read_text())
    now = {name: digest(g) for name, g in default_matrix()}
    assert set(now) == set(frozen), f"cas ajoutes/retires : {sorted(set(now) ^ set(frozen))}"
    changed = sorted(k for k in now if now[k] != frozen[k])
    assert not changed, f"graphe par defaut modifie : {changed[:10]} ({len(changed)} cas)"


def test_empty_provided_is_default():
    """Noms d'entrees fournies absents, None ou vides : meme graphe qu'avant."""
    preset = graph.presets()["photoreal-replace"]
    w = graph.plan("replace", 33)[0]
    base = {"video": "v.mp4", "ref": "r.png", "mask": "m.mkv"}
    ref = graph.window_graph(preset, "", 0, w, 0, PORTRAIT, base, "p")
    for extra in ({"pose_video": None}, {"face_video": None}, {"pose_video": "", "face_video": ""}):
        assert graph.window_graph(preset, "", 0, w, 0, PORTRAIT, {**base, **extra}, "p") == ref


# --- cablage du mode fourni ----------------------------------------------------------------------------

def _replace(names_extra, opts_params=None, preset_id="photoreal-replace", total=33, index=0):
    preset = graph.presets()[preset_id]
    windows = graph.plan(preset["mode"], total)
    names = {"video": "v.mp4", "ref": "r.png", "mask": "m.mkv", **names_extra}
    if index:
        names["continue"] = f"continue-{index}.mkv"
    return graph.window_graph(preset, "", 0, windows[index], index, PORTRAIT, names, "p",
                              opts=graph.options(preset, opts_params or {}))


def _source_of(g, link):
    """Remonte [noeud, sortie] jusqu'au LoadVideo qui l'alimente (via GetVideoComponents)."""
    node = g[link[0]]
    assert node["class_type"] == "GetVideoComponents" and link[1] == 0, node
    load = g[node["inputs"]["video"][0]]
    assert load["class_type"] == "LoadVideo"
    return load["inputs"]["file"]


def test_all_provided_skips_detection():
    g = _replace({"pose_video": "pose.mkv", "face_video": "face.mkv"})
    classes = {n["class_type"] for n in g.values()}
    assert "DWPreprocessor" not in classes and "OpenjutsuFaceCrop" not in classes
    a = g["30"]["inputs"]
    # chaque entree arrive sur SON entree de WanAnimateToVideo, pas la voisine
    assert _source_of(g, a["pose_video"]) == "pose.mkv"
    assert _source_of(g, a["face_video"]) == "face.mkv"
    # masque fourni (deja l'entree du mode replace) : character_mask et percage du decor viennent de mask
    assert a["character_mask"] == ["23", 0] and g["23"]["inputs"]["mask"] == ["17", 0]
    assert g["17"]["inputs"]["image"] == ["16", 0] and _source_of(g, ["16", 0]) == "m.mkv"
    assert g["25"]["inputs"]["mask"] == ["23", 0] and a["background_video"] == ["25", 0]
    # decor = video source (pate), jamais la pose ou le visage
    assert g["25"]["inputs"]["destination"] == ["11", 0] and _source_of(g, ["11", 0]) == "v.mp4"
    assert set(links(g)) <= set(g)
    # aucun noeud orphelin : tout noeud sert la sortie
    used = set(links(g)) | {graph.OUTPUT_NODE}
    assert set(g) == used


def test_pose_only_keeps_face_detection():
    g = _replace({"pose_video": "pose.mkv"})
    assert _source_of(g, g["30"]["inputs"]["pose_video"]) == "pose.mkv"
    assert "20" not in g  # DWPreprocessor du corps retire
    # visage toujours detecte dans la video source (preset face=true), avec les reglages d'origine du DWPose
    assert g["30"]["inputs"]["face_video"] == ["72", 0] and g["72"]["inputs"]["keypoints"] == ["71", 1]
    assert g["71"]["inputs"]["image"] == ["11", 0] and g["71"]["inputs"]["detect_face"] == "enable"
    default = _replace({})
    assert g["71"] == default["71"] and g["72"] == default["72"]
    assert set(links(g)) <= set(g)


def test_face_only_keeps_pose_detection():
    g = _replace({"face_video": "face.mkv"})
    assert g["30"]["inputs"]["pose_video"] == ["20", 0] and g["20"]["class_type"] == "DWPreprocessor"
    assert _source_of(g, g["30"]["inputs"]["face_video"]) == "face.mkv"
    assert "71" not in g and "72" not in g and set(links(g)) <= set(g)


def test_face_video_used_even_if_face_option_off():
    """Une entree fournie remplace sa detection : face=false n'empeche pas le visage fourni d'etre lu."""
    g = _replace({"face_video": "face.mkv"}, {"face": False})
    assert _source_of(g, g["30"]["inputs"]["face_video"]) == "face.mkv"


def test_provided_second_window_offset():
    """Fenetre 2 : la pose et le visage fournis sont le clip ENTIER ; WanAnimateToVideo les decale de
    video_frame_offset comme la video source et le masque (alignement image par image)."""
    g = _replace({"pose_video": "pose.mkv", "face_video": "face.mkv"}, total=150, index=1)
    w = graph.plan("replace", 150)[1]
    assert g["30"]["inputs"]["video_frame_offset"] == w["start"] + w["overlap"]
    assert _source_of(g, g["30"]["inputs"]["pose_video"]) == "pose.mkv"
    assert g["30"]["inputs"]["continue_motion"] == ["51", 0] and set(links(g)) <= set(g)


def test_recast_accepts_pose():
    g = _replace({"pose_video": "pose.mkv", "face_video": "face.mkv"}, preset_id="anime-recast")
    assert "DWPreprocessor" not in {n["class_type"] for n in g.values()}
    assert _source_of(g, g["30"]["inputs"]["pose_video"]) == "pose.mkv" and set(links(g)) <= set(g)


def test_provided_rejected_outside_wan_animate():
    preset = graph.presets()["outfit-swap"]
    with pytest.raises(ValueError, match="pose_video"):
        graph.window_graph(preset, "", 0, graph.plan("swap", 33)[0], 0, PORTRAIT,
                           {"video": "v.mp4", "mask": "m.mkv", "pose_video": "pose.mkv"}, "p")


# --- handler : validation des entrees fournies, avant tout GPU ---------------------------------------------

@pytest.fixture
def handler(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "runpod", types.SimpleNamespace())
    monkeypatch.setenv("OJ_MODES", "recast,replace,swap")
    sys.modules.pop("handler", None)
    import handler as h
    return h


def test_handler_accepts_provided(handler):
    p = handler.check_generate({"preset": "photoreal-replace"}, {"video", "ref", "mask", "pose_video", "face_video"})
    assert p["mode"] == "replace"


def test_handler_refuses_provided_on_vace(handler):
    with pytest.raises(ValueError, match="pose_video"):
        handler.check_generate({"preset": "product-swap"}, {"video", "ref", "mask", "pose_video"})


def _clip(path, n, size):
    import numpy as np
    from openjutsu import video
    w, h = size
    video.write_frames([np.zeros((h, w, 3), np.uint8)] * n, str(path), lossless=True)
    return str(path)


def test_handler_checks_provided_alignment(handler, tmp_path):
    ok = {"pose_video": _clip(tmp_path / "pose.mkv", 40, PORTRAIT), "face_video": _clip(tmp_path / "face.mkv", 40, (512, 512))}
    handler.check_provided(ok, PORTRAIT, 40)
    with pytest.raises(ValueError, match="at least 41 frames"):
        handler.check_provided(ok, PORTRAIT, 41)
    with pytest.raises(ValueError, match="face_video 480x832"):
        handler.check_provided({"face_video": ok["pose_video"]}, PORTRAIT, 40)
    with pytest.raises(ValueError, match="pose_video 512x512"):
        handler.check_provided({"pose_video": ok["face_video"]}, PORTRAIT, 40)


if __name__ == "__main__" and "--freeze" in sys.argv:
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps({name: digest(g) for name, g in default_matrix()}, indent=0, sort_keys=True) + "\n")
    print(f"{FIXTURE} : {len(json.loads(FIXTURE.read_text()))} graphes figes")
