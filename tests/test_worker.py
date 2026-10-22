"""Tests CPU du worker openjutsu (0 credit) : python -m pytest -q tests"""
import sys
import types

import numpy as np
import pytest

from openjutsu import graph, video

PORTRAIT = (480, 832)


def links(g):
    for node in g.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                yield value[0]


# --- fenetres ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode,total", [("recast", 33), ("recast", 77), ("recast", 150), ("replace", 241),
                                        ("swap", 81), ("restyle", 200)])
def test_windows_cover_every_frame(mode, total):
    plan = graph.plan(mode, total)
    covered = []
    for w in plan:
        assert (w["length"] - 1) % 4 == 0 and w["length"] >= w["count"]
        covered.extend(range(w["start"] + w["overlap"], w["start"] + w["count"]))
    assert covered == list(range(total))
    assert plan[0]["overlap"] == 0
    assert all(w["overlap"] == (5 if mode in graph.LONG else 9) for w in plan[1:])


def test_short_clip_padded_to_4n_plus_1():
    (w,) = graph.plan("replace", 31)
    assert w == {"start": 0, "count": 31, "length": 33, "overlap": 0}


# --- assemblage ---------------------------------------------------------------------------------------

def _clip(path, n, value, size=PORTRAIT):
    w, h = size
    video.write_frames([np.full((h, w, 3), value + i, np.uint8) for i in range(n)], str(path), lossless=True)


@pytest.mark.parametrize("mode", ["recast", "swap"])
def test_assemble_frame_count(tmp_path, mode):
    total = 100
    plan = graph.plan(graph.assemble_mode(mode), total, chunk_frames=41)
    chunks = []
    for i, w in enumerate(plan):
        p = tmp_path / f"chunk-{i}.mkv"
        _clip(p, w["length"], 10 * i)
        chunks.append(p)
    out = tmp_path / "result.mp4"
    video.assemble(chunks, plan, out, total, mode)
    info = video.probe(out)
    assert info["frames"] == total and (info["width"], info["height"]) == PORTRAIT


# --- graphes ---------------------------------------------------------------------------------------

def _window_graphs(preset_id, total=33, names=None):
    preset = graph.presets()[preset_id]
    out = []
    for i, w in enumerate(graph.plan(preset["mode"], total)):
        n = dict(names or {})
        if i:
            n["continue"] = f"continue-{i}.mkv"
        out.append(graph.window_graph(preset, "a boxer", 42, w, i, PORTRAIT, n, "fakinflu/r1"))
    return out


def test_recast_graph_portrait():
    (g,) = _window_graphs("anime-recast", names={"video": "oj_x_input.mp4", "ref": "oj_x_ref.png"})
    assert g["1"]["inputs"]["unet_name"] == "wan2.2_animate_14B_int8_convrot.safetensors"
    assert g["2"]["inputs"]["clip_name"] == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    assert (g["30"]["inputs"]["width"], g["30"]["inputs"]["height"]) == PORTRAIT
    assert g["30"]["inputs"]["length"] == 33 and g["30"]["inputs"]["video_frame_offset"] == 0
    assert g["10"]["inputs"]["file"] == "oj_x_input.mp4" and g["12"]["inputs"]["image"] == "oj_x_ref.png"
    assert g["5"]["inputs"]["text"].startswith("a boxer, 2D anime character")
    assert g["40"]["inputs"]["seed"] == 42 and g["40"]["inputs"]["steps"] == 20
    assert g["44"]["inputs"]["filename_prefix"] == "fakinflu/r1/chunk-0"
    assert "50" not in g and set(links(g)) <= set(g)


def test_recast_second_window_continues_motion():
    _, g = _window_graphs("anime-recast", total=120, names={"video": "v.mp4", "ref": "r.png"})
    assert g["30"]["inputs"]["video_frame_offset"] == 72 + 5
    assert g["30"]["inputs"]["continue_motion"] == ["51", 0]
    assert g["50"]["inputs"]["file"] == "continue-1.mkv"
    assert set(links(g)) <= set(g)


def test_replace_graph_portrait():
    (g,) = _window_graphs("photoreal-replace", names={"video": "v.mp4", "ref": "r.png", "mask": "m.mkv"})
    assert {g[k]["class_type"] for k in g} >= {"WanAnimateToVideo", "DWPreprocessor", "ThresholdMask", "GrowMask",
                                               "ImageCompositeMasked", "EmptyImage", "TrimVideoLatent", "SaveVideo"}
    assert g["1"]["inputs"]["unet_name"] == "wan2.2_animate_14B_int8_convrot.safetensors"
    assert g["2"]["inputs"]["clip_name"] == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    for node in ("30", "24"):
        assert (g[node]["inputs"]["width"], g[node]["inputs"]["height"]) == PORTRAIT
    assert g["15"]["inputs"]["file"] == "m.mkv" and g["18"]["inputs"]["expand"] == 12
    assert g["30"]["inputs"]["background_video"] == ["25", 0] and g["30"]["inputs"]["character_mask"] == ["18", 0]
    assert set(links(g)) <= set(g)
    assert graph.assemble_mode("replace") == "recast"


def test_reference_optional_nodes_removed():
    preset = dict(graph.presets()["anime-recast"], requires_reference_image=False)
    g = graph.window_graph(preset, "", 0, graph.plan("recast", 33)[0], 0, PORTRAIT, {"video": "v.mp4"}, "p")
    assert not {"12", "13", "14"} & set(g) and set(links(g)) <= set(g)


def test_unet_dtype_env(monkeypatch):
    monkeypatch.setenv("OJ_UNET_DTYPE", "fp8_e4m3fn")
    (g,) = _window_graphs("photoreal-replace", names={"video": "v", "ref": "r", "mask": "m"})
    assert g["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"


def test_every_preset_has_its_workflow():
    for p in graph.presets().values():
        assert (graph.ROOT / "workflows" / f"{p['workflow']}.json").is_file(), p["id"]


# --- handler : refus avant tout calcul ---------------------------------------------------------------

@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setitem(sys.modules, "runpod", types.SimpleNamespace())
    monkeypatch.setenv("OJ_MODES", "recast,replace")
    sys.modules.pop("handler", None)
    import handler as h
    return h


def test_replace_without_mask_refused(handler):
    with pytest.raises(ValueError, match="mask"):
        handler.check_generate({"preset": "photoreal-replace"}, {"video", "ref"})


def test_reference_required(handler):
    with pytest.raises(ValueError, match="ref"):
        handler.check_generate({"preset": "anime-recast"}, {"video"})


def test_mode_not_in_image_refused(handler):
    with pytest.raises(ValueError, match="not served"):
        handler.check_generate({"preset": "product-swap"}, {"video", "ref", "mask"})


def test_events_unknown_action(handler):
    events = list(handler.handler({"id": "j", "input": {"params": {"action": "nope"}}}))
    assert events[0]["type"] == "version" and events[-1]["type"] == "error"


def test_points_checked(handler):
    info = {"width": 480, "height": 832}
    handler.check_points([[240, 400, 1], [10, 10, 0]], info)
    for bad in ([], [[500, 10, 1]], [[1, 2, 3]], "x"):
        with pytest.raises(ValueError):
            handler.check_points(bad, info)


def test_unnormalized_video_refused(handler, tmp_path):
    p = tmp_path / "v.mkv"
    _clip(p, 5, 0, size=(640, 360))
    with pytest.raises(ValueError, match="not normalized"):
        handler.check_video(str(p))
