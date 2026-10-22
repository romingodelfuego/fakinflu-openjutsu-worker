#!/usr/bin/env python3
"""Garde-fou de BUILD : chaque poids nomme par les graphes de l'image existe et n'est pas vide.

Usage : check_weights.py <dossier workflows> <dossier models> [graphe ...]
"""
import json
import os
import sys
from pathlib import Path

LOADERS = {"UNETLoader": ("unet_name", "diffusion_models"), "CLIPLoader": ("clip_name", "text_encoders"),
           "VAELoader": ("vae_name", "vae"), "CLIPVisionLoader": ("clip_name", "clip_vision")}


def main(workflows, models, names):
    bad = []
    for name in names:
        graph = json.loads((Path(workflows) / f"{name}.json").read_text())
        for node in graph.values():
            if node["class_type"] in LOADERS:
                key, sub = LOADERS[node["class_type"]]
                path = Path(models) / sub / node["inputs"][key]
                if not path.is_file() or not os.path.getsize(path):
                    bad.append(f"{name}: {path}")
    if bad:
        print("missing weights:", *bad, sep="\n  ")
        return 1
    print("graph weights OK:", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["recast_wan22_animate", "replace_wan22_animate"]))
