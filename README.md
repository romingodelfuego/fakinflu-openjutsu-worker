# fakinflu — worker `openjutsu`

[Openjutsu](https://github.com/SydFloyd/openjutsu) (clone open source de Higgsfield Genjutsu, commit
`f783a001`, Apache-2.0, voir `NOTICE`) en serverless RunPod, sans FastAPI ni comfy-sdk : le handler
soumet les graphes d'Openjutsu au ComfyUI local de l'image, fenêtre par fenêtre.

| Image | Modes | Endpoint |
| --- | --- | --- |
| `Dockerfile` | `recast`, `replace` (Wan 2.2 Animate 14B int8, DWPose) + masque SAM 2.1 | `fakinflu-openjutsu-worker` |
| `Dockerfile.vace` (à venir) | `swap`, `restyle` (Wan 2.1 VACE 14B) | `fakinflu-openjutsu-vace-worker` |

## Contrat (`contracts/openjutsu.input.schema.json` côté fakinflu)

```json
{"input": {"run_id": "…", "r2": true,
  "inputs":  {"video": {…}, "ref": {…}, "mask": {…}},
  "outputs": {"mask": {…}, "mask_preview": {…}, "video": {…}, "preview": {…}, "probe": {…}},
  "params": {"action": "probe|mask|generate", "points": [[x, y, 1]], "preset": "photoreal-replace",
             "prompt": "", "seed": 42, "max_frames": 33, "steps": null, "cfg": null, "shift": null,
             "style": null, "negative": null, "sampler": null, "scheduler": null,
             "relight": null, "face": null, "distill": null, "mask_expand": null, "mask_block": null, "unet": null}}}
```

- `video` : clip **déjà normalisé** par le client (`make oj-prep` : 16 i/s, 480×832 ou 832×480) ; refusé sinon.
- `mask` : `points` en pixels du clip, `[x, y, label]` sur la première image ou `[x, y, label, frame]` sur
  n'importe quelle image (label 1 = sujet, 0 = à exclure). Chaque image annotée corrige le suivi (bras levé,
  sujet qui réapparaît) ; la première image annotée doit avoir un point sujet. SAM 2.1 suit le sujet sur
  tout le clip (en avant, puis en arrière si la première image annotée n'est pas l'image 0) → `mask.mkv`
  (FFV1 sans perte) + `mask_preview.png`.
- `generate` : preset de `openjutsu/presets/` servi par l'image (`OJ_MODES`). Recast et replace :
  fenêtres de 77 images, 5 reportées (`continue_motion`, `video_frame_offset`). Sortie `result.mp4`
  sans audio (le client remet l'audio d'origine), `preview.jpg`.
- `replace` : Wan 2.2 Animate en mode remplacement : squelette DWPose de la source, décor d'origine
  (zone du masque élargi remplie de gris), personnage de la référence. Preset `photoreal-replace`.
- Tout réglage du preset se remplace par job (`null` = celui du preset) : `style` remplace le texte de style
  du preset (ajouté après `prompt` ; `""` = aucun), `negative` le prompt négatif du graphe, `sampler` /
  `scheduler` ceux du KSampler (`graph.SAMPLERS` / `SCHEDULERS`), plus `steps`, `cfg`, `shift` et les options
  Wan Animate (`relight`, `face`, `distill` = LoRA lightx2v, `mask_expand`, `mask_block`, `unet`). Le preset ne
  sert que de point de départ : fakinflu y ajoute ses propres presets (base + réglages).
- `probe` : VRAM, RAM du conteneur, disque, poids présents, version.

## Tests (0 crédit)

```bash
python -m pytest -q tests     # fenêtres, assemblage, graphes recast/replace verticaux, refus du handler
```
