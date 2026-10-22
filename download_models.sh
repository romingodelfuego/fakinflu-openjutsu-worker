#!/usr/bin/env bash
# fakinflu — poids du worker openjutsu (image recast + replace), BAKES dans l'image.
# Cibles core / dwpose : memes fichiers, commits et sha256 que wananimate/serverless/download_models.sh ; int8 : unet
# Wan2.2-Animate-14B v1 (celui des graphes d'Openjutsu), pas Wan-Animate-2.
# (sans la LoRA lightx2v, inutile : Openjutsu echantillonne 20 etapes a CFG 1 sans distillation).
# sam2 : checkpoint SAM 2.1 Hiera Large fp16 lu par openjutsu/sam2_mask.py (chargement strict du state dict).
#
#   MODELS_DIR=/comfyui/models ./download_models.sh core     # ~8 Go : umT5 fp8, CLIP vision H, VAE Wan 2.1
#   MODELS_DIR=/comfyui/models ./download_models.sh int8     # ~18 Go : Wan2.2-Animate-14B (v1) int8 convrot
#   MODELS_DIR=<controlnet_aux>/ckpts ./download_models.sh dwpose
#   MODELS_DIR=/models/sam2 ./download_models.sh sam2       # ~0,45 Go
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/comfyui/models}"

# DL <depot> <commit> <chemin dans le depot> <sous-dossier local> <sha256>
DL() {
  local repo="$1" rev="$2" src="$3" sub="$4" sha="$5" name
  name="$(basename "$src")"
  local dest="$MODELS_DIR/$sub/$name" url="https://huggingface.co/${repo}/resolve/${rev}/${src}"
  mkdir -p "$MODELS_DIR/$sub"
  echo ">> $sub/$name"
  local t0=$SECONDS
  if command -v aria2c >/dev/null; then
    aria2c -q -c -x 16 -s 16 -k 64M --max-tries=10 --retry-wait=5 --timeout=30 \
           --lowest-speed-limit=1M --file-allocation=none --console-log-level=warn \
           -d "$MODELS_DIR/$sub" -o "$name" "$url"
  else
    wget -q -c --tries=10 --timeout=30 --waitretry=5 -O "$dest" "$url"
  fi
  echo "   downloaded in $((SECONDS - t0)) s"
  echo "$sha  $dest" | sha256sum -c --quiet - || { echo "!! bad sha256: $dest"; rm -f "$dest"; exit 1; }
  test -s "$dest"
}

WA=e7181fe1896b7f2ce34120d1a3663413548240d7   # Comfy-Org/Wan-Animate-2 @ 2026-10-14 (encodeurs, CLIP vision, VAE)
W22=ee6f4a40737a995bf5818954cfce6d59443b0f04  # Comfy-Org/Wan_2.2_ComfyUI_Repackaged @ 2026-10-22 (unet Animate v1)

case "${1:-}" in
  core)
    DL Comfy-Org/Wan-Animate-2 "$WA" text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders \
       c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
    DL Comfy-Org/Wan-Animate-2 "$WA" clip_vision/clip_vision_h.safetensors clip_vision \
       64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161
    DL Comfy-Org/Wan-Animate-2 "$WA" vae/Wan2_1_VAE_bf16.safetensors vae \
       1ab9a32cc2c740f6e39d80d367ce5dcc28db8c71b79b28670546b8973e9d75f9 ;;
  int8)
    # Wan2.2-Animate-14B (v1), le modele des graphes d'Openjutsu (WanAnimateToVideo : pose, decor, masque).
    # PAS Comfy-Org/Wan-Animate-2 (Wan2.2-Animate-2-14B, noeud WanAnimate2ToVideo, sans mode remplacement) :
    # avec ce checkpoint, le mode replace rendait la zone masquee en silhouette grise (essai 5, 2026-10-22).
    DL Comfy-Org/Wan_2.2_ComfyUI_Repackaged "$W22" split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors diffusion_models \
       419aa0b3d9079a8a39126f2894c3cb2427c288189d2dc8342c61cf0663731cb3 ;;
  dwpose)
    # MODELS_DIR = <comfyui_controlnet_aux>/ckpts, sous-dossier = depot HF (custom_hf_download du pack).
    DL hr16/yolox-onnx a124b32c3b7c5cebda1c7cd96178f0f9d2050125 yolox_l.torchscript.pt hr16/yolox-onnx \
       80bc14b13c260c24b3014cd42c02994bf52296ab8fa2d80a60b6afe08c93ef42
    DL hr16/DWPose-TorchScript-BatchSize5 359d662a9b33b73f6d0f21732baf8845f17bb4be dw-ll_ucoco_384_bs5.torchscript.pt \
       hr16/DWPose-TorchScript-BatchSize5 d86a0b2b59fddc0901a7076e9f59c9f8602602133ed72511c693fd11eea23d91 ;;
  sam2)
    # Kijai/sam2-safetensors (apache-2.0) ; sha256 = oid LFS lu via l'API HF le 2026-10-22.
    DL Kijai/sam2-safetensors f885607d88bb3f9145efa49c3e3c50a9e5bf13eb sam2.1_hiera_large-fp16.safetensors . \
       278af5296fcc153aa94645a4981e93fe8b7a936a9fa27dbd71e8c850df6d7038 ;;
  *) echo "usage: $0 core|int8|dwpose|sam2" >&2; exit 2 ;;
esac
