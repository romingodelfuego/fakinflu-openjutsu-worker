# fakinflu — worker RunPod : openjutsu, image recast + replace (Wan 2.2 Animate 14B int8 + DWPose + SAM 2.1).
# Openjutsu : SydFloyd/openjutsu @ f783a001 (Apache-2.0, voir NOTICE). Build par RunPod a chaque push sur main.
# Image swap + restyle (Wan 2.1 VACE) : Dockerfile.vace, endpoint separe (a venir).
#
# Base : celle de wananimate. 5.10.0-base = ComfyUI 0.34.0, qui sait charger l'unet Wan Animate 2 int8
# 'convrot' (le latest-base initial faisait "no match" sur ce checkpoint). Les garde-fous ci-dessous
# echouent le BUILD si un noeud des graphes manque : on le sait sans payer un job GPU.
ARG WORKER_TAG=5.10.0-base
FROM runpod/worker-comfyui:${WORKER_TAG}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN grep -q 'character_mask' /comfyui/comfy_extras/nodes_wan.py \
 && for n in WanAnimateToVideo TrimVideoLatent ThresholdMask GrowMask ImageToMask ImageCompositeMasked EmptyImage \
             LoadVideo GetVideoComponents CreateVideo SaveVideo CLIPVisionEncode ModelSamplingSD3 \
             LoraLoaderModelOnly ImageScale; do \
      grep -rqw "$n" /comfyui/nodes.py /comfyui/comfy_extras || { echo "!! missing ComfyUI node: $n"; exit 1; }; \
    done

# ComfyUI dimensionne sa memoire epinglee sur la RAM de l'HOTE, pas du conteneur (cf. scail, wananimate) :
# ${COMFY_EXTRA_ARGS} ajoute au start.sh (defaut ci-dessous, modifiable dans l'env sans rebuild).
RUN sed -i 's#--log-stdout &#--log-stdout ${COMFY_EXTRA_ARGS} \&#' /start.sh \
 && test "$(grep -c 'COMFY_EXTRA_ARGS} &' /start.sh)" -ge 1
ENV COMFY_EXTRA_ARGS="--disable-pinned-memory"

RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
 && rm -rf /var/lib/apt/lists/*

# Versions de l'image de base FIGEES pour toutes les installations : un torch remplace casserait CUDA,
# un huggingface-hub 1.x casserait ComfyUI au demarrage.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|huggingface-hub|huggingface_hub)==' > /tmp/pins.txt \
 && cat /tmp/pins.txt

# --- DWPose (comfyui_controlnet_aux, commit teste par Openjutsu) ------------------------------------
ARG AUX_COMMIT=a587a4c1594524c1d2de5cd1a86cd868ff448e1c
RUN git clone -q https://github.com/Fannovel16/comfyui_controlnet_aux /comfyui/custom_nodes/comfyui_controlnet_aux \
 && git -C /comfyui/custom_nodes/comfyui_controlnet_aux checkout -q ${AUX_COMMIT} \
 && pip install --no-cache-dir -r /comfyui/custom_nodes/comfyui_controlnet_aux/requirements.txt -c /tmp/pins.txt \
 && grep -rq '"DWPreprocessor"' /comfyui/custom_nodes/comfyui_controlnet_aux/

# --- SAM 2.1 (action mask, hors ComfyUI) ------------------------------------------------------------
ARG SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
RUN SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-build-isolation -c /tmp/pins.txt \
      "git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}" \
 && pip install --no-cache-dir -c /tmp/pins.txt av opencv-python-headless imageio-ffmpeg iopath safetensors websocket-client \
 && python3 -c "import sam2, av, cv2, imageio_ffmpeg, iopath; from sam2.build_sam import build_sam2_video_predictor" \
 && pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|huggingface-hub|huggingface_hub)==' | diff - /tmp/pins.txt

# --- Poids BAKES (~26 Go) ----------------------------------------------------------------------------
COPY download_models.sh /tmp/download_models.sh
RUN MODELS_DIR=/comfyui/custom_nodes/comfyui_controlnet_aux/ckpts bash /tmp/download_models.sh dwpose \
 && test -s /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt \
 && test -s /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt
RUN MODELS_DIR=/models/sam2 bash /tmp/download_models.sh sam2 \
 && test -s /models/sam2/sam2.1_hiera_large-fp16.safetensors
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh core \
 && test -s /comfyui/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
 && test -s /comfyui/models/clip_vision/clip_vision_h.safetensors \
 && test -s /comfyui/models/vae/Wan2_1_VAE_bf16.safetensors
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh int8 \
 && test -s /comfyui/models/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh loras \
 && test -s /comfyui/models/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors \
 && test -s /comfyui/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
 && rm -f /tmp/download_models.sh

# --- Handler fakinflu (remplace /handler.py de worker-comfyui) ---------------------------------------
COPY transfer.py VERSION* /fakinflu/
COPY openjutsu /fakinflu/openjutsu
COPY handler.py /handler.py
COPY test_input.json /test_input.json
# Chaque graphe nomme des poids presents dans l'image (echoue le build sinon).
COPY check_weights.py /tmp/check_weights.py
RUN python3 /tmp/check_weights.py /fakinflu/openjutsu/workflows /comfyui/models && rm /tmp/check_weights.py
ENV PYTHONPATH=/fakinflu \
    OJ_MODES=recast,replace \
    SAM2_CKPT=/models/sam2/sam2.1_hiera_large-fp16.safetensors \
    COMFY_LOG_LEVEL=INFO

CMD ["/start.sh"]
