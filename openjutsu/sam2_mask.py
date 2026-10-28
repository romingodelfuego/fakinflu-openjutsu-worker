"""SAM 2.1 click tracking with binary, lossless FFV1 masks.

From SydFloyd/openjutsu app/sam2_mask.py @ f783a001 (Apache-2.0). Modified: release() (frees the GPU for
ComfyUI after a mask job) and points on several frames ([x, y, label, frame]) : each annotated frame is a
SAM 2 prompt, propagation forward from the first one, then backward if it is not frame 0."""
from fractions import Fraction
import os
from pathlib import Path
import tempfile
import threading

import av
import cv2
import numpy as np
import torch

CKPT = os.environ.get("SAM2_CKPT", "models/sam2/sam2.1_hiera_large-fp16.safetensors")
CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
_pred = None
_lock = threading.Lock()


def _predictor():
    global _pred
    if _pred is None:
        from sam2.build_sam import build_sam2_video_predictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if Path(CKPT).suffix == ".safetensors":
            from safetensors.torch import load_file
            predictor = build_sam2_video_predictor(CFG, device="cpu", apply_postprocessing=False)
            predictor.load_state_dict(load_file(CKPT))
            predictor.to(device)
        else:
            predictor = build_sam2_video_predictor(CFG, CKPT, device=device, apply_postprocessing=False)
        _pred = predictor
    return _pred


def propagate_mask(video, points, output, preview):
    if not points or any(len(p) not in (2, 3) for p in points):
        raise ValueError("Click at least one point on the subject")
    with _lock, tempfile.TemporaryDirectory(prefix="openjutsu-sam2-") as temporary:
        folder = Path(temporary)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 16
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            cv2.imwrite(str(folder / f"{count:05d}.jpg"), image)
            count += 1
        cap.release()
        if not count:
            raise ValueError("Could not decode video")
        pred = _predictor()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            state = pred.init_state(video_path=str(folder), offload_video_to_cpu=True, offload_state_to_cpu=True)
            by_frame = {}
            for p in points:
                frame = min(int(p[3]) if len(p) == 4 else 0, count - 1)
                by_frame.setdefault(frame, []).append(p)
            for frame, group in sorted(by_frame.items()):
                coordinates = np.asarray([p[:2] for p in group], dtype=np.float32)
                labels = np.asarray([p[2] if len(p) >= 3 else 1 for p in group], dtype=np.int32)
                pred.add_new_points_or_box(state, frame_idx=frame, obj_id=1, points=coordinates, labels=labels)
            first = cv2.imread(str(folder / "00000.jpg"))
            height, width = first.shape[:2]
            # masques binaires gardes en memoire (480x832 : ~0,4 Mo/image) pour les ecrire dans l'ordre
            masks = {}
            for index, _, logits in pred.propagate_in_video(state):
                masks[index] = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
            if min(by_frame) > 0:
                for index, _, logits in pred.propagate_in_video(state, start_frame_idx=min(by_frame), reverse=True):
                    masks.setdefault(index, (logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255)
            empty = np.zeros((height, width), np.uint8)
            overlay = first.copy()
            overlay[masks.get(0, empty) > 0] = (0, 0, 255)
            cv2.imwrite(str(preview), cv2.addWeighted(first, 0.5, overlay, 0.5, 0))
            with av.open(str(output), "w") as container:
                stream = container.add_stream("ffv1", rate=Fraction(str(fps)))
                stream.width, stream.height, stream.pix_fmt = width, height, "gray"
                for index in range(count):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(masks.get(index, empty), format="gray")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            pred.reset_state(state)
        return {"frames": count, "fps": fps, "codec": "ffv1", "binary_mask": True}


def release():
    global _pred
    with _lock:
        _pred = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
