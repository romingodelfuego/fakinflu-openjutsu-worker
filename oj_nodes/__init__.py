"""Noeuds ComfyUI propres au worker openjutsu (fakinflu).

OpenjutsuFaceCrop : `face_video` de WanAnimateToVideo = gros plan REEL du visage (pixels), comme le
pretraitement officiel de Wan 2.2 Animate (process_pipepline.py : get_face_bboxes(scale=1.3) puis
cv2.resize 512x512). ComfyUI le passe tel quel a l'encodeur de visage du modele (`face_video_pixels`).
Avant (<= v1.3.0) le graphe y envoyait un dessin DWPose des points du visage sur fond noir, calcule sur
l'image entiere : le modele ne recevait aucun visage.

Boite par image : points du visage DWPose (POSE_KEYPOINT, personne la plus grande), surface x1.3 en
gardant les proportions, bornee a l'image. Images sans visage detecte (dos tourne) : boite des voisines
(interpolation lineaire, ou la plus proche aux bords). Centre et taille lisses dans le temps (moyenne
glissante) pour que le gros plan ne tremble pas. Fonctions pures testables sans ComfyUI (face_boxes).
"""
import numpy as np

FACE_SIZE = 512
SCALE = 1.3      # surface de la boite / surface des points (Wan : get_face_bboxes scale=1.3)
MIN_POINTS = 5   # en dessous : visage non detecte sur cette image
SMOOTH = 5       # fenetre de la moyenne glissante (images)


def _face_points(frame_kps, width, height):
    """Points (x, y) en pixels du plus grand visage d'une image (dict POSE_KEYPOINT), ou None."""
    best = None
    for person in (frame_kps or {}).get("people") or []:
        flat = person.get("face_keypoints_2d") or []
        pts = np.asarray(flat, dtype=np.float64).reshape(-1, 3) if len(flat) % 3 == 0 and flat else np.zeros((0, 3))
        pts = pts[pts[:, 2] > 0][:, :2]
        if len(pts) < MIN_POINTS:
            continue
        if pts.max() <= 1.0 + 1e-6:  # coordonnees normalisees (controlnet_aux) -> pixels
            pts = pts * np.array([width, height])
        area = np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])
        if best is None or area > best[0]:
            best = (area, pts)
    return None if best is None else best[1]


def _box(pts):
    """(cx, cy, w, h) : boite des points, surface x SCALE, memes proportions (Wan get_face_bboxes)."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = max(x1 - x0, 2.0), max(y1 - y0, 2.0)
    k = np.sqrt(SCALE)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, w * k, h * k])


def face_boxes(keypoints, width, height, frames):
    """Boites (frames, 4) en pixels entiers x0, y0, x1, y1 ; ValueError si aucun visage sur tout le clip."""
    raw = [None] * frames
    for i in range(frames):
        pts = _face_points(keypoints[i] if i < len(keypoints) else None, width, height)
        if pts is not None:
            raw[i] = _box(pts)
    known = [i for i, b in enumerate(raw) if b is not None]
    if not known:
        raise ValueError("face_video : aucun visage detecte sur le clip (face=false pour ce job)")
    boxes = np.stack([raw[i] if raw[i] is not None else np.full(4, np.nan) for i in range(frames)])
    idx = np.arange(frames)
    for c in range(4):  # trous : interpolation lineaire, valeur la plus proche aux bords
        boxes[:, c] = np.interp(idx, known, boxes[known, c])
    if SMOOTH > 1 and frames > 1:
        pad = SMOOTH // 2
        padded = np.pad(boxes, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(SMOOTH) / SMOOTH
        boxes = np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(4)], axis=1)
    out = np.empty((frames, 4), dtype=np.int64)
    for i, (cx, cy, w, h) in enumerate(boxes):
        w, h = min(w, width), min(h, height)
        x0 = int(round(min(max(cx - w / 2, 0), width - w)))
        y0 = int(round(min(max(cy - h / 2, 0), height - h)))
        out[i] = (x0, y0, max(x0 + int(round(w)), x0 + 1), max(y0 + int(round(h)), y0 + 1))
    return out


class OpenjutsuFaceCrop:
    """Images (B, H, W, C) + POSE_KEYPOINT de DWPreprocessor -> gros plans du visage (B, 512, 512, C)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "keypoints": ("POSE_KEYPOINT",),
                             "size": ("INT", {"default": FACE_SIZE, "min": 64, "max": 1024, "step": 8})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "fakinflu/openjutsu"

    def crop(self, images, keypoints, size=FACE_SIZE):
        import torch
        import torch.nn.functional as F

        frames, height, width = images.shape[0], images.shape[1], images.shape[2]
        boxes = face_boxes(list(keypoints or []), width, height, frames)
        out = []
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            face = images[i:i + 1, y0:y1, x0:x1, :].movedim(-1, 1)
            out.append(F.interpolate(face, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1).movedim(1, -1))
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {"OpenjutsuFaceCrop": OpenjutsuFaceCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenjutsuFaceCrop": "Openjutsu : gros plan du visage (face_video)"}
