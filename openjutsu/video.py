"""Local video preparation, overlapping windows, and output assembly.

From SydFloyd/openjutsu app/video.py @ f783a001 (Apache-2.0). fakinflu changes: normalize() keeps the
orientation (480x832 for a portrait source) instead of cropping every clip to 832x480; replace_mask()."""
from fractions import Fraction
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
import numpy as np

FPS = 16
WIDTH, HEIGHT = 832, 480


def ffmpeg(*args):
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *map(str, args)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())


def probe(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = stream.frames or sum(1 for _ in container.decode(video=0))
        return {"width": stream.width, "height": stream.height,
                "frames": count, "fps": float(stream.average_rate or FPS)}


def target_size(source):
    info = probe(source)
    return (HEIGHT, WIDTH) if info["height"] > info["width"] else (WIDTH, HEIGHT)


def normalize(source, destination):
    w, h = target_size(source)
    ffmpeg("-i", source, "-map", "0:v:0", "-an", "-vf",
           f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS}",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", destination)
    info = probe(destination)
    if not info["frames"]:
        raise ValueError("The clip contains no video frames")
    return info


def windows(total, capacity, overlap):
    """Cover each source frame; pad the final window to Wan's 4n+1 length."""
    if total < 1 or capacity < 9 or (capacity - 1) % 4 or overlap >= capacity:
        raise ValueError("Need frames>0, a 4n+1 window>=9, and overlap<window")
    start = 0
    while True:
        count = min(capacity, total - start)
        length = max(1, ((count - 1 + 3) // 4) * 4 + 1)
        yield {"start": start, "count": count, "length": length,
               "overlap": 0 if start == 0 else overlap}
        if start + count >= total:
            break
        start += capacity - overlap


def slice_video(source, destination, start, count, length, mask=False):
    filters = f"trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS"
    if length > count:
        filters += f",tpad=stop_mode=clone:stop={length - count}"
    filters += f",setpts=N/({FPS}*TB)"
    codec = ["-c:v", "ffv1", "-pix_fmt", "gray"] if mask else ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    ffmpeg("-i", source, "-an", "-vf", filters, "-r", FPS, "-frames:v", length, *codec, destination)


def read_frames(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def write_frames(frames, path, fps=FPS, lossless=False):
    height, width = frames[0].shape[:2]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1" if lossless else "libx264", rate=Fraction(str(fps)))
        stream.width, stream.height = width, height
        stream.pix_fmt = "bgr0" if lossless else "yuv420p"
        if not lossless:
            stream.options = {"crf": "18"}
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def block_mask(mask, expand, block):
    """Masque binaire (H, W) uint8 0/1 -> elargi de `expand` px (carre), puis, si `block` > 0, chaque
    case de `block` px de la boite englobante touchee par le masque est remplie (BlockifyMask du
    template ComfyUI, get_aug_mask de Wan 2.2). Meme algorithme que l'apercu de l'interface fakinflu."""
    out = mask.astype(np.uint8)
    if expand > 0:
        import cv2
        out = cv2.dilate(out, np.ones((2 * expand + 1, 2 * expand + 1), np.uint8))
    if block > 0 and out.any():
        ys, xs = np.nonzero(out)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        res = out.copy()
        for y in range(y0, y1, block):
            for x in range(x0, x1, block):
                ye, xe = min(y + block, y1), min(x + block, x1)
                if out[y:ye, x:xe].any():
                    res[y:ye, x:xe] = 1
        out = res
    return out


def replace_mask(source, destination, expand, block, frames=None):
    """Masque du mode replace : chaque image binarisee (> 127), elargie, en blocs -> FFV1 sans perte."""
    out = []
    for i, image in enumerate(read_frames(source)):
        if frames is not None and i >= frames:
            break
        m = block_mask(image[:, :, 0] > 127, expand, block)
        out.append(np.repeat(m[:, :, None] * 255, 3, axis=2).astype(np.uint8))
    write_frames(out, destination, lossless=True)
    return len(out)


def assemble(chunks, plan, destination, total, mode, audio_source=None):
    # At 480p a 30-second clip occupies under 600 MB here, on the CPU app host.
    frames = []
    for path, window in zip(chunks, plan):
        current = read_frames(path)[:window["count"]]
        if len(current) != window["count"]:
            raise ValueError(f"Chunk has {len(current)} frames, expected {window['count']}")
        overlap = window["overlap"]
        if overlap and mode != "recast":
            for i in range(overlap):
                alpha = (i + 1) / (overlap + 1)
                frames[-overlap + i] = np.rint(frames[-overlap + i] * (1 - alpha) + current[i] * alpha).astype(np.uint8)
        frames.extend(current[overlap:])
    if len(frames) != total:
        raise ValueError(f"Assembled {len(frames)} frames, expected {total}")
    silent = Path(destination).with_name("assembled-silent.mp4")
    write_frames(frames, silent)
    if audio_source is None:
        silent.replace(destination)
    else:
        ffmpeg("-i", silent, "-i", audio_source, "-map", "0:v:0", "-map", "1:a?",
               "-c:v", "copy", "-c:a", "aac", "-t", str(total / FPS), "-movflags", "+faststart", destination)
        silent.unlink()
