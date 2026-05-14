"""Shared video frame sampling for VSI-Bench adapters."""
import numpy as np
from PIL import Image


def sample_frames_decord(video_path, num_frames=32):
    """Return list[PIL.Image] of `num_frames` uniformly sampled frames."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()   # (T,H,W,3) uint8
    return [Image.fromarray(f).convert("RGB") for f in frames]


def sample_frames_pyav(video_path, num_frames=32):
    """Fallback: same interface, using PyAV."""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames
    if total == 0:
        # Some streams don't expose count; fall back to decoding all.
        frames = [f.to_image() for f in container.decode(video=0)]
        total = len(frames)
        if total <= num_frames:
            return frames
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        return [frames[i] for i in indices]
    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    iset = set(indices)
    out, i = [], 0
    for f in container.decode(video=0):
        if i in iset:
            out.append(f.to_image().convert("RGB"))
            if len(out) == num_frames:
                break
        i += 1
    return out


def sample_frames(video_path, num_frames=32):
    """Public entry: try decord first, fall back to pyav."""
    try:
        return sample_frames_decord(video_path, num_frames)
    except ImportError:
        return sample_frames_pyav(video_path, num_frames)
