#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import warnings

# ────────────────────────────────────────────────────────────────────────────────
# 0) GLOBAL SETTINGS: suppress warnings + disable cuDNN
# ────────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
import torch
torch.backends.cudnn.enabled = False

# clear any cached CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ────────────────────────────────────────────────────────────────────────────────
# 1) Pick best GPU
# ────────────────────────────────────────────────────────────────────────────────
def select_best_gpu():
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ]).decode().splitlines()
    free = [int(x) for x in out]
    best = max(range(len(free)), key=lambda i: free[i])
    print(f"→ Using GPU {best} (free {free[best]} MiB)")
    return best

if torch.cuda.is_available():
    best = select_best_gpu()
    device = torch.device(f"cuda:{best}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"→ Running on {device}\n")

# ────────────────────────────────────────────────────────────────────────────────
# 2) Argument parsing / user config
# ────────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_url",
                   default="https://www.youtube.com/watch?v=YThX7_8I3m0")
    p.add_argument("--start", type=float, default=233.0)
    p.add_argument("--end",   type=float, default=243.0)
    p.add_argument("--queries", nargs='+', default=[
        "guy in black performing tricks on a bike",
        "a black bike used to perform tricks"
    ])
    # ───── ADDED FOR SUMMARY ────────────────────────────────────────────────────
    p.add_argument("--summary_percent", type=int, default=50,
                   help="Percent of video for summary (e.g. 50 for 50%)")
    p.add_argument("--summary_dir", type=str, default=".",
                   help="Directory to save the summary video")
    # ────────────────────────────────────────────────────────────────────────────
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# 3) Download & extract subclip
# ────────────────────────────────────────────────────────────────────────────────
from yt_dlp import YoutubeDL
from moviepy import VideoFileClip

def download_and_clip(url, start, end,
                      full="full_video.mp4",
                      clip="input_clip.mp4"):
    assert 0 < end-start <= 10, "Clip must be ≤10 s"
    # download
    with YoutubeDL({
        "format":"best[height<=360]",
        "outtmpl": full, "overwrites": True
    }) as ydl:
        ydl.download([url])
    # extract subclip
    with VideoFileClip(full) as v:
        sc = v.subclipped(start, end)
        sc.write_videofile(clip, audio_codec="aac")
    return clip

# ────────────────────────────────────────────────────────────────────────────────
# 4) Load MTTR model & checkpoint non-strict
# ────────────────────────────────────────────────────────────────────────────────
import ruamel.yaml
from torch.hub import get_dir, download_url_to_file
from models import build_model

def load_mttr(window=12):
    hub = get_dir()
    cfg_dir = os.path.join(hub, "configs"); os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "refer_youtube_vos.yaml")
    download_url_to_file(
        "https://raw.githubusercontent.com/mttr2021/MTTR/main/configs/refer_youtube_vos.yaml",
        cfg_path, progress=False
    )
    raw = ruamel.yaml.safe_load(open(cfg_path))
    cfg = {k:v["value"] for k,v in raw.items()}
    cfg.update(device="cpu", running_mode="eval")
    args = argparse.Namespace(**cfg)
    model, _, post = build_model(args)

    ckpt_dir = os.path.join(hub, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"refer-youtube-vos_window-{window}.pth.tar")
    if not os.path.exists(ckpt_path):
        import gdown
        gdown.download(
          id="1R_F0ETKipENiJUnVwarHnkPmUIcKXRaL",
          output=ckpt_path, quiet=True
        )
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd, strict=False)

    return model.to(device), post

# ────────────────────────────────────────────────────────────────────────────────
# 5) Helpers: NestedTensor & mask application
# ────────────────────────────────────────────────────────────────────────────────
import torchvision
import torchvision.transforms.functional as F
from einops import rearrange
import numpy as np
from PIL import Image

class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask    = mask

def to_nested(frames: torch.Tensor):
    # frames: [T,C,H,W]
    T,C,H,W = frames.shape
    batched = frames.unsqueeze(0).transpose(0,1)  # [T,1,C,H,W]
    mask    = torch.zeros((T, 1, H, W), dtype=torch.bool, device=frames.device)
    return NestedTensor(batched, mask)

def apply_mask_np(img, mask, color, alpha=0.7):
    c  = np.array(color)[None,None]/255.0
    m3 = mask[...,None].astype(float)*alpha
    return img*(1-m3) + c*m3

# ────────────────────────────────────────────────────────────────────────────────
# 6) Inference & assemble output (window_length=12)
# ────────────────────────────────────────────────────────────────────────────────
from moviepy import AudioFileClip, ImageSequenceClip
from tqdm import tqdm

def run_inference(clip_path, queries, model, post):
    video, _, meta = torchvision.io.read_video(clip_path, pts_unit='sec')
    vid = rearrange(video, 't h w c -> t c h w').float().div(255).to(device)
    vid = F.resize(vid, size=360, max_size=640)
    vid = F.normalize(vid,
                      mean=[0.485,0.456,0.406],
                      std=[0.229,0.224,0.225])
    T,C,H,W = vid.shape

    ws, wo = 12, 3
    windows = [vid[i:i+ws] for i in range(0, T, ws-wo)]
    qs      = [q.lower().strip() for q in queries]

    all_masks = []
    for q in tqdm(qs, desc="text queries"):
        buf = torch.zeros((T,1,H,W), device="cpu")
        for i, win in enumerate(tqdm(windows, desc="windows")):
            nt  = to_nested(win)
            idx = torch.arange(nt.tensors.shape[0], device=device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                out = model(nt, idx, [q])
                pm  = post(
                    out,
                    [{"resized_frame_size":(H,W),
                      "original_frame_size":(H,W)}],
                    (H,W)
                )[0]["pred_masks"]
            start = i*(ws-wo)
            buf[start:start+ws] = pm.cpu()
        all_masks.append(buf.numpy())

    colors = [(41,171,226),(237,30,121),(35,161,90),(255,148,59)]
    frames = rearrange(vid, 't c h w -> t h w c').cpu().numpy()
    out    = []
    for f, masks in zip(frames, zip(*all_masks)):
        img = f.copy()
        for m,c in zip(masks, colors):
            img = apply_mask_np(img, m[0], c)
        out.append((img*255).astype(np.uint8))

    out_path = "output_clip.mp4"
    clip_v   = ImageSequenceClip(out, fps=meta["video_fps"])
    clip_v   = clip_v.with_audio(AudioFileClip(clip_path))
    clip_v.write_videofile(out_path, audio=True, fps=meta["video_fps"])
    return out_path

# ────────────────────────────────────────────────────────────────────────────────
# 7) UNIFORM-SAMPLING SUMMARY INTEGRATION
# ────────────────────────────────────────────────────────────────────────────────
import cv2

def summarise_video(video_path, percent, out_dir):
    # open original clip
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # compute sampling rate
    sampling_rate = 100 // percent
    if (100 % percent) != 0:
        sampling_rate += 1

    # select frames uniformly
    frame_indices = [i for i in range(frame_count) if i % sampling_rate == 0]
    print(f"Uniformly sampling {len(frame_indices)}/{frame_count} frames")

    # prepare writer
    summary_path = os.path.join(out_dir, f"summary_{percent}pct.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(summary_path, fourcc, fps, (width, height))

    # write selected frames
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Summary saved to {summary_path}")
    return summary_path

# ────────────────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    clip    = download_and_clip(args.video_url, args.start, args.end)
    model, post = load_mttr(window=12)
    out     = run_inference(clip, args.queries, model, post)

    # ─── call summarisation ─────────────────────────────────────────────────────
    summarise_video(out,
                    args.summary_percent,
                    args.summary_dir)
    # ───────────────────────────────────────────────────────────────────────────────

    print(f"\nOutput saved to {out}\n")

if __name__ == "__main__":
    main()