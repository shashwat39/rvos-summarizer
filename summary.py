#!/usr/bin/env python3
import os, sys, subprocess, argparse, warnings

# ────────────────────────────────────────────────────────────────────────────────
# 0) GLOBAL SETUP: disable warnings & cuDNN, pick best GPU
# ────────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
import torch; torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def select_best_gpu():
    out = subprocess.check_output([
        "nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"
    ]).decode().splitlines()
    free = [int(x) for x in out]
    best= max(range(len(free)), key=lambda i: free[i])
    print(f"→ GPU {best} ({free[best]} MiB free)")
    return best

if torch.cuda.is_available():
    dev = select_best_gpu()
    device = torch.device(f"cuda:{dev}"); torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"→ Running on {device}\n")

# ────────────────────────────────────────────────────────────────────────────────
# 1) ARGS
# ────────────────────────────────────────────────────────────────────────────────
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--video_url", default="https://www.youtube.com/watch?v=YThX7_8I3m0")
    p.add_argument("--start", type=float, default=233.0)
    p.add_argument("--end",   type=float, default=243.0)
    p.add_argument("--queries", nargs='+', default=[
        "guy in black performing tricks on a bike",
        "a black bike used to perform tricks"
    ])
    p.add_argument("--summary_percent", type=int, default=50,
                   help="Percent of frames to keep")
    p.add_argument("--summary_dir", type=str, default=".",
                   help="Where to save outputs")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# 2) DOWNLOAD & SUBCLIP
# ────────────────────────────────────────────────────────────────────────────────
from yt_dlp import YoutubeDL
from moviepy import VideoFileClip

def download_and_clip(url, start, end,
                      full="full_video.mp4",
                      clip="input_clip.mp4"):
    assert 0 < end-start <=10, "Clip must be ≤10 s"
    with YoutubeDL({"format":"best[height<=360]","outtmpl":full,"overwrites":True}) as ydl:
        ydl.download([url])
    with VideoFileClip(full) as v:
        sc = v.subclipped(start, end)
        sc.write_videofile(clip, audio_codec="aac")
    return clip

# ────────────────────────────────────────────────────────────────────────────────
# 3) LOAD MTTR MODEL
# ────────────────────────────────────────────────────────────────────────────────
import ruamel.yaml
from torch.hub import get_dir, download_url_to_file
from models import build_model

def load_mttr(window=12):
    hub = get_dir()
    cfg_dir = os.path.join(hub,"configs"); os.makedirs(cfg_dir,exist_ok=True)
    cfg_path= os.path.join(cfg_dir,"refer_youtube_vos.yaml")
    download_url_to_file(
      "https://raw.githubusercontent.com/mttr2021/MTTR/main/configs/refer_youtube_vos.yaml",
      cfg_path, progress=False)
    raw=ruamel.yaml.safe_load(open(cfg_path))
    cfg={k:v["value"] for k,v in raw.items()}
    cfg.update(device="cpu", running_mode="eval")
    args=argparse.Namespace(**cfg)
    model,_,post=build_model(args)

    ckpt_dir=os.path.join(hub,"checkpoints"); os.makedirs(ckpt_dir,exist_ok=True)
    ckpt_path=os.path.join(ckpt_dir,f"refer-youtube-vos_window-{window}.pth.tar")
    if not os.path.exists(ckpt_path):
        import gdown
        gdown.download(
          id="1R_F0ETKipENiJUnVwarHnkPmUIcKXRaL",
          output=ckpt_path, quiet=True)
    sd=torch.load(ckpt_path, map_location="cpu")
    sd=sd.get("model_state_dict",sd)
    model.load_state_dict(sd, strict=False)
    return model.to(device), post

# ────────────────────────────────────────────────────────────────────────────────
# 4) MTTR INFERENCE
# ────────────────────────────────────────────────────────────────────────────────
import torchvision
import torchvision.transforms.functional as F
from einops import rearrange
import numpy as np
from PIL import Image
from moviepy import AudioFileClip, ImageSequenceClip
from tqdm import tqdm

class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors, self.mask = tensors, mask

def to_nested(frames: torch.Tensor):
    T,C,H,W = frames.shape
    batched = frames.unsqueeze(0).transpose(0,1)
    mask = torch.zeros((T,1,H,W), dtype=torch.bool, device=frames.device)
    return NestedTensor(batched, mask)

def apply_mask_np(img, mask, color, alpha=0.7):
    c = np.array(color)[None,None]/255.0
    m3=mask[...,None].astype(float)*alpha
    return img*(1-m3) + c*m3

def run_inference(clip_path, queries, model, post):
    video,_,meta = torchvision.io.read_video(clip_path, pts_unit="sec")
    vid=rearrange(video,'t h w c->t c h w').float().div(255).to(device)
    vid=F.resize(vid,size=360,max_size=640)
    vid=F.normalize(vid,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    T,C,H,W = vid.shape

    ws,wo = 12,3
    windows=[vid[i:i+ws] for i in range(0,T,ws-wo)]
    qs=[q.lower().strip() for q in queries]

    all_masks=[]
    for q in tqdm(qs,desc="queries"):
        buf=torch.zeros((T,1,H,W),device="cpu")
        for i,win in enumerate(tqdm(windows,desc="wins")):
            nt=to_nested(win)
            idx=torch.arange(nt.tensors.shape[0], device=device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                out=model(nt,idx,[q])
                pm=post(out,[{"resized_frame_size":(H,W),"original_frame_size":(H,W)}],(H,W))[0]["pred_masks"]
            start=i*(ws-wo)
            buf[start:start+ws]=pm.cpu()
        all_masks.append(buf.numpy())

    # overlay masks and write video
    colors=[(41,171,226),(237,30,121),(35,161,90),(255,148,59)]
    frames=rearrange(vid,'t c h w->t h w c').cpu().numpy()
    out=[]
    for f,masks in zip(frames, zip(*all_masks)):
        img=f.copy()
        for m,c in zip(masks,colors):
            img=apply_mask_np(img, m[0], c)
        out.append((img*255).astype(np.uint8))

    out_path="output_clip.mp4"
    clip_v=ImageSequenceClip(out,fps=meta["video_fps"])
    clip_v=clip_v.with_audio(AudioFileClip(clip_path))
    clip_v.write_videofile(out_path,audio=True,fps=meta["video_fps"])
    return out_path, all_masks

# ────────────────────────────────────────────────────────────────────────────────
# 5) FULL-PIPELINE ANALYSIS, METRICS & SUMMARY VIDEO
# ────────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import librosa
from moviepy import AudioFileClip, ImageSequenceClip, VideoFileClip

def full_analysis(video_path, masks, percent, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --- gather basic video info ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- 1) Optical-flow magnitudes ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mags = []
    for i in range(1, total_frames):
        ret, frm = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None,
                                            pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2,
                                            flags=0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append((i, mag.mean()))
        prev_gray = gray
    cap.release()

    idxs, vals = zip(*mags)
    # plot + save
    plt.figure(); plt.plot(idxs, vals)
    plt.title("Optical-Flow Magnitude"); plt.xlabel("Frame"); plt.ylabel("Mean |flow|")
    plt.savefig(os.path.join(out_dir, "motion_curve.png")); plt.close()

    plt.figure(); plt.hist(vals, bins=50)
    cutoff = sorted(vals, reverse=True)[int(len(vals) * percent / 100)]
    plt.axvline(cutoff, linestyle='--', label=f"Top {percent}% cutoff")
    plt.legend(); plt.title("Motion Distribution")
    plt.savefig(os.path.join(out_dir, "motion_hist.png")); plt.close()

    sorted_vals = np.sort(vals)
    pdf         = sorted_vals / float(sum(sorted_vals))
    cdf         = np.cumsum(pdf)
    plt.figure(); plt.plot(sorted_vals, 1 - cdf)
    plt.title("Motion CDF"); plt.xlabel("Magnitude"); plt.ylabel("Fraction ≥ Mag")
    plt.savefig(os.path.join(out_dir, "motion_cdf.png")); plt.close()

    # --- 2) Scene-change ---
    cap = cv2.VideoCapture(video_path)
    ret, frm = cap.read()
    prev_h = cv2.calcHist([frm], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten()
    scene_scores = []
    for _ in range(1, total_frames):
        ret, frm = cap.read()
        if not ret: break
        h = cv2.calcHist([frm], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten()
        d = cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA)
        scene_scores.append(d)
        prev_h = h
    cap.release()
    plt.figure(); plt.plot(scene_scores)
    plt.title("Scene-Change Score"); plt.xlabel("Frame"); plt.ylabel("Distance")
    plt.savefig(os.path.join(out_dir, "scene_change.png")); plt.close()
    print("Detected scene cuts:", sum(1 for s in scene_scores if s > 0.5))

    # --- 3) Mask coverage & object count ---
    stacked   = np.stack(masks, axis=0)   # [Q, T,1,H,W]
    combined  = stacked.max(axis=0)       # [T,1,H,W]
    coverage  = [m[0].sum() / (height * width) for m in combined]
    plt.figure(); plt.plot(coverage)
    plt.title("Mask Coverage"); plt.xlabel("Frame"); plt.ylabel("Fraction")
    plt.savefig(os.path.join(out_dir, "mask_coverage.png")); plt.close()
    avg_cov_full = np.mean(coverage)

    counts = [sum(m[0].any() for m in per_frame) for per_frame in zip(*masks)]
    plt.figure(); plt.plot(counts)
    plt.title("Objects Detected per Frame"); plt.xlabel("Frame"); plt.ylabel("Count")
    plt.savefig(os.path.join(out_dir, "object_count.png")); plt.close()

    # --- 4) Frame entropy ---
    cap = cv2.VideoCapture(video_path)
    entropies = []
    while True:
        ret, frm = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        entropies.append(shannon_entropy(gray))
    cap.release()
    plt.figure(); plt.plot(entropies)
    plt.title("Frame Shannon Entropy"); plt.xlabel("Frame"); plt.ylabel("Entropy")
    plt.savefig(os.path.join(out_dir, "frame_entropy.png")); plt.close()

    # --- 5) Audio onset ---
    audio_path = os.path.join(out_dir, "extracted_audio.wav")
    with AudioFileClip(video_path) as audio_clip:
        audio_clip.write_audiofile(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(env, sr=sr)
    plt.figure(); plt.plot(times, env)
    plt.title("Audio Onset Strength"); plt.xlabel("Time (s)"); plt.ylabel("Strength")
    plt.savefig(os.path.join(out_dir, "audio_envelope.png")); plt.close()

    # --- 6) Summarize metrics printout ---
    mean_full_motion    = np.mean(vals)
    top_n               = int(len(vals) * percent / 100)
    top_frames          = sorted(mags, key=lambda x: x[1], reverse=True)[:top_n]
    summary_vals        = [v for i,v in top_frames]
    mean_summary_motion = np.mean(summary_vals)
    summary_cov_vals    = [coverage[i] for i,_ in top_frames]
    mean_summary_cov    = np.mean(summary_cov_vals)
    sorted_ent          = sorted(list(enumerate(entropies)), key=lambda x: x[1], reverse=True)[:top_n]
    mean_summary_ent    = np.mean([e for _,e in sorted_ent])

    print(f"Avg motion (full):     {mean_full_motion:.3f}")
    print(f"Avg motion (summary):  {mean_summary_motion:.3f}  ({mean_summary_motion/mean_full_motion:.2f}×)")
    print(f"Avg coverage (full):   {avg_cov_full:.3f}")
    print(f"Avg coverage (summary):{mean_summary_cov:.3f}")
    print(f"Avg entropy (full):    {np.mean(entropies):.3f}")
    print(f"Avg entropy (summary): {mean_summary_ent:.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 7) Build & save the summary video
    frame_indices = sorted(i for i,_ in top_frames)
    # pull frames directly via moviepy
    vc = VideoFileClip(video_path)
    summary_frames = [vc.get_frame(idx / fps) for idx in frame_indices]
    # assemble
    summary_clip = ImageSequenceClip(summary_frames, fps=fps)
    # optionally add the original audio (trimmed to same duration)
    audio = AudioFileClip(video_path).subclipped(0, len(summary_frames)/fps)
    summary_clip = summary_clip.with_audio(audio)
    summary_path = os.path.join(out_dir, f"summary_{percent}p.mp4")
    summary_clip.write_videofile(summary_path, audio=True, fps=fps)
    vc.close()

    print("Summary video saved to", summary_path)
    # ──────────────────────────────────────────────────────────────────────────

def main():
    args=parse_args()
    clip=download_and_clip(args.video_url,args.start,args.end)
    model,post=load_mttr(window=12)
    out, masks = run_inference(clip,args.queries,model,post)
    full_analysis(out, masks, args.summary_percent, args.summary_dir)
    print("\nAll outputs saved in", args.summary_dir)

if __name__=="__main__":
    main()
