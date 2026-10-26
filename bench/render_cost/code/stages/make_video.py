#!/usr/bin/env python
"""Five variants side by side, the timed pass's own frames (pose order), 30 fps. make_video.py <tag> <traj>"""
import subprocess, sys
import numpy as np
tag, traj = sys.argv[1], sys.argv[2]
d = f"/tmp/irsim_render_cost/stages/out/{tag}/{traj}/"
seqs = [np.load(d + f"display_seq_{v}.npy", mmap_mode="r") for v in ("12", "123", "1236", "12356", "123456")]
n, h, w, _ = seqs[0].shape
out = f"/tmp/irsim_render_cost/stages/out/{tag}/five_up_{traj}.mp4"
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{5 * w}x{h}", "-r", "30",
                      "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
for j in range(n):
    p.stdin.write(np.concatenate([np.asarray(s[j]) for s in seqs], axis=1).tobytes())
p.stdin.close(); print(out, p.wait())
