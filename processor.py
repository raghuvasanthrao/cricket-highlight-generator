# processor.py
import os
import json
import numpy as np
from moviepy import VideoFileClip
from ultralytics import YOLO
import sys



def process_video(
    video_path,
    output_dir,
    gap,
    pre_sec,
    post_sec,
    movement_thresh,
    audio_factor,
    progress_callback,
    log_callback
):
    def log(msg):
        if log_callback:
            log_callback(msg)

    

    def resource_path(rel_path):
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, rel_path)
        return rel_path

    

    # ---------- Load resources ----------
    log("Loading video...")
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)

    audio = clip.audio
    model = YOLO(resource_path("yolov8n.pt"))

    # ---------- Load batting zone ----------
    with open("batting_zone.json") as f:
        zone = json.load(f)

    x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]

    def is_in_batting_zone(cx, cy):
        return x1 <= cx <= x2 and y1 <= cy <= y2

    # ---------- Audio helpers ----------
    def audio_energy_at(t, window=0.15):
        start = max(0, t - window)
        end = min(clip.duration, t + window)
        snd = audio[start:end]
        samples = snd.to_soundarray(fps=22050)
        return np.mean(samples ** 2)

    def is_audio_spike(t):
        e_now = audio_energy_at(t)
        bg = []
        for dt in (-2, -1, 1, 2):
            tt = t + dt
            if 0 <= tt < clip.duration:
                bg.append(audio_energy_at(tt))
        if not bg:
            return False
        return e_now > (sum(bg) / len(bg)) * audio_factor

    # ---------- Detect shots ----------
    log("Detecting shots...")
    shot_times = []
    prev_center_y = None

    for t in range(duration):
        frame = clip.get_frame(t)
        h, w, _ = frame.shape

        results = model(frame)[0]

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if model.names[int(cls)] != "person":
                continue

            xA, yA, xB, yB = box.tolist()
            cx = (xA + xB) / 2
            cy = (yA + yB) / 2

            if not is_in_batting_zone(cx, cy):
                continue

            if prev_center_y is not None:
                movement = abs(cy - prev_center_y)

                if movement > movement_thresh and is_audio_spike(t):
                    shot_times.append(t)

            prev_center_y = cy
            break

        # ---------- Progress update ----------
        percent = int((t / duration) * 70)
        progress_callback(percent)

    # ---------- Merge shot times ----------
    log("Merging shots...")
    shot_times.sort()

    merged = []
    current = []

    for t in shot_times:
        if not current or t - current[-1] <= gap:
            current.append(t)
        else:
            merged.append(current)
            current = [t]
    if current:
        merged.append(current)

    # ---------- Trim & save clips ----------
    os.makedirs(output_dir, exist_ok=True)

    log("Saving clips...")
    for i, group in enumerate(merged):
        center = group[len(group) // 2]
        start = max(0, center - pre_sec)
        end = min(clip.duration, center + post_sec)

        out = clip[start:end]
        out_path = os.path.join(output_dir, f"shot_{i+1}.mp4")

        out.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            fps=30
        )

        progress_callback(70 + int(((i + 1) / len(merged)) * 30))

    progress_callback(100)
    log(f"✅ Done. {len(merged)} shots saved.")
