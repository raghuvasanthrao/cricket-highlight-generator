import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, StringVar, IntVar, DoubleVar
import threading
from processor import process_video
import cv2
import json
from moviepy import VideoFileClip
import os


#pyinstaller --onefile --windowed --add-data "yolov8n.pt;." --collect-all=imageio --collect-all=moviepy --collect-all=numpy ui.py



# ---------------- APP WINDOW ----------------
app = tb.Window(themename="darkly")
app.title("Cricket Shot Detector")
app.geometry("900x650")
app.minsize(800, 600)


# -------- SCROLLABLE ROOT --------
container = tb.Frame(app)
container.pack(fill=BOTH, expand=True)

canvas = tb.Canvas(container, highlightthickness=0)
scrollbar = tb.Scrollbar(container, orient=VERTICAL, command=canvas.yview)

scrollable = tb.Frame(canvas)

scrollable.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)


# ---------------- STATE VARIABLES ----------------
video_path = StringVar(value="No video selected")

gap_var = IntVar(value=2)
pre_sec_var = IntVar(value=2)     # center - X
post_sec_var = IntVar(value=2)    # center + X
movement_var = IntVar(value=30)
audio_factor_var = DoubleVar(value=1.8)
output_dir = StringVar(value="No output directory selected")

progress_var = IntVar(value=0)
is_processing = False


# ---------------- HELPERS ----------------
def choose_video():
    path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.mkv *.avi")]
    )
    if path:
        video_path.set(path)

def sync_slider_to_entry(slider, var):
    slider.set(var.get())

def sync_entry_to_slider(var, slider):
    var.set(slider.get())

def select_batting_zone():
    if video_path.get() == "No video selected":
        log("❌ Please select a video before choosing batting zone")
        return

    log("Opening batting zone selector...")

    clip = VideoFileClip(video_path.get())

    # take a frame where batsman is visible
    frame = clip.get_frame(10)
    orig_h, orig_w, _ = frame.shape

    # ---- resize for screen ----
    display_width = 900
    scale = display_width / orig_w
    display_height = int(orig_h * scale)

    frame_small = cv2.resize(frame, (display_width, display_height))
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)

    roi = cv2.selectROI(
        "Draw Batting Zone (Press ENTER)",
        frame_small,
        showCrosshair=True
    )
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        log("❌ Batting zone selection cancelled")
        return

    x, y, w, h = roi

    # ---- map back to original resolution ----
    zone = {
        "x1": int(x / scale),
        "y1": int(y / scale),
        "x2": int((x + w) / scale),
        "y2": int((y + h) / scale)
    }

    with open("batting_zone.json", "w") as f:
        json.dump(zone, f)

    zone_status.set("Batting zone: SELECTED ✔")
    log(f"Batting zone saved: {zone}")

def choose_output_dir():
    path = filedialog.askdirectory()
    if path:
        output_dir.set(path)
        log(f"Output directory set to: {path}")

def make_scrollable(parent):
    container = tb.Frame(parent)
    canvas = tb.Canvas(container, highlightthickness=0)
    scrollbar = tb.Scrollbar(container, orient=VERTICAL, command=canvas.yview)

    scrollable_frame = tb.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    container.pack(fill=BOTH, expand=True)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)

    return scrollable_frame

def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)


# ---------------- UI LAYOUT ----------------



# -------- Video Selection --------
frame_video = tb.Frame(scrollable)
frame_video.pack(fill=X, padx=20, pady=10)

tb.Button(
    frame_video,
    text="Select Video",
    bootstyle=PRIMARY,
    command=choose_video
).pack(side=LEFT)

tb.Label(
    frame_video,
    textvariable=video_path,
    wraplength=550
).pack(side=LEFT, padx=10)

zone_status = StringVar(value="Batting zone: NOT selected")

tb.Label(
    scrollable,
    textvariable=zone_status,
    bootstyle=INFO
).pack(pady=5)

# -------- Output Directory Selection --------
frame_output = tb.Frame(scrollable)
frame_output.pack(fill=X, padx=20, pady=10)

tb.Button(
    frame_output,
    text="Select Output Folder",
    bootstyle=SECONDARY,
    command=choose_output_dir
).pack(side=LEFT)

tb.Label(
    frame_output,
    textvariable=output_dir,
    wraplength=550
).pack(side=LEFT, padx=10)

# ---------------- PARAMETER PANEL ----------------
param_container = tb.LabelFrame(scrollable, text="Detection Parameters")
param_container.pack(fill=BOTH, expand=True, padx=15, pady=10)

param_frame = tb.LabelFrame(
    scrollable,
    text="Detection Parameters"
)
param_frame.pack(fill=X, padx=15, pady=10)




log_box = tb.Text(scrollable, height=8)
log_box.pack(fill=X, padx=20, pady=10)

progress_bar = tb.Progressbar(
    scrollable,
    variable=progress_var,
    maximum=100,
    bootstyle=SUCCESS,
    length=600
)
progress_bar.pack(padx=20, pady=10, fill=X)

def progress_callback(percent):
    app.after(0, lambda: progress_var.set(percent))

def on_processing_done():
    global is_processing
    is_processing = False
    enable_ui()
    progress_var.set(100)
    log("✅ Processing completed")

def log(msg):
    log_box.insert(END, msg + "\n")
    log_box.see(END)

def start_processing():
    global is_processing

    if is_processing:
        return

    if video_path.get() == "No video selected":
        log("❌ Please select a video first")
        return

    if output_dir.get() == "No output directory selected":
        log("❌ Please select an output directory")
        return

    if not os.path.exists("batting_zone.json"):
        log("❌ Please select batting zone first")
        return

    is_processing = True
    progress_var.set(0)
    disable_ui()
    log("▶ Processing started...")

    def run():
        try:
            process_video(
                video_path.get(),
                output_dir.get(),
                gap_var.get(),
                pre_sec_var.get(),
                post_sec_var.get(),
                movement_var.get(),
                audio_factor_var.get(),
                progress_callback,
                log
            )
        finally:
            app.after(0, on_processing_done)

    threading.Thread(target=run, daemon=True).start()



# ---------- Helper function for slider + entry ----------
def slider_row(parent, label, from_, to_, var, step=1):
    row = tb.Frame(parent)
    row.pack(fill=X, pady=5)

    tb.Label(row, text=label, width=25).pack(side=LEFT)

    slider = tb.Scale(
        row,
        from_=from_,
        to=to_,
        variable=var,
        orient=HORIZONTAL,
        length=300
    )
    slider.pack(side=LEFT, padx=10)

    entry = tb.Entry(row, textvariable=var, width=8)
    entry.pack(side=LEFT)

    return slider, entry

def disable_ui():
    btn_start.config(state=DISABLED)
    btn_zone.config(state=DISABLED)

def enable_ui():
    btn_start.config(state=NORMAL)
    btn_zone.config(state=NORMAL)

# -------- GAP (merge seconds) --------
slider_row(
    param_frame,
    "Merge gap (seconds)",
    0, 5,
    gap_var
)

# -------- PRE seconds --------
slider_row(
    param_frame,
    "Trim before shot (sec)",
    0, 5,
    pre_sec_var
)

# -------- POST seconds --------
slider_row(
    param_frame,
    "Trim after shot (sec)",
    0, 5,
    post_sec_var
)

# -------- MOVEMENT threshold --------
slider_row(
    param_frame,
    "Movement threshold (px)",
    10, 100,
    movement_var
)

# -------- AUDIO factor --------

slider_row(
    param_frame,
    "Audio spike factor",
    1.0, 3.0,
    audio_factor_var
)
# ---------------- ACTION BUTTONS ----------------
action_frame = tb.Frame(scrollable)
action_frame.pack(pady=20)

btn_zone = tb.Button(
    action_frame,
    text="Select Batting Zone",
    bootstyle=WARNING,
    command=select_batting_zone
)
btn_zone.pack(side=LEFT, padx=10)

btn_start = tb.Button(
    action_frame,
    text="Start Processing",
    bootstyle=SUCCESS,
    command=start_processing
)
btn_start.pack(side=LEFT, padx=10)



tb.Button(
    action_frame,
    text="Exit",
    bootstyle=DANGER,
    command=app.destroy
).pack(side=LEFT, padx=10)

# ---------------- DEBUG / PREVIEW ----------------
def print_config():
    print("Video:", video_path.get())
    print("gap:", gap_var.get())
    print("pre_sec:", pre_sec_var.get())
    print("post_sec:", post_sec_var.get())
    print("movement:", movement_var.get())
    print("audio factor:", audio_factor_var.get())

tb.Button(
    scrollable,
    text="Print Config (Debug)",
    command=print_config
).pack(pady=5)

app.mainloop()


