"""
Assignment 0 – Play a video with:
  1. Random text overlaid at random positions (changes every ~30 frames / 1 second)
  2. A scrolling ribbon at the bottom: "python assignment, version 0"
  Also saves the output video to assignment0_output.mp4
"""

import cv2
import random
import string
import numpy as np

# ── Settings ──────────────────────────────────────────────────────────────────
VIDEO_PATH  = "Video for assignment.mp4"
OUTPUT_PATH = "assignment0_output.mp4"
RIBBON_TEXT = "python assignment, version 0"
RANDOM_TEXT_INTERVAL = 30  # frames between new random text placement

# ── Random-text helpers ───────────────────────────────────────────────────────
def random_string(length=12):
    """Return a random alphanumeric string."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))

def random_color():
    """Return a random BGR colour tuple."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# ── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
delay  = int(1000 / fps) if fps > 0 else 33  # ms between frames

# ── Video writer setup ────────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ── Ribbon setup ──────────────────────────────────────────────────────────────
ribbon_height = 40
font       = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness  = 2

# Measure ribbon text for marquee
(text_w, text_h), _ = cv2.getTextSize(RIBBON_TEXT, font, font_scale, thickness)
gap = 120  # pixel gap between repeated copies
scroll_cycle = text_w + gap  # total length of one copy + gap
ribbon_offset = 0  # starts at 0 so text is visible immediately
scroll_speed = 3   # pixels per frame
ribbon_y_pos = height - (ribbon_height - text_h) // 2

# ── Random-text state ─────────────────────────────────────────────────────────
rand_text  = random_string()
rand_x     = random.randint(10, max(width - 200, 10))
rand_y     = random.randint(40, max(height - ribbon_height - 40, 50))
rand_color = random_color()
frame_idx  = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
saved = False  # tracks whether we've finished saving one full pass
print("Playing video … press 'q' to quit.")
print(f"Saving overlaid video to: {OUTPUT_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        if not saved:
            saved = True
            out.release()
            print(f"Video saved to {OUTPUT_PATH}")
        # Loop the video for playback
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    # 1) Random text – refreshed every RANDOM_TEXT_INTERVAL frames
    if frame_idx % RANDOM_TEXT_INTERVAL == 0:
        rand_text  = random_string(random.randint(6, 18))
        rand_x     = random.randint(10, max(width - 300, 10))
        rand_y     = random.randint(40, max(height - ribbon_height - 60, 50))
        rand_color = random_color()

    # Draw random text with a dark outline for readability
    cv2.putText(frame, rand_text, (rand_x, rand_y), font, 0.9,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, rand_text, (rand_x, rand_y), font, 0.9,
                rand_color, thickness, cv2.LINE_AA)

    # 2) Bottom ribbon – dark semi-transparent bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - ribbon_height),
                  (width, height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Scrolling marquee – draw enough copies to fill the screen
    base_x = -ribbon_offset
    i = 0
    while base_x + i * scroll_cycle < width:
        x = base_x + i * scroll_cycle
        if x + text_w > 0:  # only draw if on screen
            cv2.putText(frame, RIBBON_TEXT, (int(x), ribbon_y_pos), font,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        i += 1
    ribbon_offset = (ribbon_offset + scroll_speed) % scroll_cycle

    # Save frame (only first pass)
    if not saved:
        out.write(frame)

    # Display
    cv2.imshow("Assignment 0", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
if not saved:
    out.release()
cv2.destroyAllWindows()
print("Done.")
