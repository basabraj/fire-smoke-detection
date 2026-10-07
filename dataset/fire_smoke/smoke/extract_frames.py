"""
extract_frames.py — convert every video in videos/ into frames saved under
frames/, sampling 1 frame per second (matches the seconds_per_frame=1.0
convention already used by Footage/extract_frames.py in this codebase).

Uses grab()-for-skip / read()-for-keep instead of read()-on-every-frame: a
full decode is far more expensive than a grab(), and at 1 frame/sec on a
30fps video that's the difference between decoding 30 frames and decoding 1
for every second of footage (see the earlier fix to Footage/extract_frames.py
for the measured ~5x speedup this gives).

Run:  python extract_frames.py
"""
import os
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(HERE, "videos")
FRAMES_DIR = os.path.join(HERE, "frames")
SECONDS_PER_FRAME = 1.0
IMAGE_FORMAT = "jpg"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")


def extract_one(video_path, video_identifier):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"  [SKIP] could not open: {video_path}")
        return 0

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 1:
        source_fps = 15.0
    frame_interval = max(1, round(source_fps * SECONDS_PER_FRAME))

    frame_index = 0
    saved_count = 0
    failed_count = 0

    while True:
        if frame_index % frame_interval == 0:
            success, frame = capture.read()  # frame we're keeping — full decode
            if not success:
                break
            timestamp_seconds = frame_index / source_fps
            # frame_index (not just the timestamp) is in the filename so two
            # saved frames can never collide on the same name — on a
            # variable-frame-rate video, source_fps can be off enough that
            # two different frame_index values round to the same 2-decimal
            # timestamp, and one silently overwrites the other (this is
            # exactly what shrank the first run's 2190 reported saves down
            # to 1575 files actually on disk).
            filename = f"{video_identifier}_f{frame_index:06d}_t{timestamp_seconds:07.2f}.{IMAGE_FORMAT}"
            ok = cv2.imwrite(os.path.join(FRAMES_DIR, filename), frame)
            if ok:
                saved_count += 1
            else:
                failed_count += 1  # imwrite() return value was never checked before — silent failures counted as saved
        else:
            success = capture.grab()  # frame we're skipping — cheap, no decode
            if not success:
                break
        frame_index += 1

    capture.release()
    if failed_count:
        print(f"  [WARN] {failed_count} frame(s) failed to write")
    return saved_count


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    videos = sorted(
        f for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    if not videos:
        print(f"No video files found in: {VIDEOS_DIR}")
        return

    print(f"Found {len(videos)} videos. Extracting 1 frame/sec into {FRAMES_DIR}\n")

    total_saved = 0
    for i, filename in enumerate(videos, 1):
        video_path = os.path.join(VIDEOS_DIR, filename)
        video_identifier = os.path.splitext(filename)[0].replace(" ", "_")
        print(f"[{i}/{len(videos)}] {filename}")
        saved = extract_one(video_path, video_identifier)
        total_saved += saved
        print(f"  -> {saved} frames saved")

    print(f"\nDone. {total_saved} frames total saved to {FRAMES_DIR}")


if __name__ == "__main__":
    main()
