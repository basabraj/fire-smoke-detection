import os, re, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"D:\Fire & Smoke\Home\dataset\fire_smoke"


PATTERN_NEW = re.compile(r"^(.*)_f\d+_t\d+\.\d+\.jpg$")  # "{id}_f000000_t0000.00.jpg"
PATTERN_OLD = re.compile(r"^(.*)_t\d+\.\d+\.jpg$")        # "{id}_t0000.00.jpg"


def match_identifier(fname):
    m = PATTERN_NEW.match(fname)
    if m:
        return m.group(1)
    m = PATTERN_OLD.match(fname)
    if m:
        return m.group(1)
    return None

for category in ["no_fire", "no_smoke", "fire", "smoke"]:
    videos_dir = os.path.join(BASE, category, "videos")
    frames_dir = os.path.join(BASE, category, "frames")

    print(f"\n=== {category} ===")

    if not os.path.isdir(videos_dir):
        print("  (no videos/ folder)")
        continue

    videos = [f for f in os.listdir(videos_dir) if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]

    if not os.path.isdir(frames_dir):
        print(f"  {len(videos)} video(s), but NO frames/ folder exists")
        continue

    frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]

    # Recover each frame's source-video identifier
    covered = set()
    for fname in frame_files:
        identifier = match_identifier(fname)
        if identifier:
            covered.add(identifier)

    print(f"  {len(videos)} video(s) in videos/, {len(frame_files)} frame(s) in frames/")

    missing = []
    for v in videos:
        identifier = os.path.splitext(v)[0].replace(" ", "_")
        if identifier not in covered:
            missing.append(v)

    if missing:
        print(f"  NOT YET CONVERTED ({len(missing)}):")
        for v in missing:
            print(f"    - {v}")
    else:
        print("  All videos have at least some frames extracted.")

    
    video_identifiers = {os.path.splitext(v)[0].replace(" ", "_") for v in videos}
    orphaned = covered - video_identifiers
    if orphaned:
        print(f"  Frames present for {len(orphaned)} video(s) no longer in videos/ (already deleted source):")
        for o in sorted(orphaned):
            print(f"    - {o}")
