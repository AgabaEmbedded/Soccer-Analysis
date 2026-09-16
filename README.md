# Soccer Player Tracking & Jersey Identification

Computer vision pipeline that tracks players, referees, goalkeepers, and the ball in soccer match footage, clusters players into teams by shirt color, and reads jersey numbers off each player's back to give every track a stable, human-readable identity, including across occlusions and frames where the player temporarily leaves view.

## How it works

The pipeline runs in **three offline passes** over the video rather than trying to make every decision live, frame by frame. This matters because team color and jersey number are properties that don't change during a match, so instead of committing to a guess the instant a player appears, the system gathers evidence across the whole clip first and only resolves each player's final identity once it has seen everything.

```
Video ─▶ Pass 1: Detection, Tracking & Evidence Collection
              │  (YOLO detects players/ball/GK/ref, BoT-SORT tracks them,
              │   shirt-color clustering tags a team per frame,
              │   a second YOLO model reads jersey digits off the torso)
              ▼
        Pass 2: Global Identity Resolution
              │  (majority-vote each track's team and jersey number
              │   across its entire lifetime, resolve collisions)
              ▼
        Pass 3: Rendering
              │  (draw stable boxes + "Player <#> | Team <n>" labels)
              ▼
        Output video
```

### Pass 1 — Detection, Tracking & Evidence Collection
- A YOLO model (`Yolov26`) detects and tracks 4 classes every frame using BoT-SORT: `0` Ball, `1` Goalkeeper, `2` Player, `3` Referee.
- For each tracked player, the torso region of the crop is fed into a color-clustering step (`assign_teams_by_anchor`) that assigns a team (`0` or `1`) based on shirt color, using anchors seeded via k-means on early frames.
- The torso crop is also fed into a second YOLO model (`Yolov11`) trained specifically to detect individual jersey digits. Detected digits are sorted left-to-right by x-position and concatenated into a number (e.g. digit `1` then `0` → `"10"`).
- Every frame's raw detections (track id, class, team, bbox) are logged to `metadata["frames"]`.
- Jersey-digit evidence is accumulated per track as a confidence-weighted vote: `metadata["track_votes"][track_id][digit_string] += confidence`, so a single misread frame can't override many consistent reads.

### Pass 2 — Global Identity Resolution
- For each track, the **full-history mode** of its per-frame team reads is taken as the resolved team — a much stronger signal than trusting any single frame, since a player's team never actually changes mid-match.
- For each track, the jersey number with the highest accumulated vote weight is selected as its resolved number.
- Within each team, if two tracks resolve to the same jersey number, the one with more accumulated evidence wins it; the other is flagged `"Conflict"`. Tracks with too little accumulated evidence are flagged `"Unconfirmed"`.

### Pass 3 — Rendering
- Re-reads the source video frame by frame and draws each detection's box and label (`Player <jersey> | Team <n>`, or `Ball` / `Goalkeeper` / `Referee` with track id) using the *resolved* identities from Pass 2, so labels are stable and consistent for the track's entire time on screen.



---
## Demo Before and After Video

`"Befor"`: **[SNMOT-066.mp4 →](https://drive.google.com/file/d/1ZHZVy-bEXTZiV5JtFMaRMpegNw051bhO/view?usp=drive_link)**

`"After"`: **[SNMOT-066-Analyzed.mp4 →](https://drive.google.com/file/d/1O2LdlM1BMon9Kig5TanoygTRV3BZ1_Fk/view?usp=drive_link)**

---

## Accessing the Notebook

Run the pipeline in Google Colab:

**[Soccer Analysis Inference notebook →](https://colab.research.google.com/drive/15RSF_mQX6mfGt11SKi-J-yPZ6JINtHP7?usp=sharing)**

Modify the inference notebook to suit your taste, contributions are welcomed

---
## Models

| Model | Task | Dataset | Metric | Score |
| --- | --- | --- | --- | --- |
| **YOLOv26** | Player & Ball Detection | SoccerNet | mAP@50 | 98.6% |
| **YOLOv11** | Jersey Digit Detection | Custom Annotated Dataset | mAP@50 | 88.5% |

---



## Dataset

Training/evaluation footage comes from [SoccerNet](https://www.soccer-net.org/)'s tracking task, downloaded via the `SoccerNet` pip package:
```python
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="SoccerNet")
downloader.downloadDataTask(task="tracking", split=["train"])
```
SoccerNet provides the data as sequences of extracted JPEG frames (e.g. `SNMOT-061/img1/`) and are stitched into `.mp4` clips with `frames_to_video()` before being run through the pipeline for inference.


## Known issues / in-progress fixes

- **Team assignment flicker.** The original per-frame color-clustering feature (flattened raw HSV pixels) is sensitive to pose, alignment, and background bleed, causing a track's team read to flip frame to frame. Mitigations in progress: (1) a position-invariant HSV color-histogram feature in place of flattened pixels, and (2) resolving each track's team by full-history majority vote in Pass 2 rather than trusting any single frame.
- **Jersey-vote fragmentation.** Evidence used to be keyed as `team_track_votes[team_id][track_id]`, so a flickering team read split one player's jersey-digit evidence across two buckets, weakening both. Fixed by keying evidence purely by `track_id` (`track_votes[track_id]`), independent of team, and resolving team separately in Pass 2.
---