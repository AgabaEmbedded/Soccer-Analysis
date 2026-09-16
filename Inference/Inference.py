import cv2
import numpy as np
from sklearn.cluster import KMeans
from paddleocr import PaddleOCR
from paddleocr import TextRecognition




botsortyaml = """# custom_botsort.yaml
tracker_type: botsort       # Specifies BoT-SORT tracking logic
track_high_thresh: 0.3      # Threshold for first association step
track_low_thresh: 0.1       # Threshold for second association step
new_track_thresh: 0.4       # Threshold for initializing a brand new track
track_buffer: 30            # Number of frames to keep a lost track in memory
match_thresh: 0.8           # IoU threshold for matching tracks

# --- CRITICAL: GLOBAL MOTION COMPENSATION (GMC) SETUP ---
gmc_method: sparseOptFlow   # Fixed accept values: sparseOptFlow, orb, sift, ecc, none

# --- RE-ID EMBEDDING SETUP ---
proximity_thresh: 0.5       # Spatial distance gate
appearance_thresh: 0.95     # High threshold to prevent identical jersey swaps
with_reid: False            # Disables built-in Re-ID embeddings

# --- ULTRALYTICS ENGINE SAFETY ANCHORS (Fixes the AttributeError) ---
model: null                 # Tells BoT-SORT no external Re-ID weight file is being supplied
fuse_score: True            # Intersects confidence metrics with spatial proximity"""

with open("./botsort.yaml", "w") as file:
    file.write(botsortyaml)





#ocr = PaddleOCR(use_angle_cls=False, lang="en",)#, det=False)#, show_log=False)
ocr = TextRecognition(engine="paddle",)

TEAM_ANCHORS = None
TRACK_IDENTITY_VOTES = {}

CLASS_COLORS = {0: (0, 0, 255), 1: (255, 255, 0), 3: (0, 255, 255)}
TEAM_COLORS = {0: (255, 0, 0), 1: (0, 255, 0)}

results = model.track("/content/drive/MyDrive/test/test_video_116.mp4", 
                      save=False, 
                      imgsz=[1920, 1088], 
                      conf=0.25, 
                      iou=0.45, 
                      batch=32, 
                      stream=True, 
                      verbose=False,
                      tracker="/content/botsort.yaml")

video_writer = None  

for result in results:
    result = result.cpu()
    if result.boxes is None or len(result.boxes) == 0:
        continue
        
    annotated_frame = result.orig_img.copy()
    
    if video_writer is None:
        h, w = annotated_frame.shape[:2]
        video_writer = cv2.VideoWriter('/content/output_tracked_match.mp4', 
                                       cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))

    players_crop = extract_player_crops(result, result.orig_img, player_class_id=2)
    
    team_map = {}
    if players_crop:
        players_crop, TEAM_ANCHORS = assign_teams_by_anchor(players_crop, resize_dim=(64, 64), anchors=TEAM_ANCHORS)
        
        for player in players_crop:
            track_id = player['id']
            crop_img = player['crop']
            team_map[track_id] = player.get('team_cluster', None)
            
            h_crop, w_crop, _ = crop_img.shape
            back_region = crop_img[int(h_crop * 0.15):int(h_crop * 0.55), :]
            
            if back_region.size > 0:
                ocr_result = ocr.predict(back_region)

                # 1. Ensure the output list is not empty and the first prediction element is valid
                if ocr_result and len(ocr_result) > 0:
                    res = ocr_result[0]
                    
                    # Safely handle different PaddleX / TextRecognition version payload schemas
                    detected_text = res.get('rec_text', '') if isinstance(res, dict) else getattr(res, 'rec_text', '')
                    confidence = res.get('rec_score', 0.0) if isinstance(res, dict) else getattr(res, 'rec_score', 0.0)
                    
                    if detected_text:
                        # 2. Extract digits instead of dropping the whole read due to minor wrinkles/spaces
                        digit_string = "".join([char for char in str(detected_text) if char.isdigit()])
                        
                        # 3. Enforce confidence constraints on your cleaned string
                        if digit_string and confidence > 0.65:
                            if track_id not in TRACK_IDENTITY_VOTES:
                                TRACK_IDENTITY_VOTES[track_id] = {}
                            
                            TRACK_IDENTITY_VOTES[track_id][digit_string] = TRACK_IDENTITY_VOTES[track_id].get(digit_string, 0) + 1

    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].numpy().astype(int)
        
        if cls_id == 2:  
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            team_id = team_map.get(track_id, None)
            color = TEAM_COLORS.get(team_id, (255, 255, 255))
            
            
            resolved_jersey = "Unconfirmed"
            if track_id in TRACK_IDENTITY_VOTES and TRACK_IDENTITY_VOTES[track_id]:
                
                resolved_jersey = max(TRACK_IDENTITY_VOTES[track_id], key=TRACK_IDENTITY_VOTES[track_id].get)
            
            label = f"Player {resolved_jersey} | Team {team_id if team_id is not None else 'N/A'}"
            
        else:            
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label_names = {0: "Ball", 1: "Goalkeeper", 3: "Referee"}
            track_str = f" ID: {int(boxes.id[i].item())}" if boxes.id is not None else ""
            label = f"{label_names.get(cls_id, 'Unknown')}{track_str}"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(annotated_frame, (x1, y1_label - label_size[1] - 5), (x1 + label_size[0], y1_label + base_line), color, cv2.FILLED)
        cv2.putText(annotated_frame, label, (x1, y1_label - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    video_writer.write(annotated_frame)

if video_writer is not None:
    video_writer.release()
    print("🚀 Tracking video complete with Team and OCR resolution configurations!")