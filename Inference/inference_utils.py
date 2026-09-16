import cv2
import numpy as np
from sklearn.cluster import KMeans

def cluster_player_teams_from_crops(player_data, resize_dim=(64, 64)):
    
    if not player_data:
        return []

    feature_list = []
    valid_players = []

    
    for player in player_data:
        crop = player['crop']
  
        if crop is None or crop.size == 0:
            continue
        height, width, _ = crop.shape
        torso_crop = crop[0:int(height * 0.6), :] 
        
        if torso_crop.size == 0:
            continue
        resized = cv2.resize(torso_crop, resize_dim)
        hsv_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        flattened_features = hsv_resized.flatten()
        feature_list.append(flattened_features)
        valid_players.append(player)

    if not feature_list:
        return player_data
    feature_matrix = np.array(feature_list)

    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    for idx, player in enumerate(valid_players):
        player['team_cluster'] = int(cluster_labels[idx])

    return valid_players

def extract_player_crops(yolo_results, frame, player_class_id=1):
    
    player_data = []
    
    if yolo_results is None or yolo_results.boxes is None:
        return player_data
    boxes = yolo_results.boxes
    frame_h, frame_w = frame.shape[:2]
    for i in range(len(boxes)):
        if boxes.id is None:
            continue
        cls_id = int(boxes.cls[i].item())
        if cls_id != player_class_id:
            continue

        track_id = int(boxes.id[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            player_data.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'crop': crop
            })

    return player_data
def assign_teams_by_anchor(players_crop, resize_dim=(64, 64), anchors=None):
    if not players_crop:
        return [], anchors

    feature_list = []
    valid_players = []
    for player in players_crop:
        crop = player['crop']
        if crop is None or crop.size == 0:
            continue
        height, width, _ = crop.shape
        torso_crop = crop[0:int(height * 0.6), :] 
        
        if torso_crop.size == 0:
            continue

        resized = cv2.resize(torso_crop, resize_dim)
        hsv_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        flattened_features = hsv_resized.flatten()
        
        feature_list.append(flattened_features)
        valid_players.append(player)

    if not feature_list:
        return players_crop, anchors

    feature_matrix = np.array(feature_list)

    if anchors is None:
        print("Automatically initializing team color anchors from the first frame...")
        kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        anchors = kmeans.cluster_centers_ 
        
        for idx, player in enumerate(valid_players):
            player['team_cluster'] = int(cluster_labels[idx])
            
    else:
        for player_idx, feature_vector in enumerate(feature_matrix):
            dist_to_team_0 = np.linalg.norm(feature_vector - anchors[0])
            dist_to_team_1 = np.linalg.norm(feature_vector - anchors[1])
            assigned_team = 0 if dist_to_team_0 < dist_to_team_1 else 1
            valid_players[player_idx]['team_cluster'] = assigned_team

    return valid_players, anchors
