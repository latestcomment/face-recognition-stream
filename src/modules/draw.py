import cv2

def draw_face(frame, bbox, name, score, liveness_status, spoof_prob, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)

    # If spoof, set box color to red
    if liveness_status.lower() == 'spoof':
        color = (0, 0, 255)  # Red in BGR

    # Draw face rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # --- Top label: Name + score ---
    top_label = f"{name} ({score:.2f})"
    (text_w, text_h), baseline = cv2.getTextSize(top_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, top_label, (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Bottom label: Liveness + spoof_prob ---
    bottom_label = f"{liveness_status} ({spoof_prob:.2f})"
    (text_w2, text_h2), baseline2 = cv2.getTextSize(bottom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y2), (x1 + text_w2, y2 + text_h2 + baseline2), color, -1)
    cv2.putText(frame, bottom_label, (x1, y2 + text_h2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame