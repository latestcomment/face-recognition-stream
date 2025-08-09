import cv2

def draw_face(frame, bbox, name, score, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Prepare label text
    label = f"{name} ({score:.2f})"

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Draw background rectangle for text
    cv2.rectangle(frame, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)

    # Put label text
    cv2.putText(frame, label, (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame