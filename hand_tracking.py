#===================
# README... FOR NOW
#===================
# hand gestures you can do: 
# 1. peace sign
# 2. middle finger
# 3. closed fist
# 4. open palm: if you close your fist and turn, then it changes the color that you draw with
# 5. hang loose
# 6. thumbs down
# 7. rock and roll
# 8. i love you
# 9. ok
# 10. pinch
# 11. pointer: used to draw
# 12. tight (or the promise sign): used as an eraser for when you draw
# 13. "i" in ASL erases the whole drawing


# import modules and make it easier to refer to submodules 
import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# CYBERNETIC THEME COLORS
CYBER_CYAN = (255, 255, 0)
CYBER_MAGENTA = (255, 0, 255)
CYBER_BLUE = (255, 100, 0)
CYBER_DARK = (20, 20, 20)
CYBER_GRID = (80, 80, 40)

#===================
# GESTURE FUNCTIONS
#===================
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def what_finger_be_doing(hand_landmarks):
    fingers = {}
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    wrist = hand_landmarks.landmark[0]
    is_right_hand = thumb_tip.x < wrist.x

    if is_right_hand:
        fingers["thumb"] = thumb_tip.x < thumb_ip.x
    else:
        fingers["thumb"] = thumb_tip.x > thumb_ip.x
        
    fingers["index"] = hand_landmarks.landmark[8].y  < hand_landmarks.landmark[6].y
    fingers["middle"] = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    fingers["ring"] = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    fingers["pinky"] = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    return fingers


def curled_fingers(tip, pip, mcp):
    dist_tip_mcp = distance(tip, mcp)
    dist_tip_pip = distance(tip, pip)
    return dist_tip_mcp < dist_tip_pip


def classify_gesture(fingers, hand_landmarks):
    # landmark showing where each fingertip and wrist is
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip  = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # distance for certain gestures
    pinch_dist = distance(thumb_tip, index_tip)
    index_middle_dist = distance(index_tip, middle_tip)

    # actual fingers if they are extended
    pointer = fingers["index"]
    middle  = fingers["middle"]
    ring    = fingers["ring"]
    pinky   = fingers["pinky"]
    thumb   = fingers["thumb"]

    # curled fingers
    curled_index  = curled_fingers(index_tip, hand_landmarks.landmark[6], hand_landmarks.landmark[5])
    curled_middle = curled_fingers(middle_tip, hand_landmarks.landmark[10], hand_landmarks.landmark[9])
    curled_ring   = curled_fingers(ring_tip, hand_landmarks.landmark[14], hand_landmarks.landmark[13])
    curled_pinky  = curled_fingers(pinky_tip, hand_landmarks.landmark[18], hand_landmarks.landmark[17])

    #================================================
    # actual gestures
    #================================================
    
    # "i" or complete eraser
    if not pointer and not middle and not ring and pinky and not thumb:
        return "i in ASL"
    
    # hang loooosseee
    if thumb and pinky and curled_index and curled_middle and curled_ring:
        return "Waaazzzaaapppp"
    
    # thumbs down
    if thumb and curled_index and curled_middle and curled_ring and curled_pinky:
        if thumb_tip.y > wrist.y:
            return "Thumbs down. Nuh uh!"

    # rock and rollll || <3
    if pointer and not middle and not ring and pinky:
        if thumb:
            return "I love you!"
        else:
            return "Rock and Rollllll!!!!!"

    # ok!
    thumb_tip  = hand_landmarks.landmark[4]
    index_tip  = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip   = hand_landmarks.landmark[16]
    pinky_tip  = hand_landmarks.landmark[20]

    pinch_dist = distance(thumb_tip, index_tip)
    other_fingers = middle and ring and pinky

    if pinch_dist < 0.03 and other_fingers:
        return "OK"

    # closed fist
    curled_index  = curled_fingers(index_tip, hand_landmarks.landmark[6], hand_landmarks.landmark[5])
    curled_middle = curled_fingers(middle_tip, hand_landmarks.landmark[10], hand_landmarks.landmark[9])
    curled_ring   = curled_fingers(ring_tip, hand_landmarks.landmark[14], hand_landmarks.landmark[13])
    curled_pinky  = curled_fingers(pinky_tip, hand_landmarks.landmark[18], hand_landmarks.landmark[17])
    curled_thumb  = distance(thumb_tip, hand_landmarks.landmark[2]) < distance(thumb_tip, hand_landmarks.landmark[1])

    if curled_index and curled_middle and curled_ring and curled_pinky and curled_thumb:
        return "Closed Fist"

    # pinch
    curled_index = curled_fingers(index_tip, hand_landmarks.landmark[6], hand_landmarks.landmark[5])
    if pinch_dist < 0.035 and not (curled_index and other_fingers):
        return "Pinch"

    # eraser (or 2) || peace
    if pointer and middle and not ring and not pinky:
        if index_middle_dist < 0.04:
            return "Eraser"
        else:
            return "Peaceee!!!!"

    # pointer
    if pointer and not middle and not ring and not pinky:
        return "Pointing/Drawing"

    # middle finger
    if middle and not pointer and not ring and not pinky:
        return "Middle finger"

    # open palm
    if pointer and middle and ring and pinky:
        return "Open Palm"

    return None


def hand_roll_angle(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]
    pinky_mcp = hand_landmarks.landmark[17]

    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y
    angle = math.degrees(math.atan2(dy, dx))

    if angle > 180:
        angle -= 360
    if angle < -180:
        angle += 360

    return angle

def shift_canvas(canvas, dx, dy):
    h, w, _ = canvas.shape
    new_canvas = np.zeros_like(canvas)

    x1 = max(0, dx)
    x2 = min(w, w + dx)
    y1 = max(0, dy)
    y2 = min(h, h + dy)

    src_x1 = max(0, -dx)
    src_x2 = src_x1 + (x2 - x1)
    src_y1 = max(0, -dy)
    src_y2 = src_y1 + (y2 - y1)

    new_canvas[y1:y2, x1:x2] = canvas[src_y1:src_y2, src_x1:src_x2]
    return new_canvas

def draw_cyber_grid(frame, grid_size=40, frame_count=0):
    h, w = frame.shape[:2]
    offset = int(frame_count * 0.5) % grid_size
    
    # vertical lines
    for x in range(-offset, w, grid_size):
        alpha = 0.3 if (x // grid_size) % 2 == 0 else 0.15
        color = tuple(int(c * alpha) for c in CYBER_GRID)
        cv2.line(frame, (x, 0), (x, h), color, 1)
    
    # horizontal lines
    for y in range(-offset, h, grid_size):
        alpha = 0.3 if (y // grid_size) % 2 == 0 else 0.15
        color = tuple(int(c * alpha) for c in CYBER_GRID)
        cv2.line(frame, (0, y), (w, y), color, 1)

def draw_scanlines(frame, spacing=4):
    h, w = frame.shape[:2]
    for y in range(0, h, spacing):
        frame[y:y+1] = frame[y:y+1] * 0.9

def draw_glitch_box(frame, x1, y1, x2, y2, color, thickness=2):
    corner_len = 20
    
    # top left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    
    # top right
    cv2.line(frame, (x2 - corner_len, y1), (x2, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    
    # bottom left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2 - corner_len), (x1, y2), color, thickness)
    
    # bottom right
    cv2.line(frame, (x2 - corner_len, y2), (x2, y2), color, thickness)
    cv2.line(frame, (x2, y2 - corner_len), (x2, y2), color, thickness)

#===================
# TRACKING VIDEO  
#===================
def run_tracking():
    cam = cv2.VideoCapture(0)
    canvas = None
    dragging = False
    prev_fist_pos = None
    prev_point = None
    frame_count = 0

    # CYBERPUNK COLOR PALETTE
    colors = [
        (255, 255, 255),
        (0, 0, 255),
        (0, 165, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (255, 0, 128)]

    color_index = 0
    current_color = colors[0]

    # init MediaPipe 
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        # open cam
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                continue

            frame_count += 1

            # create canvas for drawing
            if canvas is None:
                canvas = np.zeros_like(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # change to rbg
            hand_results = hands.process(frame_rgb) # detect hands

            gesture_text = ""
            draw_point = None
            # detect each hand
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:

                    # draw hands with cyber glow
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=CYBER_CYAN, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=CYBER_MAGENTA, thickness=2)
                    )

                    # classify the gesture
                    fingers = what_finger_be_doing(hand_landmarks)
                    gesture = classify_gesture(fingers, hand_landmarks)
                    if gesture:
                        gesture_text = gesture

                    frame_w = frame.shape[1]
                    frame_h = frame.shape[0]
                    x = int(hand_landmarks.landmark[8].x * frame_w)
                    y = int(hand_landmarks.landmark[8].y * frame_h)

                    x = frame_w - x
                    draw_point = (x, y)

                    if gesture == "Closed Fist":
                        if not dragging:
                            dragging = True
                            prev_fist_pos = draw_point
                        else:
                            dx = draw_point[0] - prev_fist_pos[0]
                            dy = draw_point[1] - prev_fist_pos[1]

                            if abs(dx) > 0 or abs(dy) > 0:
                                canvas = shift_canvas(canvas, -dx, dy)

                            prev_fist_pos = draw_point


                    else:
                        dragging = False
                        prev_fist_pos = None

                    if gesture == "Open Palm":
                        # compute roll using wrist
                        wrist = hand_landmarks.landmark[0]
                        middle_tip = hand_landmarks.landmark[12]
                        dx = middle_tip.x - wrist.x
                        # if open hand is tilted enough, switch color
                        if dx < -0.10:
                            color_index = 7
                        elif -0.10 <= dx < -0.05:
                            color_index = 6
                        elif -0.05 <= dx < 0.00:
                            color_index = 5
                        elif 0.00 <= dx < 0.05:
                            color_index = 4
                        elif 0.05 <= dx < 0.10:
                            color_index = 3
                        elif 0.10 <= dx < 0.15:
                            color_index = 2
                        elif 0.15 <= dx < 0.20:
                            color_index = 1
                        else:
                            color_index = 0
                        current_color = colors[color_index]

            # flip frame
            frame = cv2.flip(frame, 1)
            canvas = cv2.flip(canvas, 1)

            # add grid background
            draw_cyber_grid(frame, grid_size=40, frame_count=frame_count)

            # cyberpunk styling
            if gesture_text:
                # text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                font_thickness = 2
                text_size = cv2.getTextSize(gesture_text, font, font_scale, font_thickness)[0]
                
                # background and cyber corners
                padding = 20
                bg_x1 = 20
                bg_y1 = 20
                bg_x2 = bg_x1 + text_size[0] + padding * 2
                bg_y2 = bg_y1 + text_size[1] + padding * 2
                
                # dark background
                overlay = frame.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), CYBER_DARK, -1)
                frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
                
                # cyber corner brackets
                draw_glitch_box(frame, bg_x1, bg_y1, bg_x2, bg_y2, CYBER_CYAN, 2)
                
                # make text glow
                text_x = bg_x1 + padding
                text_y = bg_y1 + text_size[1] + padding - 5
                
                # outer glow
                for offset in range(3, 0, -1):
                    alpha = 0.3 / offset
                    glow_color = tuple(int(c * alpha) for c in CYBER_CYAN)
                    cv2.putText(frame, gesture_text, (text_x, text_y),
                               font, font_scale, glow_color, font_thickness + offset * 2)
                
                # main text
                cv2.putText(frame, gesture_text, (text_x, text_y),
                           font, font_scale, CYBER_CYAN, font_thickness)

            # drawing/erasing
            if gesture_text == "Eraser" and draw_point is not None:
                if prev_point is not None:
                    cv2.line(canvas, prev_point, draw_point, (0, 0, 0), 20)
                prev_point = draw_point
            elif gesture_text == "Pointing/Drawing" and draw_point is not None:
                if prev_point is not None:
                    # making drawing glow
                    cv2.line(canvas, prev_point, draw_point, current_color, 8)
                    cv2.line(canvas, prev_point, draw_point, (255, 255, 255), 3)
                prev_point = draw_point
            elif gesture_text == "i in ASL":
                canvas[:] = 0
                prev_point = None
            else:
                prev_point = None

            # combine canvas and camera
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
            frame[mask] = canvas[mask]
            
            # colors
            palette_width = 300
            palette_height = 90
            palette_x = frame.shape[1] - palette_width - 20
            palette_y = 20
            
            # dark background
            overlay = frame.copy()
            cv2.rectangle(overlay, (palette_x, palette_y), 
                         (palette_x + palette_width, palette_y + palette_height), 
                         CYBER_DARK, -1)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
            
            # cyber corner brackets
            draw_glitch_box(frame, palette_x, palette_y, 
                          palette_x + palette_width, palette_y + palette_height, 
                          CYBER_MAGENTA, 2)
            
            # title with glow
            title_text = "COLOR MATRIX"
            cv2.putText(frame, title_text, (palette_x + 12, palette_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYBER_DARK, 3)
            cv2.putText(frame, title_text, (palette_x + 10, palette_y + 26),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYBER_MAGENTA, 1)
            
            # draw color circles with cyber styling
            circle_radius = 16
            start_x = palette_x + 20
            circle_y = palette_y + 60
            spacing = 35
            
            for i, color in enumerate(colors):
                circle_x = start_x + i * spacing
                
                # draw outer glow ring if selected
                if i == color_index:
                    for r in range(28, 22, -2):
                        alpha = 0.4 * (28 - r) / 6
                        glow_color = tuple(int(c * alpha) for c in CYBER_CYAN)
                        cv2.circle(frame, (circle_x, circle_y), r, glow_color, 1)
                    
                    # hexagon effect
                    cv2.circle(frame, (circle_x, circle_y), circle_radius + 6, 
                              CYBER_CYAN, 2)
                
                # draw color circle with dark border
                cv2.circle(frame, (circle_x, circle_y), circle_radius + 1, 
                          CYBER_DARK, -1)
                cv2.circle(frame, (circle_x, circle_y), circle_radius, color, -1)
                
                # inner highlight
                cv2.circle(frame, (circle_x - 4, circle_y - 4), 3, 
                          (255, 255, 255), -1)
            
            # add scanline effect
            draw_scanlines(frame, spacing=4)
            
            # HUD-style corner indicators
            h, w = frame.shape[:2]
            cv2.line(frame, (10, 10), (40, 10), CYBER_CYAN, 2)
            cv2.line(frame, (10, 10), (10, 40), CYBER_CYAN, 2)
            cv2.line(frame, (w-40, 10), (w-10, 10), CYBER_CYAN, 2)
            cv2.line(frame, (w-10, 10), (w-10, 40), CYBER_CYAN, 2)
            
            # display window
            cv2.imshow("CYBERHAND INTERFACE", frame)

            # unflip canvas so drawing maps onto the finger
            canvas = cv2.flip(canvas, 1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()

#===================
# MAIN FUNCTION
#===================
if __name__ == "__main__":
    run_tracking()
