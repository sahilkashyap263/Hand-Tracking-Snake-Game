import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from collections import deque
import json
import os
import time

# ===== CONFIGURATION =====
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS

# ===== CAMERA SETUP WITH OPTIMIZATION =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# ===== HAND DETECTOR WITH OPTIMIZATION =====
detector = HandDetector(detectionCon=0.7, maxHands=1, minTrackCon=0.5)

# ===== HIGH SCORE MANAGER =====
class HighScoreManager:
    def __init__(self, filename='snake_highscore.json'):
        self.filename = filename
        self.highscore = self.load()
    
    def load(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    return data.get('highscore', 0)
        except:
            pass
        return 0
    
    def save(self, score):
        if score > self.highscore:
            self.highscore = score
            try:
                with open(self.filename, 'w') as f:
                    json.dump({'highscore': self.highscore}, f)
            except:
                pass
    
    def get(self):
        return self.highscore

# ===== OPTIMIZED SMOOTHING FILTER =====
class PointSmoother:
    def __init__(self, buffer_size=3):  # Reduced for better responsiveness
        self.buffer = deque(maxlen=buffer_size)
    
    def smooth(self, point):
        self.buffer.append(point)
        if len(self.buffer) == 0:
            return point
        
        # Weighted average - more weight to recent points
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for i, p in enumerate(self.buffer):
            weight = i + 1  # More recent = higher weight
            weighted_x += p[0] * weight
            weighted_y += p[1] * weight
            total_weight += weight
        
        return (int(weighted_x / total_weight), int(weighted_y / total_weight))
    
    def clear(self):
        self.buffer.clear()

# ===== OPTIMIZED GRADIENT CACHE =====
class GradientCache:
    def __init__(self, width, height):
        self.gradient = self._create_gradient(width, height)
    
    def _create_gradient(self, width, height):
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            alpha = i / height
            color = (
                int(20 * (1 - alpha) + 40 * alpha),
                int(20 * (1 - alpha) + 30 * alpha),
                int(30 * (1 - alpha) + 50 * alpha)
            )
            gradient[i, :] = color
        return gradient
    
    def apply(self, img):
        return cv2.addWeighted(self.gradient, 0.6, img, 0.4, 0)

class SnakeGameClass:
    def __init__(self, pathFood, highscore_manager):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = None
        self.smoother = PointSmoother(buffer_size=3)
        self.highscore_manager = highscore_manager
        
        # Cached gradient background
        self.gradient_cache = GradientCache(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Load and resize food image once
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            print(f"Error: Could not load {pathFood}")
            exit()

        if self.imgFood.shape[2] == 3:
            self.imgFood = cv2.cvtColor(self.imgFood, cv2.COLOR_BGR2BGRA)

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False
        self.gameStarted = False
        
        # Animation states
        self.foodPulse = 0
        self.ateFood = False
        self.ateAnimationCounter = 0
        
        # Pre-calculated values
        self.boundary_min = 40
        self.boundary_max_x = SCREEN_WIDTH - 40
        self.boundary_max_y = SCREEN_HEIGHT - 40

    def randomFoodLocation(self):
        """Generate food in safe zones"""
        self.foodPoint = (
            random.randint(150, SCREEN_WIDTH - 150), 
            random.randint(150, SCREEN_HEIGHT - 150)
        )

    def drawStartScreen(self, imgMain):
        """Optimized start screen"""
        imgMain = self.gradient_cache.apply(imgMain)
        
        # Semi-transparent overlay
        overlay = imgMain.copy()
        cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, imgMain, 0.5, 0, imgMain)
        
        # Title
        cvzone.putTextRect(imgMain, "SNAKE GAME", [300, 120],
                           scale=8, thickness=10, offset=25,
                           colorR=(30, 30, 50), colorT=(0, 255, 255))
        
        # High score
        highscore = self.highscore_manager.get()
        if highscore > 0:
            cvzone.putTextRect(imgMain, f'High Score: {highscore}', [480, 220],
                               scale=3, thickness=4, offset=12,
                               colorR=(30, 30, 50), colorT=(255, 215, 0))
        
        # Instructions Panel
        panel_x, panel_y = 200, 280
        panel_w, panel_h = 880, 280
        cv2.rectangle(imgMain, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (40, 40, 60), -1)
        cv2.rectangle(imgMain, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (100, 200, 255), 3)
        
        cvzone.putTextRect(imgMain, "HOW TO PLAY", [470, 320],
                          scale=3, thickness=4, offset=10,
                          colorR=(40, 40, 60), colorT=(0, 255, 255))
        
        instructions = [
            ("1", "Use INDEX FINGER to control the snake", 370),
            ("2", "Eat donuts to grow and score points", 410),
            ("3", "Avoid BOUNDARIES (stay in the box)", 450),
            ("4", "Don't let snake bite ITSELF", 490)
        ]
        
        for num, text, y_pos in instructions:
            cv2.circle(imgMain, (240, y_pos - 10), 18, (0, 255, 255), -1)
            cv2.putText(imgMain, num, (233, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cvzone.putTextRect(imgMain, text, [280, y_pos],
                              scale=2, thickness=2, offset=5,
                              colorR=(40, 40, 60), colorT=(255, 255, 255))
        
        cvzone.putTextRect(imgMain, "TIP: Smooth movements work best!", 
                          [350, 530], scale=1.8, thickness=2, offset=5,
                          colorR=(40, 40, 60), colorT=(255, 215, 0))
        
        # Animated play button
        pulse = abs(math.sin(self.foodPulse)) * 30
        button_size = int(80 + pulse)
        button_x = 640 - button_size // 2
        button_y = 620 - button_size // 2
        
        cv2.circle(imgMain, (640, 620), button_size + 10, (0, 255, 255), 3)
        cv2.circle(imgMain, (640, 620), button_size, (0, 200, 255), -1)
        cv2.circle(imgMain, (640, 620), button_size, (255, 255, 255), 4)
        
        # Play icon
        triangle_size = 30
        triangle = np.array([
            [640 - triangle_size//2, 620 - triangle_size//2],
            [640 - triangle_size//2, 620 + triangle_size//2],
            [640 + triangle_size//2, 620]
        ], np.int32)
        cv2.fillPoly(imgMain, [triangle], (255, 255, 255))
        
        cvzone.putTextRect(imgMain, "Touch to Start", [480, 680],
                           scale=2.5, thickness=3, offset=8,
                           colorR=(30, 30, 50), colorT=(255, 255, 255))
        
        # Controls info
        cv2.putText(imgMain, "Press 'F' for Fullscreen | ESC to Quit", 
                   (420, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        button_hitbox = (button_x - 30, button_y - 30, button_size + 60, button_size + 60)
        return imgMain, button_hitbox

    def drawUI(self, imgMain):
        """Optimized UI rendering"""
        # Score panel
        cv2.rectangle(imgMain, (30, 30), (350, 120), (30, 30, 50), -1)
        cv2.rectangle(imgMain, (30, 30), (350, 120), (100, 200, 255), 3)
        cvzone.putTextRect(imgMain, f'Score: {self.score}', [45, 75],
                          scale=2.5, thickness=3, offset=5, 
                          colorR=(30, 30, 50), colorT=(255, 255, 255))
        
        highscore = self.highscore_manager.get()
        if highscore > 0:
            cvzone.putTextRect(imgMain, f'Best: {highscore}', [45, 110],
                              scale=1.5, thickness=2, offset=3,
                              colorR=(30, 30, 50), colorT=(200, 200, 200))
        
        # Length bar
        length_percent = min(100, int((self.currentLength / self.allowedLength) * 100))
        bar_width = int(300 * (length_percent / 100))
        cv2.rectangle(imgMain, (30, 140), (330, 160), (40, 40, 60), -1)
        
        bar_color = (0, 255 - int(length_percent * 2.55), int(length_percent * 2.55))
        if bar_width > 0:
            cv2.rectangle(imgMain, (30, 140), (30 + bar_width, 160), bar_color, -1)
        cv2.rectangle(imgMain, (30, 140), (330, 160), (100, 200, 255), 2)

    def drawSnake(self, imgMain):
        """Optimized snake rendering"""
        if len(self.points) < 2:
            return
        
        # Draw body segments in batches
        num_points = len(self.points)
        for i in range(1, num_points):
            ratio = i / num_points
            color = (int(50 * (1 - ratio)), int(100 * ratio), int(255 * ratio))
            thickness = int(15 + 5 * ratio)
            cv2.line(imgMain, tuple(self.points[i - 1]), tuple(self.points[i]), color, thickness)
        
        # Glowing head
        cx, cy = self.points[-1]
        cv2.circle(imgMain, (cx, cy), 25, (0, 255, 255), -1)
        cv2.circle(imgMain, (cx, cy), 20, (0, 255, 0), -1)
        cv2.circle(imgMain, (cx, cy), 12, (255, 255, 255), -1)
        cv2.circle(imgMain, (cx, cy), 30, (0, 255, 255), 2)

    def drawFood(self, imgMain):
        """Optimized food rendering with animation"""
        rx, ry = self.foodPoint
        
        pulse_scale = 1.0 + 0.1 * math.sin(self.foodPulse)
        
        scaled_w = int(self.wFood * pulse_scale)
        scaled_h = int(self.hFood * pulse_scale)
        
        # Only resize if necessary
        if scaled_w != self.wFood or scaled_h != self.hFood:
            resized_food = cv2.resize(self.imgFood, (scaled_w, scaled_h), 
                                     interpolation=cv2.INTER_LINEAR)
        else:
            resized_food = self.imgFood
        
        glow_radius = int(40 * pulse_scale)
        cv2.circle(imgMain, (rx, ry), glow_radius, (0, 255, 255), 2)
        cv2.circle(imgMain, (rx, ry), glow_radius + 5, (255, 255, 0), 1)
        
        imgMain = cvzone.overlayPNG(imgMain, resized_food,
                                    (rx - scaled_w // 2, ry - scaled_h // 2))
        
        return imgMain

    def update(self, imgMain, currentHead):
        imgMain = self.gradient_cache.apply(imgMain)
        
        self.foodPulse = (self.foodPulse + 0.2) % (2 * math.pi)  # Smooth animation
        
        if not self.gameStarted:
            return self.drawStartScreen(imgMain)
        
        if self.gameOver:
            overlay = imgMain.copy()
            cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, imgMain, 0.3, 0, imgMain)
            
            cvzone.putTextRect(imgMain, "GAME OVER", [280, 280],
                               scale=8, thickness=8, offset=20,
                               colorR=(139, 0, 0), colorT=(255, 255, 255))
            cvzone.putTextRect(imgMain, f'Score: {self.score}', [400, 400],
                               scale=5, thickness=5, offset=15,
                               colorR=(30, 30, 50), colorT=(0, 255, 255))
            
            highscore = self.highscore_manager.get()
            if self.score == highscore and self.score > 0:
                cvzone.putTextRect(imgMain, "NEW HIGH SCORE!", [320, 500],
                                   scale=4, thickness=4, offset=12,
                                   colorR=(30, 30, 50), colorT=(255, 215, 0))
            
            cvzone.putTextRect(imgMain, "Press 'R' to Restart", [420, 600],
                               scale=3, thickness=3, offset=10,
                               colorR=(30, 30, 50), colorT=(200, 200, 200))
            return imgMain, None

        cx, cy = currentHead
        
        if self.previousHead is None:
            self.previousHead = cx, cy
            self.drawUI(imgMain)
            return imgMain, None

        px, py = self.previousHead
        distance = math.hypot(cx - px, cy - py)
        
        if distance > 2:
            self.points.append([cx, cy])
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

        # Length management
        while self.currentLength > self.allowedLength and self.lengths:
            self.currentLength -= self.lengths[0]
            self.lengths.pop(0)
            self.points.pop(0)

        # Food collision
        rx, ry = self.foodPoint
        if (rx - self.wFood // 2 < cx < rx + self.wFood // 2 and 
            ry - self.hFood // 2 < cy < ry + self.hFood // 2):
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1
            self.ateFood = True
            self.ateAnimationCounter = 10

        # Draw elements
        self.drawSnake(imgMain)
        imgMain = self.drawFood(imgMain)
        
        if self.ateFood and self.ateAnimationCounter > 0:
            rx, ry = self.foodPoint
            radius = 40 + (10 - self.ateAnimationCounter) * 5
            alpha = self.ateAnimationCounter / 10
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.circle(imgMain, (rx, ry), radius, color, 3)
            self.ateAnimationCounter -= 1
            if self.ateAnimationCounter == 0:
                self.ateFood = False
        
        self.drawUI(imgMain)

        # Boundary collision
        if (cx < self.boundary_min or cx > self.boundary_max_x or 
            cy < self.boundary_min or cy > self.boundary_max_y):
            self.gameOver = True
            self.highscore_manager.save(self.score)
            return imgMain, None

        # Self-collision (more forgiving)
        if len(self.points) > 40:
            pts = np.array(self.points[:-20], np.int32)
            pts = pts.reshape((-1, 1, 2))
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if -15 <= minDist <= 15:
                self.gameOver = True
                self.highscore_manager.save(self.score)

        # Draw boundary
        cv2.rectangle(imgMain, (self.boundary_min, self.boundary_min), 
                     (self.boundary_max_x, self.boundary_max_y), (100, 100, 150), 2)

        return imgMain, None

    def reset(self):
        """Reset game state"""
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = None
        self.smoother.clear()
        self.randomFoodLocation()
        self.score = 0
        self.gameOver = False
        self.gameStarted = False
        self.ateFood = False
        self.ateAnimationCounter = 0


# ===== MAIN GAME LOOP =====
def main():
    highscore_mgr = HighScoreManager()
    game = SnakeGameClass("Donut.jpg", highscore_mgr)
    
    fullscreen = False
    cv2.namedWindow("Snake Game", cv2.WINDOW_NORMAL)
    
    # FPS tracking
    fps_buffer = deque(maxlen=30)
    frame_start = time.time()
    
    while True:
        loop_start = time.time()
        
        success, img = cap.read()
        if not success:
            continue
        
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=True, draw=False)
        
        if not game.gameStarted and not game.gameOver:
            result = game.update(img, (0, 0))
            if isinstance(result, tuple):
                img, button_coords = result
            else:
                img = result
                button_coords = None
            
            if hands and button_coords:
                lmList = hands[0]['lmList']
                if lmList:
                    pointIndex = lmList[8][0:2]
                    bx, by, bw, bh = button_coords
                    if bx < pointIndex[0] < bx + bw and by < pointIndex[1] < by + bh:
                        cv2.circle(img, pointIndex, 30, (0, 255, 0), -1)
                        cv2.circle(img, pointIndex, 35, (255, 255, 255), 3)
                        game.gameStarted = True
                    else:
                        cv2.circle(img, pointIndex, 15, (255, 0, 255), -1)
                        cv2.circle(img, pointIndex, 20, (255, 255, 255), 2)
        
        elif game.gameStarted and not game.gameOver:
            if hands:
                lmList = hands[0]['lmList']
                if lmList:
                    pointIndex = lmList[8][0:2]
                    smoothed_point = game.smoother.smooth(pointIndex)
                    
                    cv2.circle(img, smoothed_point, 15, (255, 0, 255), -1)
                    cv2.circle(img, smoothed_point, 20, (255, 255, 255), 2)
                    
                    result = game.update(img, smoothed_point)
                    if isinstance(result, tuple):
                        img, _ = result
                    else:
                        img = result
            else:
                img = game.gradient_cache.apply(img)
                cvzone.putTextRect(img, "Show your hand!", [450, 360],
                                  scale=4, thickness=4, offset=15,
                                  colorR=(30, 30, 50), colorT=(255, 100, 100))
        
        elif game.gameOver:
            result = game.update(img, (0, 0))
            if isinstance(result, tuple):
                img, _ = result
            else:
                img = result
        
        # FPS calculation
        frame_end = time.time()
        fps = 1.0 / (frame_end - frame_start) if frame_end > frame_start else 0
        frame_start = frame_end
        fps_buffer.append(fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        cv2.putText(img, f'FPS: {int(avg_fps)}', (SCREEN_WIDTH - 130, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Snake Game", img)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') or key == ord('R'):
            game.reset()
        elif key == ord('f') or key == ord('F'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty("Snake Game", cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("Snake Game", cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_NORMAL)
        elif key == 27:  # ESC
            break
        
        # Frame rate limiting
        elapsed = time.time() - loop_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()