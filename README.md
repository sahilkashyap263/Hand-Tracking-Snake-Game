# Hand-Controlled Snake Game 🐍

A computer vision-based Snake game controlled by hand gestures using OpenCV and MediaPipe. Guide the snake using your index finger to collect food and grow longer!

## 🎮 Game Features

- **Hand Gesture Control**: Control the snake using your index finger
- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand detection
- **Dynamic Scoring**: Earn points by collecting food items
- **Collision Detection**: Game ends when the snake hits itself
- **Visual Feedback**: Live score display and game over screen

## 📋 Requirements

```bash
opencv-python
cvzone
mediapipe
numpy
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd hand-snake-game
```

2. **Install dependencies**
```bash
pip install opencv-python cvzone mediapipe numpy
```

3. **Add game assets**
   - Place a `Donut.jpg` (or any food image) in the project directory
   - The image will be used as the food item in the game

## 🎯 How to Play

1. **Run the game**
```bash
python snake_game.py
```

2. **Game Controls**
   - Move your **index finger** to control the snake's direction
   - The snake follows your finger position in real-time
   - Collect food items (donuts) to increase your score
   - Avoid hitting the snake's own body

3. **Keyboard Controls**
   - Press `R` to restart after game over
   - Press `ESC` or close window to quit

## 🎨 Game Mechanics

- **Starting Length**: 150 pixels
- **Growth Rate**: +50 pixels per food item
- **Score**: +1 point per food collected
- **Collision**: Game ends if snake head touches its body

## ⚙️ Configuration

You can modify these parameters in the code:

```python
# Camera settings
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Hand detection sensitivity
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initial snake length
self.allowedLength = 150

# Growth per food
self.allowedLength += 50
```

## 🔧 Troubleshooting

**Camera not working?**
- Check if your camera is connected
- Try changing the camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

**Food image not loading?**
- Ensure `Donut.jpg` exists in the project directory
- Use PNG format with transparency for best results
- Check file permissions

**Hand not detected?**
- Ensure good lighting conditions
- Keep hand within camera frame
- Adjust `detectionCon` value (lower = more sensitive)

## 📁 Project Structure

```
hand-snake-game/
├── snake_game.py      # Main game file
├── Donut.jpg          # Food image asset
└── README.md          # This file
```

## 🎓 How It Works

1. **Hand Detection**: Uses CVZone's HandDetector (built on MediaPipe) to track hand landmarks
2. **Snake Movement**: Index finger tip (landmark 8) coordinates control the snake head
3. **Collision Detection**: Uses OpenCV's `pointPolygonTest` to detect self-collision
4. **Food Generation**: Randomly spawns food at safe locations
5. **Rendering**: Overlays food image and draws snake using OpenCV functions

## 🤝 Contributing

Feel free to fork this project and submit pull requests for:
- Bug fixes
- New features (power-ups, obstacles, levels)
- UI improvements
- Additional control methods

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Credits

- **CVZone**: Computer vision library by Murtaza Hassan
- **OpenCV**: Open source computer vision library
- **MediaPipe**: Google's ML solutions for hand tracking

## 🎮 Future Enhancements

- [ ] Multiple difficulty levels
- [ ] Power-ups and special items
- [ ] Obstacles and walls
- [ ] High score tracking
- [ ] Sound effects
- [ ] Two-hand mode for multiplayer

---

**Enjoy the game! 🎉**

If you encounter any issues, please open an issue on GitHub.