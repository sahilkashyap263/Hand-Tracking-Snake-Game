# Hand-Controlled Snake Game 🐍✋

Control a snake using your hand gestures! Move your index finger to guide the snake, collect donuts, and beat your high score.

## ✨ Features

- **Hand Gesture Control**: Use your index finger to control the snake
- **High Score Tracking**: Your best score is saved automatically
- **Smooth Gameplay**: Optimized for 60 FPS performance
- **Interactive Tutorial**: Easy-to-follow instructions on start screen
- **Fullscreen Mode**: Immersive gameplay experience

## 📋 Requirements

```bash
opencv-python
cvzone
numpy
```

## 🚀 Quick Start

1. **Install dependencies**
```bash
pip install opencv-python cvzone numpy
```

2. **Add the food image**
   - Place a `Donut.jpg` in the project folder

3. **Run the game**
```bash
python snake_game.py
```

## 🎮 How to Play

- **Move**: Use your index finger to control the snake
- **Goal**: Collect donuts to grow and score points
- **Avoid**: Hitting the boundaries or yourself

### Controls
- `F` - Toggle fullscreen
- `R` - Restart after game over
- `ESC` - Quit game

## 🔧 Troubleshooting

**Camera not detected?**
- Check if camera is connected
- Try changing camera index to `1` or `2` in code

**Hand not detected?**
- Ensure good lighting
- Keep hand clearly visible to camera
- Position hand 1-2 feet from camera

**Performance issues?**
- Close other applications using the camera
- Reduce screen resolution in code if needed

## 📁 Files

```
├── snake_game.py           # Main game file
├── Donut.jpg               # Food image
├── snake_highscore.json    # High score (auto-generated)
└── README.md
```

## 🎯 Tips

- Make smooth, gradual movements for better control
- Play in a well-lit area
- Watch the boundary box to avoid collisions
- The snake gets harder to control as it grows longer!

## 🤝 Contributing

Ideas for improvement:
- Multiple difficulty levels
- Power-ups and obstacles
- Sound effects
- Multiplayer mode
- Custom themes

## 📝 License

MIT License - feel free to use and modify!

## 🙏 Credits

- **CVZone** - Computer vision library
- **OpenCV** - Image processing
- **MediaPipe** - Hand tracking

---

**Enjoy the game! 🎉 Beat your high score!**

Star ⭐ this repo if you like it!