# Basketball Shot Analyzer

A real-time basketball shooting form analyzer that uses your webcam to track
body pose and score shooting mechanics as you shoot.

## What it does

Using [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
for real-time body landmark detection, the script measures five components of
shooting form on every frame and gives live visual feedback:

- **Elbow angle** - wants 80-110°
- **Knee bend** - wants 140°+
- **Elbow height** (shoulder angle) - wants 70°+
- **Stance width** (ankle-to-shoulder ratio) - wants 0.8-1.4
- **Wrist position** - wrist above elbow at release

Each metric is drawn on screen in green (good) or red (needs work), along
with an overall 0-100 form score and a live FPS counter.

## Demo

- [Demo clip 1](assets/demo-1.mp4)
- [Demo clip 2](assets/demo-2.mp4)

## Run it

```bash
pip install -r requirements.txt
python jumpshot.py
```

Stand back far enough that your full body is in frame, then take some shots.
Press `q` to quit - average FPS over the session is printed to the console.

By default it tracks the right side; change `SHOOTING_SIDE` in `jumpshot.py`
to `'left'` if you shoot left-handed.
