# Hand Gesture Cursor

Control your computer cursor using nothing but your hand.

This project uses **computer vision and hand tracking** to turn real-time hand movements into mouse movement and gestures — no physical mouse required.

I built this project to get hands-on with **MediaPipe, real-time computer vision, coordinate mapping, gesture detection, and input automation**.

---

## What It Does

* Tracks your hand through the webcam in real time
* Uses the **palm center** for stable cursor navigation
* Maps hand movement to screen coordinates
* Detects a **pinch gesture** for mouse clicking
* Uses normalized 3D landmark distances to make pinch detection more robust to hand size and camera distance
* Uses an **active interaction box** to reduce accidental cursor movement near the edges
* Applies smoothing to make cursor movement less jittery

---

## Demo

> Move your hand → move the cursor
> Pinch your thumb and index finger → click

![Hand Gesture Cursor Demo](demo.gif)

---

## How It Works

The webcam feed is processed frame-by-frame using MediaPipe's hand landmark detection.

### 1. Hand Tracking

MediaPipe detects key landmarks on the hand and provides their normalized:

```text
x → horizontal position
y → vertical position
z → relative depth
```

These landmarks form the foundation of the cursor and gesture system.

### 2. Cursor Movement

Instead of following the index fingertip directly, the project uses the **palm center** as the navigation point.

The palm position is mapped from the camera's coordinate system to the screen coordinate system.

An active interaction box is used so that small movements near the camera frame boundaries don't cause extreme cursor jumps.

### 3. Pinch Detection

A click is triggered when the thumb and index finger come close together.

Rather than relying only on raw pixel distance, the project uses a **scale-normalized 3D distance**, making the detection more consistent when the hand moves closer to or farther from the camera.

### 4. Smoothing

Raw hand tracking can be noisy.

A smoothing mechanism is applied to the detected position before moving the cursor, producing more natural movement.

---

## Tech Stack

* **Python**
* **MediaPipe** — hand landmark detection
* **OpenCV** — webcam/video processing
* **PyAutoGUI** — mouse control
* **NumPy** — numerical calculations

---

## Project Structure

```text
hand-gesture-cursor/
│
├── main.py
├── verifySetup.py
├── requirements.txt
└── .gitignore
```

### `main.py`

The main application containing:

* webcam capture
* hand tracking
* coordinate mapping
* cursor movement
* pinch detection
* smoothing
* interaction logic

### `verifySetup.py`

Checks whether the required environment and dependencies are correctly configured before running the project.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kunalsinverma/hand-gesture-cursor.git
cd hand-gesture-cursor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify the setup:

```bash
python verifySetup.py
```

Run:

```bash
python main.py
```

Make sure your webcam is available.

---

## Current Version

### v1.0 — MVP

The first version focuses on making the core interaction actually work:

* Hand tracking
* Palm-based cursor navigation
* Screen coordinate mapping
* Active interaction region
* 3D scale-normalized pinch detection
* Mouse click automation
* Position smoothing

---

## What I Learned

This project started as a simple "move the cursor with my hand" idea, but quickly became a practical lesson in dealing with noisy real-world input.

The interesting part wasn't just detecting a hand — it was making the interaction **stable enough to feel usable**.

Some of the concepts I explored:

* Coordinate-system transformations
* Normalized vs. absolute coordinates
* 3D landmark geometry
* Scale normalization
* Real-time processing
* Signal smoothing
* Gesture state detection
* Human-computer interaction

---

## Why I Built This

I wanted a project where I couldn't hide behind tutorials.

A webcam, some landmarks, a cursor, and a problem:

**How do I make this actually feel good to use?**

That's what this project is about.

---

## Author

**Kunal Singh Verma**

B.Tech — Artificial Intelligence & Data Science

Interested in **AI, systems, computer vision, and open source**.

[GitHub](https://github.com/kunalsinverma)
