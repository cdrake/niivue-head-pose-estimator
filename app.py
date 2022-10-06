from sys import stdout
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from utils import base64_to_pil_image, pil_image_to_base64
import cv2
import numpy as np
import base64
import io
from imageio import imread
import matplotlib.pyplot as plt

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
height = 150
width = 300

# 2. Introduce a pose estimator to solve pose.
pose_estimator = PoseEstimator(img_size=(height, width))

# 3. Introduce a mark detector to detect landmarks.
mark_detector = MarkDetector()

@socketio.on('input image', namespace='/pose')
def pose_message(input):
    image_data = input.split(",")[1]    
    img = imread(io.BytesIO(base64.b64decode(image_data)))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # frame = imread(io.BytesIO(base64.b64decode(image_data)))
    # Step 1: Get a face from current frame.
    facebox = mark_detector.extract_cnn_facebox(frame)

    # Any face found?
    if facebox is not None:

        # Step 2: Detect landmarks. Crop and feed the face area into the
        # mark detector.
        x1, y1, x2, y2 = facebox
        face_img = frame[y1: y2, x1: x2]

        # Run the detection.
        marks = mark_detector.detect_marks(face_img)

        # Convert the locations from local face area to the global image.
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1

        # Try pose estimation with 68 points.
        pose = pose_estimator.solve_pose_by_68_points(marks)

        # All done. The best way to show the result would be drawing the
        # pose on the frame in realtime.

        # Do you want to see the pose annotation?
        pose_estimator.draw_annotation_box(
            frame, pose[0], pose[1], color=(0, 255, 0))

    cv2_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("reconstructed.jpg", frame)
    retval, buffer = cv2.imencode('.jpg', frame)
    b = base64.b64encode(buffer)
    b = b.decode()
    image_data = "data:image/jpeg;base64," + b

    emit('out-image-event', {'image_data': image_data}, namespace='/pose')

@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""

    app.logger.info("starting to generate frames!")
    while True:
        frame = camera.get_frame() #pil_image_to_base64(camera.get_frame())

        # print(type(frame))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app)
