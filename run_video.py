from gooey import Gooey, GooeyParser

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import model
import cv2
from subprocess import call
import os

@Gooey(required_cols=2,
       program_name="Run on Video - End to End Self Driving",
       timing_options={
           'show_time_remaining': False,
       }
       )
def parse_args():
    prog_description = 'Test driving model on a video. '
    parser = GooeyParser(description=prog_description)

    sub_parsers = parser.add_subparsers(help='commands', dest='command')

    range_parser = sub_parsers.add_parser('range')
    range_parser.add_argument('--video-file',
                              default=str(os.path.join(os.path.join(os.getcwd(), "driving_dataset", "videos", "1.mp4"))),
                              widget="FileChooser")

    return parser.parse_args()

def run_video(video_path):
    # check if on windows OS
    windows = False
    if os.name == 'nt':
        windows = True

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/model.ckpt")

    img = cv2.imread('./assets/steering_wheel_image.jpg', 0)
    rows, cols = img.shape

    smoothed_angle = 0

    cap = cv2.VideoCapture(video_path)
    while (cv2.waitKey(10) != ord('q')):
        ret, frame = cap.read()
        image = cv2.resize(frame, (200, 66)) / 255.0
        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / 3.14159265
        if not windows:
            call("clear")
        print("Predicted steering angle: " + str(degrees) + " degrees")
        cv2.imshow('frame', frame)
        # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        # and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    conf = parse_args()
    print(conf)
    run_video(conf.video_file)