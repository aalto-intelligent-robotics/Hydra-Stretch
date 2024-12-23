import os

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros


class stretchSegNode:
    def __init__(self):
        rospy.init_node("stretch_oneformer_node", log_level=rospy.DEBUG)
        rospy.loginfo("Starting stretchSegNode.")
        self.rgb_sub = rospy.Subscriber("/astra/color/image_raw", Image, self.rgb_cb)
        self.rgb_info_sub = rospy.Subscriber(
            "/astra/color/camera_info", CameraInfo, self.rgb_info_cb
        )
        self.depth_info_sub = rospy.Subscriber(
            "/astra/depth/camera_info", CameraInfo, self.depth_info_cb
        )
        self.rgb_info_pub = rospy.Publisher(
            "/astra/color/camera_info_proper", CameraInfo, queue_size=10
        )
        self.depth_info_pub = rospy.Publisher(
            "/astra/depth/camera_info_proper", CameraInfo, queue_size=10
        )

        self.seg_dir = str(
            rospy.get_param(
                "~segmentation_path",
                "/home/ros/bags/stretch/stretch_run_000/seg"
            )
        )
        if self.seg_dir.lower() != "none":
            self.seg_pub = rospy.Publisher("/astra/seg_cam/image_raw", Image, queue_size=10)
        self.bridge = CvBridge()
        self.cnt = 0

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.rate = rospy.Rate(10.0)

    def rgb_cb(self, msg):
        if self.seg_dir.lower() != "none":
            frame_path = os.path.join(
                self.seg_dir,
                f"{str(self.cnt).zfill(5)}.png",
            )
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.cnt += 1
                imgMsg = self.bridge.cv2_to_imgmsg(frame, "rgb8")
                imgMsg.header = msg.header
                self.seg_pub.publish(imgMsg)
                rospy.logdebug(f"Published {frame_path}")
            else:
                rospy.logerr(f"{frame_path} doesn't exist")

    def rgb_info_cb(self, msg):
        pub_msg = msg
        pub_msg.header.frame_id = "astra_color_optical_frame"
        self.rgb_info_pub.publish(pub_msg)

    def depth_info_cb(self, msg):
        pub_msg = msg
        pub_msg.header.frame_id = "astra_depth_optical_frame"
        self.depth_info_pub.publish(pub_msg)

if __name__ == "__main__":
    stretch_oneformer_node = stretchSegNode()
    rospy.spin()
