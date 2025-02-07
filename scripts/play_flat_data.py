#!/usr/bin/env python3

import os
import csv
import pickle
from typing import List, Dict, Tuple
from PIL import Image as PilImage
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from hydra_stretch_msgs.msg import Mask, Masks, HydraVisionPacket
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix
import tf2_ros

from std_srvs.srv import Empty, EmptyResponse

CLASS_IDS = [2, 6, 7, 8, 13, 14, 16, 18, 20, 22, 27]


class FlatDataPlayer(object):
    def __init__(self):
        """Initialize ros node and read params"""
        rospy.init_node("flat_data_player")
        # params
        self.data_path = str(
            rospy.get_param("~data_path", "/home/ros/bags/flat_dataset/run1")
        )
        self.global_frame_name = str(rospy.get_param("~global_frame_name", "map"))
        self.sensor_frame_name = str(rospy.get_param("~sensor_frame_name", "depth_cam"))
        self.seg_model = str(rospy.get_param("~seg_model", "inst"))
        self.play_rate = rospy.get_param("~play_rate", 1.0)
        self.wait = rospy.get_param("~wait", False)
        max_frames = rospy.get_param("~max_frames", int(1e8))
        assert isinstance(
            max_frames, int
        ), f"max_frames needs to be int, got {max_frames}"
        self.max_frames = max_frames
        self.timer_pediod = 0.5  # seconds

        # ROS
        self.color_pub = rospy.Publisher("~color/image_raw", Image, queue_size=100)
        self.depth_pub = rospy.Publisher("~depth/image_raw", Image, queue_size=100)
        self.id_pub = rospy.Publisher("~semantics/image_raw", Image, queue_size=100)
        self.packet_pub = rospy.Publisher(
            "~vision_packet", HydraVisionPacket, queue_size=100
        )
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=100)
        self.cam_info_pub = rospy.Publisher(
            "~depth/camera_info", CameraInfo, queue_size=100
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # setup
        self.cv_bridge = CvBridge()
        stamps_file = os.path.join(self.data_path, "timestamps.csv")
        self.times = []
        self.ids = []
        self.current_index = 0  # Used to iterate through
        self.mask_id = 0
        if not os.path.isfile(stamps_file):
            rospy.logfatal("No timestamp file '%s' found." % stamps_file)
        with open(stamps_file, "r") as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if row[0] == "ImageID":
                    continue
                self.ids.append(str(row[0]))
                self.times.append(float(row[1]) / 1e9)

        self.ids = [x for _, x in sorted(zip(self.times, self.ids))]
        self.times = sorted(self.times)
        self.times = [(x - self.times[0]) / self.play_rate for x in self.times]
        self.start_time = None

        if self.wait:
            self.start_srv = rospy.Service("~start", Empty, self.start)
        else:
            self.start(Empty)

    def start(self, _):
        self.running = True
        self.timer = rospy.Timer(rospy.Duration(self.timer_pediod), self.callback)
        return EmptyResponse()

    def get_semantics(self, files: Dict):
        seg_path = ""
        if self.seg_model == "inst":
            seg_path = "hydra_inst"
        elif self.seg_model == "none":
            seg_path = ""
        else:
            raise NotImplementedError

        if seg_path != "":
            pred_file = os.path.join(
                self.data_path,
                seg_path,
                self.ids[self.current_index] + "_segmentation.png",
            )
            files["semantics"] = pred_file

        if self.seg_model == "inst":
            mask_path = "hydra_inst/mask"
            mask_file = os.path.join(
                self.data_path,
                mask_path,
                self.ids[self.current_index] + "_masks.pkl",
            )
            files["masks"] = mask_file

        for f in files.values():
            if not os.path.isfile(f):
                rospy.logwarn("Could not find file '%s', skipping frame." % f)
                self.current_index += 1
                return

    def get_color_image(self, color_cv_img: np.ndarray, stamp: rospy.Time) -> Image:

        # Load and publish Color image.
        color_cv_img = cv2.cvtColor(color_cv_img, cv2.COLOR_BGR2RGB)
        color_img_msg = self.cv_bridge.cv2_to_imgmsg(color_cv_img, "rgb8")
        color_img_msg.header.stamp = stamp
        color_img_msg.header.frame_id = self.sensor_frame_name
        return color_img_msg

    def get_depth_image(self, fname: str, stamp: rospy.Time) -> Image:
        # Load and publish depth image.
        depth_cv_img = PilImage.open(fname)
        depth_img_msg = self.cv_bridge.cv2_to_imgmsg(np.array(depth_cv_img), "32FC1")
        depth_img_msg.header.stamp = stamp
        depth_img_msg.header.frame_id = self.sensor_frame_name
        return depth_img_msg

    def get_masks_msg(
        self, mask_file: str, stamp: rospy.Time, image_header: Header
    ) -> Masks:
        # Load and publish instance masks.
        masks_dict = {}
        with open(mask_file, "rb") as mf:
            masks_dict = pickle.load(mf)
            mf.close()
        masks_msg = Masks()
        masks_msg.masks = []
        masks_msg.header.stamp = stamp
        masks_msg.header.frame_id = self.sensor_frame_name
        masks_msg.image_header = image_header
        for mask_dict in masks_dict.values():
            cls_id, mask = mask_dict["class_id"], mask_dict["mask"]
            if cls_id in CLASS_IDS:
                mask_msg = Mask()
                mask_msg.class_id = cls_id
                mask_cv = mask.astype(np.uint8)
                mask_msg.data = self.cv_bridge.cv2_to_imgmsg(mask_cv, "mono8")
                mask_msg.mask_id = self.mask_id
                self.mask_id += 1
                masks_msg.masks.append(mask_msg)
        return masks_msg

    def get_camera_info_msg(self, stamp: rospy.Time) -> CameraInfo:
        # Publish camera info
        # Intrinsics:
        # - Res_x: 640
        # - Res_y: 480
        # - u: 320
        # - v: 240
        # - f_x: 320 (px)
        # - f_y: 320
        # (-->horizontal fov = 90deg, vertical fov ~72deg)
        cam_info_msg = CameraInfo()
        cam_info_msg.header.stamp = stamp
        cam_info_msg.header.frame_id = self.sensor_frame_name
        cam_info_msg.height = 480
        cam_info_msg.width = 640
        cam_info_msg.distortion_model = "radial-tangential"
        cam_info_msg.D = [0.0, 0.0, 0.0, 0.0]
        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        cam_info_msg.K = [320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]
        # Rectification matrix (stereo cameras only)
        cam_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Projection/camera matrix
        #     [fx'  0  cx' Tx]
        # P = [ 0  fy' cy' Ty]
        #     [ 0   0   1   0]
        cam_info_msg.P = [
            320.0,
            0.0,
            320.0,
            0.0,
            0.0,
            320.0,
            240.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return cam_info_msg

    def publish_pose(self, pose_file: str, stamp: rospy.Time):
        # Load and publish transform.
        if os.path.isfile(pose_file):
            pose_data = [float(x) for x in open(pose_file, "r").read().split()]
            transform = np.eye(4)
            for row in range(4):
                for col in range(4):
                    transform[row, col] = pose_data[row * 4 + col]
            tf2_msg = TransformStamped()
            tf2_msg.header.stamp = stamp
            tf2_msg.header.frame_id = self.global_frame_name
            tf2_msg.child_frame_id = self.sensor_frame_name
            translation = transform[:, 3]
            rotation = quaternion_from_matrix(transform)
            tf2_msg.transform.translation.x = translation[0]
            tf2_msg.transform.translation.y = translation[1]
            tf2_msg.transform.translation.z = translation[2]
            tf2_msg.transform.rotation.x = rotation[0]
            tf2_msg.transform.rotation.y = rotation[1]
            tf2_msg.transform.rotation.z = rotation[2]
            tf2_msg.transform.rotation.w = rotation[3]
            self.tf_broadcaster.sendTransform(tf2_msg)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = self.global_frame_name
            pose_msg.pose.position.x = pose_data[3]
            pose_msg.pose.position.y = pose_data[7]
            pose_msg.pose.position.z = pose_data[11]
            pose_msg.pose.orientation.x = rotation[0]
            pose_msg.pose.orientation.y = rotation[1]
            pose_msg.pose.orientation.z = rotation[2]
            pose_msg.pose.orientation.w = rotation[3]
            self.pose_pub.publish(pose_msg)

    def callback(self, _):
        # Check we should be publishing.
        if not self.running:
            return

        # Check we're not done.
        if self.current_index >= len(self.times):
            rospy.loginfo("Finished playing the dataset.")
            rospy.signal_shutdown("Finished playing the dataset.")
            return

        # Check the time.
        now = rospy.Time.now()
        if self.start_time is None:
            self.start_time = now
        if self.times[self.current_index] > (now - self.start_time).to_sec():
            return

        # Get all data and publish.
        file_id = os.path.join(self.data_path, self.ids[self.current_index])

        # Color.
        color_file = file_id + "_color.png"
        depth_file = file_id + "_depth.tiff"
        pose_file = file_id + "_pose.txt"
        files = {"color": color_file, "depth": depth_file, "pose": pose_file}
        self.get_semantics(files)
        vision_packet_msg = HydraVisionPacket()
        vision_packet_msg.map_view_id = self.current_index
        depth_img_msg = self.get_depth_image(files["depth"], now)
        self.depth_pub.publish(depth_img_msg)
        color_img_msg = self.get_color_image(cv2.imread(files["color"]), now)
        self.color_pub.publish(color_img_msg)
        if "semantics" in files.keys():
            label_img_msg = self.get_color_image(cv2.imread(files["semantics"]), now)
            vision_packet_msg.label = label_img_msg
            self.id_pub.publish(label_img_msg)
        if "masks" in files.keys():
            masks_msg = self.get_masks_msg(files["masks"], now, color_img_msg.header)
            vision_packet_msg.masks = masks_msg
        vision_packet_msg.color = color_img_msg
        vision_packet_msg.depth = depth_img_msg
        self.packet_pub.publish(vision_packet_msg)
        self.cam_info_pub.publish(self.get_camera_info_msg(now))
        self.publish_pose(
            files["pose"],
            now,
        )

        self.current_index += 1
        if self.current_index > self.max_frames:
            rospy.signal_shutdown(f"Played reached max frames {self.max_frames}")


if __name__ == "__main__":
    flat_data_player = FlatDataPlayer()
    rospy.spin()
