#!/usr/bin/env python3

import os
import csv
import pickle

import rospy
from sensor_msgs.msg import Image, CameraInfo 
from geometry_msgs.msg import PoseStamped 
from hydra_stretch_msgs.msg import Mask, Masks, HydraVisionPacket
from cv_bridge import CvBridge
import cv2
from PIL import Image as PilImage
import numpy as np
from tf.transformations import quaternion_from_matrix
import tf2_ros
from geometry_msgs.msg import TransformStamped

from std_srvs.srv import Empty, EmptyResponse


class FlatDataPlayer(object):
    def __init__(self):
        """Initialize ros node and read params"""
        # params
        self.data_path = str(
            rospy.get_param("~data_path", "/home/ros/bags/flat_dataset/run1")
        )
        self.global_frame_name = rospy.get_param("~global_frame_name", "world")
        self.sensor_frame_name = rospy.get_param("~sensor_frame_name", "depth_cam")
        self.seg_model = rospy.get_param("~seg_model", "gt")
        self.play_rate = rospy.get_param("~play_rate", 1.0)
        self.wait = rospy.get_param("~wait", False)
        self.max_frames = rospy.get_param("~max_frames", 1e9)
        self.refresh_rate = 100.0  # Hz

        # ROS
        self.color_pub = rospy.Publisher("~color/image_raw", Image, queue_size=100)
        self.depth_pub = rospy.Publisher("~depth/image_raw", Image, queue_size=100)
        self.id_pub = rospy.Publisher("~segmentation_image", Image, queue_size=100)
        # self.mask_pub = rospy.Publisher("~masks", Masks, queue_size=100)
        self.packet_pub = rospy.Publisher("~vision_packet", HydraVisionPacket, queue_size=100)
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
            self.start(None)

    def start(self, _):
        self.running = True
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.refresh_rate), self.callback)
        return EmptyResponse()

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
        files = [color_file, depth_file, pose_file]
        seg_path = ""
        if self.seg_model == "gt":
            seg_path = "hydra_seg_gt"
        elif self.seg_model == "gt30":
            seg_path = "hydra_seg_gt30"
        elif self.seg_model == "inst":
            seg_path = "hydra_inst"
        elif self.seg_model == "none":
            seg_path = None
        else:
            raise NotImplementedError
            
        if self.seg_model != "none":
            pred_file = os.path.join(
                self.data_path,
                seg_path,
                self.ids[self.current_index] + "_segmentation.png",
            )
            files.append(pred_file)

            if self.seg_model == "inst":
                mask_path = "hydra_inst/mask"
                mask_file = os.path.join(
                    self.data_path,
                    mask_path,
                    self.ids[self.current_index] + "_masks.pkl",
                )
                files.append(mask_file)

            for f in files:
                if not os.path.isfile(f):
                    rospy.logwarn("Could not find file '%s', skipping frame." % f)
                    self.current_index += 1
                    return

        # Load and publish Color image.
        color_cv_img = cv2.imread(color_file)
        color_cv_img = cv2.cvtColor(color_cv_img, cv2.COLOR_BGR2RGB)
        color_img_msg = self.cv_bridge.cv2_to_imgmsg(color_cv_img, "rgb8")
        color_img_msg.header.stamp = now
        color_img_msg.header.frame_id = self.sensor_frame_name
        self.color_pub.publish(color_img_msg)

        # Load and publish depth image.
        depth_cv_img = PilImage.open(depth_file)
        depth_img_msg = self.cv_bridge.cv2_to_imgmsg(np.array(depth_cv_img), "32FC1")
        depth_img_msg.header.stamp = now
        depth_img_msg.header.frame_id = self.sensor_frame_name
        self.depth_pub.publish(depth_img_msg)
        
        if self.seg_model != "none":
            # Load and publish ID image.
            label_cv_img = cv2.imread(pred_file)
            label_cv_img = cv2.cvtColor(label_cv_img, cv2.COLOR_BGR2RGB)
            label_img_msg = self.cv_bridge.cv2_to_imgmsg(label_cv_img, "rgb8")
            label_img_msg.header.stamp = now
            label_img_msg.header.frame_id = self.sensor_frame_name
            self.id_pub.publish(label_img_msg)

            if self.seg_model == "inst":
                # Load and publish instance masks.
                masks_dict = {}
                with open(mask_file, "rb") as mf:
                    masks_dict = pickle.load(mf)
                    mf.close()
                masks_msg = Masks()
                masks_msg.header.stamp = now
                masks_msg.header.frame_id = self.sensor_frame_name
                masks_msg.image_header = color_img_msg.header
                for mask_dict in masks_dict.values():
                    cls_id, mask = mask_dict["class_id"], mask_dict["mask"]
                    mask_msg = Mask()
                    mask_msg.class_id = cls_id
                    mask_msg.data = self.cv_bridge.cv2_to_imgmsg(mask.astype(np.uint8), "mono8")
                    masks_msg.masks.append(mask_msg)
                vision_packet_msg = HydraVisionPacket()
                vision_packet_msg.color = color_img_msg
                vision_packet_msg.depth = depth_img_msg
                vision_packet_msg.label = label_img_msg
                vision_packet_msg.masks = masks_msg
                self.packet_pub.publish(vision_packet_msg)
                # self.mask_pub.publish(masks_msg)

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
        cam_info_msg.header.stamp = now
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
        self.cam_info_pub.publish(cam_info_msg)

        # Load and publish transform.
        if os.path.isfile(pose_file):
            pose_data = [float(x) for x in open(pose_file, "r").read().split()]
            transform = np.eye(4)
            for row in range(4):
                for col in range(4):
                    transform[row, col] = pose_data[row * 4 + col]
            tf2_msg = TransformStamped()
            tf2_msg.header.stamp = now
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
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = self.global_frame_name
            pose_msg.pose.position.x = pose_data[3]
            pose_msg.pose.position.y = pose_data[7]
            pose_msg.pose.position.z = pose_data[11]
            pose_msg.pose.orientation.x = rotation[0]
            pose_msg.pose.orientation.y = rotation[1]
            pose_msg.pose.orientation.z = rotation[2]
            pose_msg.pose.orientation.w = rotation[3]
            self.pose_pub.publish(pose_msg)

        self.current_index += 1
        if self.current_index > self.max_frames:
            rospy.signal_shutdown(f"Played reached max frames {self.max_frames}")


if __name__ == "__main__":
    rospy.init_node("flat_data_player")
    flat_data_player = FlatDataPlayer()
    rospy.spin()
