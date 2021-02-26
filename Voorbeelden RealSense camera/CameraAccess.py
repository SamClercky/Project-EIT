# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import math as m

# Create the pipeline which handles all connected realsense devices
pipeline = rs.pipeline()

# Create a configuration object
config = rs.config()

# Enable some sensors
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)          # Depth map uint16
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)         # Color map uint8 (x3 channels)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)     # Infrared left map uint8
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)     # Infrared right map uint8

# Start the pipeline
profile = pipeline.start(config)

try:
    while True:
        # Capture temporal synchronized device data
        frames = pipeline.wait_for_frames()

        # Get the needed frames form the frames pipeline
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)

        # Create a numpy array images
        # A numpy array can be constructed using this protocol with no data marshalling overhead:
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())

        # Apply colormap to depth data (optional)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show both images in windows
        cv2.imshow("Depth Image", depth_colormap)
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Left IR Image", ir_left_image)
        cv2.imshow("Right IR Image", ir_right_image)

        cv2.waitKey(1)

finally:
    pipeline.stop()


