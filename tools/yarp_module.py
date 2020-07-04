import argparse
import math
import numpy
import os
import sys
import time
import yarp
from pyquaternion import Quaternion

sys.path.insert(0, os.getcwd())
from lib.inference import Inference


class InferenceModule (yarp.RFModule):

    def __init__(self, options):

        self.options = options

        # Set requested GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_id)

        # Initialize YARP network
        yarp.Network.init()

        # Initialize RF module
        yarp.RFModule.__init__(self)

        # Initialize inference
        self.inference = Inference(self.options.model, self.options.refine_model, self.options.width, self.options.height, self.options.cam_fx, self.options.cam_fy, self.options.cam_cx, self.options.cam_cy)

        # Initialize YARP ports
        self.rgb_in = yarp.BufferedPortImageRgb()
        self.rgb_in.open("/dense-fusion/rgb:i")

        self.depth_in = yarp.BufferedPortImageFloat()
        self.depth_in.open("/dense-fusion/depth:i")

        self.mask_in = yarp.BufferedPortImageMono()
        self.mask_in.open("/dense-fusion/mask:i")

        self.prediction_out = yarp.Port()
        self.prediction_out.open("/dense-fusion/pose:o")

        # Initialize gaze controller
        props = yarp.Property()
        props.put("device", "gazecontrollerclient")
        props.put("local", "/densefusion/gaze")
        props.put("remote", "/iKinGazeCtrl")
        self.gaze_driver = yarp.PolyDriver(props)
        self.gaze = self.gaze_driver.viewIGazeControl()

        # Input buffers initialization
        self.rgb_buffer = bytearray(numpy.zeros((self.options.height, self.options.width, 3), dtype = numpy.uint8))
        self.rgb_image = yarp.ImageRgb()
        self.rgb_image.resize(self.options.width, self.options.height)
        self.rgb_image.setExternal(self.rgb_buffer, self.options.width, self.options.height)

        self.depth_buffer = bytearray(numpy.zeros((self.options.height, self.options.width, 1), dtype = numpy.float32))
        self.depth_image = yarp.ImageFloat()
        self.depth_image.resize(self.options.width, self.options.height)
        self.depth_image.setExternal(self.depth_buffer, self.options.width, self.options.height)

        self.mask_buffer = bytearray(numpy.zeros((self.options.height, self.options.width, 1), dtype = numpy.uint8))
        self.mask_image = yarp.ImageMono()
        self.mask_image.resize(self.options.width, self.options.height)
        self.mask_image.setExternal(self.mask_buffer, self.options.width, self.options.height)

        # Stamp initialization for finite differences
        self.last_stamp = -1
        self.last_total_position = None

    def close(self):

        self.rgb_in.close()
        self.depth_in.close()
        self.mask_in.close()
        self.prediction_out.close()

        return True

    def getPeriod(self):

        return 1.0 / self.options.fps

    def updateModule(self):

        # synchronize on rgb stream
        rgb = self.rgb_in.read(True)
        depth = self.depth_in.read(False)
        mask = self.mask_in.read(False)

        if (rgb is not None) and (depth is not None) and (mask is not None):

            eye_position = yarp.Vector()
            eye_orientation = yarp.Vector()
            self.gaze.getLeftEyePose(eye_position, eye_orientation)
            camera_position = numpy.array([eye_position[0], eye_position[1], eye_position[2]])
            camera_axis = numpy.array([eye_orientation[0], eye_orientation[1], eye_orientation[2]])
            camera_angle = eye_orientation[3]

            self.rgb_image.copy(rgb)
            rgb_frame = numpy.frombuffer(self.rgb_buffer, dtype=numpy.uint8).reshape(self.options.height, self.options.width, 3)

            self.depth_image.copy(depth)
            depth_frame = numpy.frombuffer(self.depth_buffer, dtype=numpy.float32).reshape(self.options.height, self.options.width)

            self.mask_image.copy(mask)
            mask_frame = numpy.frombuffer(self.mask_buffer, dtype=numpy.uint8).reshape(self.options.height, self.options.width)

            prediction = self.inference.inference(self.options.object_id, rgb_frame, depth_frame, mask_frame)

            if len(prediction) != 0:

                object_orientation = Quaternion(axis=[prediction[3], prediction[4], prediction[5]], angle = prediction[6])
                camera_orientation = Quaternion(axis=camera_axis, angle=camera_angle).rotation_matrix
                total_orientation =  camera_orientation.dot(object_orientation.rotation_matrix)
                total_orientation_axis = Quaternion(matrix=total_orientation).axis
                total_orientation_angle = Quaternion(matrix=total_orientation).angle
                total_position = camera_position + camera_orientation.dot(prediction[0:3])

                stamp = yarp.Stamp()
                self.rgb_in.getEnvelope(stamp)
                new_stamp = stamp.getTime()

                velocity = [0, 0, 0, 0, 0, 0]
                if self.last_stamp > 0:
                    elapsed =  new_stamp - self.last_stamp
                    velocity[0:3] = (total_position - self.last_total_position) / elapsed

                    relative_orientation = total_orientation.dot(self.last_total_orientation.transpose())
                    acos_argument = (relative_orientation.trace() - 1) / 2.0
                    if acos_argument < -1.0:
                        acos_argument = -1.0
                    else:
                        acos_argument = 1.0
                    theta = math.acos(acos_argument) + sys.float_info.min
                    ang_velocity_skew = 1.0 / (2.0 * elapsed) * theta / math.sin(theta) * (relative_orientation - relative_orientation.transpose())
                    velocity[3:6] = [-ang_velocity_skew[1, 2], ang_velocity_skew[0, 2], -ang_velocity_skew[0, 1]]

                self.last_stamp = new_stamp
                self.last_total_position = total_position
                self.last_total_orientation = total_orientation

                output_vector = yarp.Vector()
                output_vector.resize(13)
                output_vector[0] = total_position[0]
                output_vector[1] = total_position[1]
                output_vector[2] = total_position[2]
                output_vector[3] = total_orientation_axis[0]
                output_vector[4] = total_orientation_axis[1]
                output_vector[5] = total_orientation_axis[2]
                output_vector[6] = total_orientation_angle
                output_vector[7] = velocity[0]
                output_vector[8] = velocity[1]
                output_vector[9] = velocity[2]
                output_vector[10] = velocity[3]
                output_vector[11] = velocity[4]
                output_vector[12] = velocity[5]

                self.prediction_out.write(output_vector)

            # print(1.0 / (time.time() - start_time))

        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_id', type=int, default = '')
    parser.add_argument('--model', type=str, default = '')
    parser.add_argument('--refine_model', type=str, default = '')
    parser.add_argument('--width', type=int, default = '')
    parser.add_argument('--height', type=int, default = '')
    parser.add_argument('--cam_fx', type=float, default = '')
    parser.add_argument('--cam_fy', type=float, default = '')
    parser.add_argument('--cam_cx', type=float, default = '')
    parser.add_argument('--cam_cy', type=float, default = '')
    parser.add_argument('--gpu_id', type=int, default = '')
    parser.add_argument('--fps', type=int, default = '')

    options = parser.parse_args()

    module = InferenceModule(options)
    module.runModule()

