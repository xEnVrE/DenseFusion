import argparse
import numpy
import os
import sys
import time
import yarp

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

        # Inumpyut buffers initialization
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

    def close(self):

        self.rgb_in.close()
        self.depth_in.close()
        self.mask_in.close()
        self.prediction_out.close()

        return True

    def getPeriod(self):

        return 1.0 / self.options.fps

    def updateModule(self):

        start_time = time.time()
        
        rgb = self.rgb_in.read(False)
        depth = self.depth_in.read(False)
        mask = self.mask_in.read(False)

        if (rgb is not None) and (depth is not None) and (mask is not None):

            self.rgb_image.copy(rgb)
            rgb_frame = numpy.frombuffer(self.rgb_buffer, dtype=numpy.uint8).reshape(self.options.height, self.options.width, 3)

            self.depth_image.copy(depth)
            depth_frame = numpy.frombuffer(self.depth_buffer, dtype=numpy.float32).reshape(self.options.height, self.options.width)

            self.mask_image.copy(mask)
            mask_frame = numpy.frombuffer(self.mask_buffer, dtype=numpy.uint8).reshape(self.options.height, self.options.width)

            prediction = self.inference.inference(self.options.object_id, rgb_frame, depth_frame, mask_frame)

            if len(prediction) != 0:
                output_vector = yarp.Vector()
                output_vector.resize(7)
                output_vector[0] = prediction[0]
                output_vector[1] = prediction[1]
                output_vector[2] = prediction[2]
                output_vector[3] = prediction[3]
                output_vector[4] = prediction[4]
                output_vector[5] = prediction[5]
                output_vector[6] = prediction[6]

                self.prediction_out.write(output_vector)
            
            print(1.0 / (time.time() - start_time))

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

