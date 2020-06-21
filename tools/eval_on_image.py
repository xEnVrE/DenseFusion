import argparse
import cv2
import numpy
import os
import struct
import sys

sys.path.insert(0, os.getcwd())
from lib.inference import Inference

def read_rgb(file_path):
    return cv2.imread(file_path)    

def read_depth(file_path):
    depth_file = open(file_path, "rb")

    # read width and height
    [width, height] = struct.unpack('QQ', depth_file.read(16))

    # read depth
    data = numpy.fromfile(depth_file, dtype='float32', offset=0)
    depth = data.reshape([height, width])

    return depth

def read_mask(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default = '')
    parser.add_argument('--object_id', type=int, default = '')    
    parser.add_argument('--model', type=str, default = '')
    parser.add_argument('--refine_model', type=str, default = '')
    parser.add_argument('--width', type=int, default = '')
    parser.add_argument('--height', type=int, default = '')
    parser.add_argument('--cam_fx', type=float, default = '')
    parser.add_argument('--cam_fy', type=float, default = '')
    parser.add_argument('--cam_cx', type=float, default = '')
    parser.add_argument('--cam_cy', type=float, default = '')    
    opt = parser.parse_args()
    
    rgb = read_rgb(opt.path + "/rgb.png")
    depth = read_depth(opt.path + "/depth.float")
    mask = read_mask(opt.path + "/mask.png")

    inference = Inference(opt.model, opt.refine_model, opt.width, opt.height, opt.cam_fx, opt.cam_fy, opt.cam_cx, opt.cam_cy)
    prediction = inference.inference(opt.object_id, rgb, depth, mask)

    print("Inferece completed.")
    print("Prediction:")
    print(prediction)

if __name__ == '__main__':
    main()
