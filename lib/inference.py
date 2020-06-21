import copy
import cv2
import numpy
import numpy.ma as numpy_ma
import torch
import torchvision.transforms as transforms
from lib.network import PoseNet, PoseRefineNet
from torch.autograd import Variable
from transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix


class Inference:
    def __init__(self, model, model_with_refinement, width, height, fx, fy, cx, cy, number_points = 1000, number_iterations = 2):

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.number_iterations = number_iterations

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

        self.xmap = numpy.array([[j for i in range(self.width)] for j in range(self.height)])
        self.ymap = numpy.array([[i for i in range(self.width)] for j in range(self.height)])
        self.number_points = number_points

        self.estimator = PoseNet(num_points = self.number_points, num_obj = 21)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(model))
        self.estimator.eval()

        self.refiner = PoseRefineNet(num_points = self.number_points, num_obj = 21)
        self.refiner.cuda()
        self.refiner.load_state_dict(torch.load(model_with_refinement))
        self.refiner.eval()

    def evaluate_bbox(self, mask):
        points = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(points)
        rmin = y
        rmax = y + h
        cmin = x
        cmax = x + w

        r_b = rmax - rmin
        for tt in range(len(self.border_list)):
            if r_b > self.border_list[tt] and r_b < self.border_list[tt + 1]:
                r_b = self.border_list[tt + 1]
                break
            c_b = cmax - cmin
        for tt in range(len(self.border_list)):
            if c_b > self.border_list[tt] and c_b < self.border_list[tt + 1]:
                c_b = self.border_list[tt + 1]
                break
            center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
            rmin = center[0] - int(r_b / 2)
            rmax = center[0] + int(r_b / 2)
            cmin = center[1] - int(c_b / 2)
            cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > self.height:
            delt = rmax - self.height
            rmax = self.height
            rmin -= delt
        if cmax > self.width:
            delt = cmax - self.width
            cmax = self.width
            cmin -= delt

        return rmin, rmax, cmin, cmax

    def inference(self, object_id, rgb, depth, mask):
        try:
            rmin, rmax, cmin, cmax = self.evaluate_bbox(mask)
    
            mask_depth = numpy_ma.getmaskarray(numpy_ma.masked_not_equal(depth, 0))
            mask_label = (mask==255)
            mask = mask_label * mask_depth
    
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > self.number_points:
                c_mask = numpy.zeros(len(choose), dtype=int)
                c_mask[:self.number_points] = 1
                numpy.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = numpy.pad(choose, (0, self.number_points - len(choose)), 'wrap')
    
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, numpy.newaxis].astype(numpy.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, numpy.newaxis].astype(numpy.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, numpy.newaxis].astype(numpy.float32)
            choose = numpy.array([choose])
    
            pt0 = (ymap_masked - self.cx) * depth_masked / self.fx
            pt1 = (xmap_masked - self.cy) * depth_masked / self.fy
            cloud = numpy.concatenate((pt0, pt1, depth_masked), axis=1)
            
            rgb_masked = numpy.transpose(rgb, (2, 0, 1))
            rgb_masked = rgb_masked[:, rmin:rmax, cmin:cmax]
    
            cloud = torch.from_numpy(cloud.astype(numpy.float32))
            choose = torch.LongTensor(choose.astype(numpy.int32))
            rgb_masked = self.norm(torch.from_numpy(rgb_masked.astype(numpy.float32)))
            index = torch.LongTensor([object_id - 1])
    
            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            rgb_masked = Variable(rgb_masked).cuda()
            index = Variable(index).cuda()
    
            cloud = cloud.view(1, self.number_points, 3)
            rgb_masked = rgb_masked.view(1, 3, rgb_masked.size()[1], rgb_masked.size()[2])
    
            pred_r, pred_t, pred_c, emb = self.estimator(rgb_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.number_points, 1)
    
            pred_c = pred_c.view(1, self.number_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(1 * self.number_points, 1, 3)
            points = cloud.view(1 * self.number_points, 1, 3)
    
            rotation = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            translation = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    
            for ite in range(0, self.number_iterations):
                T = Variable(torch.from_numpy(translation.astype(numpy.float32))).cuda().view(1, 3).repeat(self.number_points, 1).contiguous().view(1, self.number_points, 3)
                mat = quaternion_matrix(rotation)
                R = Variable(torch.from_numpy(mat[:3, :3].astype(numpy.float32))).cuda().view(1, 3, 3)
                mat[0:3, 3] = translation
    
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = self.refiner(new_cloud, emb, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                rotation_2 = pred_r.view(-1).cpu().data.numpy()
                translation_2 = pred_t.view(-1).cpu().data.numpy()
                mat_2 = quaternion_matrix(rotation_2)
    
                mat_2[0:3, 3] = translation_2
    
                mat_final = numpy.dot(mat, mat_2)
                rotation_final = copy.deepcopy(mat_final)
                rotation_final[0:3, 3] = 0
                rotation_final = quaternion_from_matrix(rotation_final, True)
                translation_final = numpy.array([mat_final[0][3], mat_final[1][3], mat_final[2][3]])
    
                rotation = rotation_final
                translation = translation_final
    
            return numpy.append(translation, rotation)
        except ValueError:
            print("Inference::inference. Error: an error occured at inference time.")
            return []
