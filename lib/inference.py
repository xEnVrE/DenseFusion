import copy
import cv2
import numpy
import numpy.ma as numpy_ma
import torch
import torchvision.transforms as transforms
from lib.network import PoseNet, PoseRefineNet
from torch.autograd import Variable
from transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from pyquaternion import Quaternion


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

    def inference(self, object_id, img, depth, mask):
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

            pt2 = depth_masked
            pt0 = (ymap_masked - self.cx) * pt2 / self.fx
            pt1 = (xmap_masked - self.cy) * pt2 / self.fy
            cloud = numpy.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = numpy.array(img)[:, :, :3]
            img_masked = numpy.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(numpy.float32))
            choose = torch.LongTensor(choose.astype(numpy.int32))
            img_masked = self.norm(torch.from_numpy(img_masked.astype(numpy.float32)))
            index = torch.LongTensor([object_id - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, self.number_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.number_points, 1)

            pred_c = pred_c.view(1, self.number_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(1 * self.number_points, 1, 3)
            points = cloud.view(1 * self.number_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = numpy.append(my_r, my_t)

            for ite in range(0, self.number_iterations):
                T = Variable(torch.from_numpy(my_t.astype(numpy.float32))).cuda().view(1, 3).repeat(self.number_points, 1).contiguous().view(1, self.number_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(numpy.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = self.refiner(new_cloud, emb, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)

                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = numpy.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final_copy = copy.deepcopy(my_r_final)
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = numpy.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = numpy.append(my_r_final, my_t_final)
                my_r_final_alt = Quaternion(matrix=my_r_final_copy[0:3, 0:3])
                my_r_final_alt_aa = numpy.append(my_r_final_alt.axis.squeeze(), my_r_final_alt.angle)
                my_pred_alt = numpy.append(my_t_final, my_r_final_alt_aa)
                my_r = my_r_final
                my_t = my_t_final

            return my_pred_alt
        except ValueError:
            print("Inference::inference. Error: an error occured at inference time.")
            return []
