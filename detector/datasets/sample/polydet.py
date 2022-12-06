from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg, to_polar_coords, to_cartesian_coords
from utils.image import draw_annotations, to_absolut_ref, to_relative_ref
import math


class PolygonDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        # Get the image
        img = cv2.imread(img_path)
        img_copy = copy.deepcopy(img)
        height, width = img.shape[0], img.shape[1]

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            # Random crop by default
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            # Otherwise scale and shift image
            else:
                sf = self.opt.scale
                cf = self.opt.shift

                # Scale image
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)

                # Shift image
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            # Flip image
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        if self.opt.rotate > 0:  # rotate the image
            if self.opt.rotate == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if self.opt.rotate == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            if self.opt.rotate == 270:
                img = cv2.rotate(img, cv2.img_rotate_90_counterclockwise)

        # Perform affine transformation
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])

        # Warp affine
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp_raw_copy = copy.deepcopy(inp)
        # Scale RGB pixels
        inp = (inp.astype(np.float32) / 255.)

        # Add color augmentation
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        # Es una matriz de 1000 filas por  "num_classes * 2" columnas (2). Es para categorías que tienen un
        # tamaño de bounding box pre definido. Entiendo que aquí se guardará el ancho y el alto, y por eso el "* 2"
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32) #max_objs = 1000 harcodeado en dataset
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        # Add for circle
        cl = np.zeros((self.max_objs, 8), dtype=np.float32) # harcodeo 8 porque es la cantidad de coordenadas necesarias
                                                            # para los 4 vértices
        dense_cl = np.zeros((1, output_h, output_w), dtype=np.float32)
        reg_cl = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_cl = np.zeros((self.max_objs), dtype=np.int64)
        cat_spec_cl = np.zeros((self.max_objs, num_classes * 8), dtype=np.float32) # supongo que son 8 TODO
        cat_spec_clmask = np.zeros((self.max_objs, num_classes * 8), dtype=np.uint8) #supongo que son 8  TODO

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian # en nuestro caso usamos la segunda: draw_umich_gaussian

        gt_det = []
        # For each object in the annotation
        for k in range(num_objs):
            # Get the annotation
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])

            # Debug print statements
            # print(self.cat_ids)
            # print(ann['category_id'])
            # print(int(self.cat_ids[int(ann['category_id'])]))

            cls_id = int(self.cat_ids[int(ann['category_id'])])

            center_point = ann['center']
            vertices_polar = ann['vertices']

            #### debug javier
            # img_1 = draw_annotations(img_copy, center_point, to_absolut_ref(to_cartesian_coords(vertices_polar), center_point), bbox)
            # cv2.imshow('img raw', img_1)
            # cv2.waitKey(0)
            # print(vertices_polar)
            ####

            # If the image was flipped, then flip the annotation
            # print(flipped)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                center_point[0] = width - center_point[0]
                vc = to_cartesian_coords(vertices_polar)
                #vc = to_absolut_ref(vc, center_point)
                vc = [-v if i%2==0 else v for i, v in enumerate(vc)]
                vertices_polar = to_polar_coords(vc)

            #### debug javier
            # img_2 = draw_annotations(img_copy, center_point, to_absolut_ref(to_cartesian_coords(vertices_polar), center_point), bbox)
            # cv2.imshow('img raw', img_2)
            # cv2.waitKey(0)
            # If the image was affine transformed, then transform the annotation
            ####
            # print(self.opt.mse_loss)
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)

            vertices_cart = to_cartesian_coords(vertices_polar)
            # print(f"cart = {vertices_cart}")
            vertices_cart_aff_abs = to_absolut_ref(vertices_cart, center_point)
            # print(f"abs_cart = {vertices_cart_aff_abs}")

            center_point_aff = affine_transform(center_point, trans_output)

            vertices_cart_aff_abs[:2] = affine_transform(vertices_cart_aff_abs[:2], trans_output)
            vertices_cart_aff_abs[2:4] = affine_transform(vertices_cart_aff_abs[2:4], trans_output)
            vertices_cart_aff_abs[4:6] = affine_transform(vertices_cart_aff_abs[4:6], trans_output)
            vertices_cart_aff_abs[6:8] = affine_transform(vertices_cart_aff_abs[6:8], trans_output)
            # cv2.imshow('img', inp_s)
            # cv2.waitKey(0)
            vertices_cart_aff = to_relative_ref(vertices_cart_aff_abs, center_point_aff)
            vertices_polar_aff = to_polar_coords(vertices_cart_aff)

            #### debug javier: para ver las etiquetas sobe la imagen en la resolución de salida
            # img_cp_out = cv2.warpAffine(img, trans_output,
            #                      (output_w, output_h),
            #                      flags=cv2.INTER_LINEAR)
            # img_l = draw_annotations(img_cp_out, center_point_aff, vertices_cart_aff_abs, bbox)
            # cv2.imshow('img', img_l)
            # cv2.waitKey(0)
            ####
            #### debug javier: para ver si la ida y vuelta de relativo y tipo de coordenadas funciona bien
            # vertices_debug = to_absolut_ref(to_cartesian_coords(vertices_polar_aff),center_point_aff)
            # img_cp_out = cv2.warpAffine(img, trans_output,
            #                      (output_w, output_h),
            #                      flags=cv2.INTER_LINEAR)
            # img_l = draw_annotations(img_cp_out, center_point_aff, vertices_debug, bbox)
            # cv2.imshow('img', img_l)
            # cv2.waitKey(0)
            ####

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0 and center_point_aff[0]>0 \
                    and center_point_aff[1]>0 and center_point_aff[0]<output_w\
                    and center_point_aff[1]<output_h:

                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                #
                ct_int = ct.astype(np.int32)
                # # draw_gaussian(hm[cls_id], ct_int, radius)
                # wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                # reg[k] = ct - ct_int

                # cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                # cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                # if self.opt.dense_wh:
                #     draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                # gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                #                ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
                if self.opt.ez_guassian_radius:
                    vertices = to_polar_coords(vertices_cart_aff)
                else:
                    vertices = [gaussian_radius((math.ceil(ver*2), math.ceil(ver*2))) for ver in to_polar_coords(vertices_cart_aff)]# TODO no entiendo que hace gaussian_radius
                # print(f"vertices: {vertices}") # estos vertices son el resultado de gausian radius sobre los vértices

                vertices = [max(0, ver) for ver in vertices]
                radius_mean = int(round(np.mean(vertices[::2]))) # esta línea la puse yo porque es el radio del heatmap en base a la media de los radios
                # vertices = [self.opt.hm_gauss if self.opt.mse_loss else vertex for vertex in vertices] #hm_gauss no existe, opt.mse_loss debe ser falso
                # la línea anterior no se usa para nada al final.
                cp = center_point_aff
                cp_int = cp.astype(np.int32)
                draw_gaussian(hm[cls_id], cp_int, radius_mean)
                ind_cl[k] = cp_int[1] * output_w + cp_int[0]
                reg_cl[k] = cp - cp_int
                reg_mask[k] = 1
                cr = vertices_polar_aff
                # print(type(vertices_polar_aff)) # list
                cl[k, :] = [1. * c for c  in cr]
                # print(cl[k], type(cl))
                # cat_spec_cl[k, cls_id * 1: cls_id * 1 + 1] = cl[k] #las comento porque al parecer no influyen
                # cat_spec_clmask[k, cls_id * 1: cls_id * 1 + 1] = 1
                if self.opt.filter_boarder:
                    if cp[0] - cr < 0 or cp[0] + cr > output_w:
                        continue
                    if cp[1] - cr < 0 or cp[1] + cr > output_h:
                        continue
                gt_det.append([cp[0], cp[1], *cr, 1, cls_id])

                # if ind_cl[0]<0:
                #     aaa = 1
                #
                # print('ind')
                # print(ind[0:10])
                # print('ind_cl')
                # print(ind_cl[0:10])

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind_cl, 'cl': cl}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg_cl})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 12), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret