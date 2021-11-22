from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation

from data import cfg, set_cfg

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
from collections import defaultdict
from PIL import ImageColor

import style_utils

import cv2

import transformer

path = "weights/transformer_weight_"
train_model = "weights/yolact_plus_base_140_400000.pth"
TOP_K = 1
CUDA = True
config = None

def style_transfer(img):
    print("Loading Transformer Network")
    net_style = transformer.TransformerNetwork()
    net_style.load_state_dict(torch.load(style_transform_path))
    print("Done Loading Transformer Network")
    net_style = net_style.cuda()
    content_tensor = style_utils.itot(img).cuda()
    #print(content_tensor.shape)
    generated_tensor = net_style(content_tensor)
    #print(generated_tensor.shape)
    generated_image = style_utils.ttoi(generated_tensor.detach())
    #print(generated_image.shape)
    generated_image = generated_image / 255
    return generated_image

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, style: str="undefined", color: str="undefined", undo_transform=True):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = False,
                                        score_threshold   = 0.8)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:TOP_K]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(TOP_K, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < 0.8:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if cfg.eval_mask_branch and num_dets_to_consider > 0:
        masks = masks[:num_dets_to_consider, :, :, None]
        if style != "undefined":
        # After this, mask is of size [num_dets, h, w, 1]
            style_image = torch.Tensor(style_transfer(img.detach().cpu().numpy())).cuda()
            style_image = masks[0] * style_image
            img_gpu = style_image + (1 - masks[0]) * img_gpu
        if color != "undefined":
            one_color = ImageColor.getcolor(color, "RGB")
            one_color = torch.Tensor(one_color) / 255.
            one_rgb = torch.flip(one_color, [0])
            img_gray = torch.Tensor(cv2.cvtColor(img_gpu.cpu().numpy(), cv2.COLOR_BGR2GRAY)).cuda()
            masks_cloth = masks[0] * one_rgb * img_gray.repeat(3, 1, 1).permute(1, 2, 0)
            img_gpu = masks_cloth + (1- masks[0]) * img_gpu
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    # img_numpy = (img_gpu * 255).byte().cpu().numpy()
    img_numpy = (img_gpu * 255).cpu().numpy()

    if num_dets_to_consider == 0:
        return img_numpy
   
    
    return img_numpy

def evalimage(net:Yolact, path:str, style: str="undefined", color: str="undefined", save_path:str="./static/output_img.jpg"):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, style, color, undo_transform=False)
    
    cv2.imwrite(save_path, img_numpy)

def evaluate(net:Yolact, img, style: str="undefined", color: str="undefined"):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = True
    cfg.mask_proto_debug = False

    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    evalimage(net, img, style, color)
    return

def main(img, style: str="undefined", color: str="undefined"):
    global style_transform_path
    global train_model
    global config
    if config is not None:
        set_cfg(config)

    if train_model == 'interrupt':
        train_model = SavePath.get_interrupt('weights/')
    elif train_model == 'latest':
        train_model = SavePath.get_latest('weights/', cfg.name)

    if config is None:
        model_path = SavePath.from_str(train_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % config)
        set_cfg(config)

    if style != "undefined":
        style_transform_path = path + style + ".pth"
    
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if CUDA:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
      

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(train_model)
        net.eval()
        print(' Done.')
        
        if CUDA:
            net = net.cuda()

        evaluate(net, img, style, color)

if __name__ == '__main__':
    main()