import numpy as np
import cv2
from .config import cfg
from ..utils.blob import im_list_to_blob

def _get_image_blob(im):
    """数据预处理
    (means subtracted, BGR order, ...).
    """
    im_orig = im.astype(np.float32, copy = True)
    # 按三通道的均值统一做均值化
    im_orig = -= cfg.PIXEL_MEANS

    im_shape = im_orig.im_shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    # 将图片高和宽最大放缩相同的倍数到目标的size
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

        # 将输入图片的列表转换格式作为网络的输入
        # [batch_size, img_height, img_width, channels]
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    """得到网络的输入"""
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors =  _get_image_blob(im)

    return blobs, im_scale_factors

def test_ctpn(sess, net, im, boxes = None):
    blobs, im_scales = _get_blobs(im, boxes)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype = np.float32)
    # 前向传播
    # 测试环境下把dropout关闭
    if cfg.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    rois = sess.run([net.get_output('rois')[0]], feed_dict = feed_dict)
    # net.get_output('rois')输出是rpn_rois, rpn_targets
    rois = rois[0]

    scores = rois[:, 0] # ?
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1: 5] / im_scales[0]
    return scores, boxes
