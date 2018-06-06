import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class VGGnet_train(Network):
    """docstring for VGGnet_train
    VGGnet_train实例化的时候，会传入一个Network参数，这个Network参数相当于是VGGnet_train的子类
    """
    def __init__(self, trainable = True):
        self.trainable = trainable
        self.data = tf.placeholder(tf.float32, shape = [None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape = [None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape = [None, 5])
        # 该参数定义dropout比例
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        self.trainable = trainable
        # 子类的方法重载了父类的方法
        # 调用子类的方法
        self.setup()

        # create ops and placeholders for bbox normalization process
        #建立weights,biases变量，用tf.assign来更新
        with tf.variable_scope('bbox_pred', reuse = True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape = weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape = biases.get_shape())
            # tf.assign用来更新参数值
            self.bbox_weights_assign = weights.assgin(self.bbox_weights)
            self.bbox_bias_assign = biases.assgin(self.bbox_biases)

    def setup(self):
        anchor_scales =cfg.AHCHOR_SCALES
        _feat_stride = [16, ]
        #feed就是从self.layers中的信息提取出来（包括self.data,self.im_info,self.gt_boxes）存入self.input
        #feed返回的是self，为conv第一个参数,conv参数（self,卷积核高，宽，深度，stride高，宽）
        #return self 用法：
        #class human（object）
        #   def __init__(self,weight):
        #       self.weight=weight
        #   def get_weight(self):
        #       return self.weight
        #想要调用get_weight,直接用human.get_weight(45)会告知调用之前要先实例化，weight为未绑定函数
        #而   human.get_weight(human(45))就可以正常输出，说明human(45)将get_weight绑定了
        #其实human（45）作为一个self传给get_weight（）
        #conv:卷积高、宽、输出深度、步长高、宽
        #VGG层
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name = 'conv1_1')
             .conv(3, 3, 64, 1, 1, name = 'conv1_2')
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool1')
             .conv(3, 3, 128, 1, 1, name = 'conv2_1')
             .conv(3, 3, 128, 1, 1, name = 'conv2_2')
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool2')
             .conv(3, 3, 256, 1, 1, name = 'conv3_1')
             .conv(3, 3, 256, 1, 1, name = 'conv3_2')
             .conv(3, 3, 256, 1, 1, name = 'conv3_3')
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool3')
             .conv(3, 3, 512, 1, 1, name = 'conv4_1')
             .conv(3, 3, 512, 1, 1, name = 'conv4_2')
             .conv(3, 3, 512, 1, 1, name = 'conv4_3')
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool4')
             .conv(3, 3, 512, 1, 1, name = 'conv5_1')
             .conv(3, 3, 512, 1, 1, name = 'conv5_2')
             .conv(3, 3, 512, 1, 1, name = 'conv5_3')
            )

        (self.feed('conv5_3')
             .conv(3, 3, 512, 1, 1, name = 'rpn_conv/3x3')
            )

        # bilstm层
        (self.feed('rpn_conv/3x3')
             .Bilstm(512, 128, 512, name = 'lstm_o')
            )
        # 和faster rcnn的区别是faster rcnn用1x1的卷积来做fc层
        # bilstm后面跟着anchor的bbox回归层
        (self.feed('lstm_o')
             .lstm_fc(512, len(anchor_scales) * 10 * 4, name = 'rpn_bbox_pred')
            )
        # bilstm后面跟着anchor的cls分类层
        (self.feed('lstm_o')
             .lstm_fc(512, len(anchor_scales) * 10 * 2, name = 'rpn_cls_score')
            )

        # 和faster RCNN一样做softmax
        # reshape->softmax->reshape
        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_softmax(name = 'rpn_cls_prob')
            )

        # 再reshape回去
        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
             .spatial_reshape_layer(len(anchor_scales) * 10 * 2, name = 'rpn_cls_prob_reshape')
            )

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TEST', name = 'rois')
            )






        
