import numpy as np
import tensorflow as tf
from ..fast_rcnn.config import cfg
from ..rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from ..rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py


DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
    """
    # Automatically set a name if not provided.
        #op.__name__的是各个操作函数名，如conv、max_pool
        #get_unique_name返回类似与conv_4，以name：'conv_4'存在kwargs字典
    """
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        #此情况说明刚有输入层，即取输入数据即可
        elif len(self.inputs) == 1
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        #开始做卷积，做pool操作！！！！正式开始做操作的是这里，而不是函数定义，会发现下面函数定义中与所给参数个数不符合，原因在于input没给定
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        #在self.layer中添加该name操作信息
        self.feed(layer_output)
        # Return self for chained calls
        return self

    return layer_decorated


class Network(object):
    """docstring for Network"""
    def __init__(self, inputs, trainable = True):
        self.inputs = inputs
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        """该类只能作为子类调用"""
        raise NotImplementedError('Must be subclassed.')
        
    def load(self, data_path, session, ignore_missing = False
        ):
        """读取npy数据格式文件"""
        # np.load()读取的是一个只有一个元素的列表
        # item()将之取出
        data_dict = np.load(data_path, encoding = 'latin1').item()
        for key in data_dict:
            with tf.variable_scope(key, reuse = True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except Exception as e:
                        print("ignore "+key)
                        if not ignore_missing:
                            raise e

    def feed(self, *args):
        """从本类中要取出的层
        #*args中存的是多余的变量，且无标签，存在tuple中，如果有标签，则需要将函数改为feed(self, *args，**kwargs):
        **kwargs为一个dict
        layers为一个dict，inputs为一个list
        """
        assert len(args) != 0
        self.inputs = []
        for layer in layers:
            if isinstance(layer, str):
                #self.layers在VGGnet_train 重载，为一个有值的dict
                try:
                    #将layer改为真实的variable，虽然目前还只是数据流图的一部分，还没有真正的开始运作
                    layer = self.layers[layer]
                    print(layer)
                except Exception as e:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s'%layer)
            #将取出的layer数据存入input列表
            self.inputs.append(layer)

        return self

    def get_output(self, layer):
        """从self.layers中取出相应的层
        self.layers在VGGnet_train.py中重载了，为一个dict，记录的是每一层的输出
        """
        try:
            layer = self.layers[layer]
        except Exception as e:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        """得到唯一的名字，prefix传回来的是conv、max_pool..
        self.layers为一个dict，item将其转换为可迭代形式""
        """
        # startswith() 方法用于检查字符串是否是以指定子字符串开头，返回true与false
        # #即查看有没有conv开头的key，记录有的个数（true），相加再加1为id
        id = sum(t.startswith(prefix) for t, _ in list(self.layers.items())) + 1
        #返回的就是类似与conv_4
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer = None, trainable = True, regularizer=None):
        """此函数就是在tensorflow格式下建立变量"""
        return tf.get_variable(name, shape, initializer = initializer, trainable = trainable)

    def validate_padding(self, padding):
        """判断padding类型是否符合要求"""
        assert padding in ('SAME', 'VALID')

    @layer
    #就因为上面的属性函数，是的真正的conv操作没有在这里进行，而是在上面的layer函数中进行
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased = True, relu = True, padding = DEFAULT_PADDING, trainable = True):
        # 判断padding是否为same与valid的一种
        self.validate_padding(padding)
        #shape最后一位为深度
        #input形状为[batch, in_height, in_width, in_channels]
        #c_i.c_o分别为输入激活图层的深度，与输入激活图层深度，即卷积核个数
        #get_shape是对tensorflow对象取shape
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding = padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev = 0.01)
            init_biases = tf.constant_initializer(0.0)
            # 相比于faster rcnn开启了正则化配置
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, regularizer = self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))

            if biasesd:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias, name = scope.name)
                return tf.nn.bias_add(conv, biases, name = scope.name)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv, name = scope.name)
                return conv

    @layer
    def Bilstm(self, input, d_i, d_h, d_o, name, trainable):
        """
        rpn_conv/3x3后面紧接着一个bilstm
        d_i:lstm输入中每个lstm单元输入的特征长度
        d_h:lstm单元隐藏层单元数目
        d_o:lstm单元fc输出层fc层单元数目（检测目标数目）
        """
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            # 将每个batch的img的每行叠加起来
            # lstm的输入shape:[n*h, w, c]
            # 也就是说每个lstm单元的输入是feature maps的深度方向：特征纤维
            # [batch_size,max_time_step,num_features]
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])

            # d_h是lstm单元的隐藏层的单元数目
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple = True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple = True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img, dtype = tf.float32)

            # 将两层的lstm的输出结果concat连接起来
            lstm_out = tf.concat(lstm_out, axis = -1)
            lstm_out = tf.reshape(lstm_out, [N * H * W], 2 * d_h)

            init_weights = tf.truncated_normal_initializer(stddev = 0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [2 * d_h, d_o], init_weights, trainable, regularizer = self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases

            outputs = tf.reshape(outputs, [N, H, W, d_o])
            return outputs

    @layer
    def lstm(self, input, d_i, d_h, d_o, name, trainable = True):
        """
        rpn_conv/3x3后面紧接着一个bilstm
        d_i:lstm输入中每个lstm单元输入的特征长度
        d_h:lstm单元隐藏层单元数目
        d_o:lstm单元fc输出层fc层单元数目（检测目标数目）
        """
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])

            lstm_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple = True)
            # 初始化lstm网络的状态
            # zero_state(batchsize)
            # batch_size = N * H
            initial_state = lstm_cell.zero_state(N * H, dtype = tf.float32)

            lstm_out, last_state = tf.nn.dynamic_rnn(lstm_cell, img, initial_state = initial_state, dtype = tf.float32)

            lstm_out = tf.reshape(lstm_out, [N * H * W, d_h])

            init_weights = tf.truncated_normal_initializer(stddev = 0.1)
            init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [d_h, d_o], init_weights, trainable, regularizer = self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases

            outputs = tf.reshape(outputs, [N, H, W, d_o])

            return outputs

    @layer
    def lstm_fc(self, input, d_i, d_o, name, trainable = True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            input = tf.reshape(input, [N * H * W, C])

            init_weights = tf.truncated_normal_initializer(stddev = 0.1)
            init_biases = tf.constant_initializer(0.0)

            kernel = self.make_var('weights', [d_i, d_o], init_weights, trainable, regularizer = self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)

            _O = tf.matmul(input, kernel) + biases

            return tf.reshape(_O, [N, H, W, int(d_o)])

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input, [input_shape[0], input_shape[1], -1, int(d)])

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        """
        对应proposal层：结合anchor_pred_reg和fg anchors输出proposals，并进一步剔除无效anchors
        """
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0]是anchors_cls_prob
            # shape:[1, H, W, Ax2]
            # proposal_layer_py的输出:blob, bbox_delta
            # blob: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
            # bbox_delta: [dx, dy, dw, dh]
            # 这里假装返回的blob是(1 x H x W x A, 6) [0, score,x1, y1, x2, y2]
        with tf.variable_scope(name) as scope:
            blob, bbox_delta = tf.py_func(proposal_layer_py, [input[0], input[1], input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32, tf.float32])
            rpn_rois = tf.convert_to_tensor(tf.reshape(blob, [-1, 5]), name = 'rpn_rois')
            rpn_targets = tf.convert_to_tensor(bbox_delta, name = 'rpn_targets')
            self.layers['rpn_rois'] = rpn_rois
            self.layers['rpn_targets'] = rpn_targets

            return rpn_rois, rpn_targets
        

    @layer
    def l2_regularizer(self, weight_decay = 0.0005, scope = None):
        """
        参数正则化
        """
        def regularizer(tensor):
            with tf.name_scope(scope, default_name = 'l2_regularizer', values = [tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay, dtype = tensor.dtype.base_dtype, name = 'weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name = 'value')
        return regularizer

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name = name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding = DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input, ksize = [1, k_h, k_w, 1], strides = [1, s_h, s_w, 1], padding = padding, name = name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding = DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input, ksize = [1, k_h, k_w, 1], strides = [1, s_h, s_w, 1], padding = padding, name = name)

    

    