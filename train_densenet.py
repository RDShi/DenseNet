import tensorflow as tf
import tflearn
import cPickle
import numpy as np
import sys
import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"

def unpickle(fid):
    with open(fid, 'rb') as fo:
        data = cPickle.load(fo)
    return data

def bottleneck_layer(x, filters, scope, keep_prob=1):
    with tf.name_scope(scope):
        x = tflearn.batch_normalization(x, scope=scope+'_batch1')
        x = tf.nn.relu(x)
        x = tflearn.conv_2d(x, nb_filter=4*filters, filter_size=1, strides=1, padding='same', 
                            activation='linear', bias=False, scope=scope+'_conv1',
                            regularizer='L2', weight_decay=1e-4)
        x = tflearn.dropout(x, keep_prob=keep_prob)

        x = tflearn.batch_normalization(x, scope=scope+'_batch2')
        x = tf.nn.relu(x)
        x = tflearn.conv_2d(x, nb_filter=filters, filter_size=3, strides=1, padding='same', 
                            activation='linear', bias=False, scope=scope+'_conv2',
                            regularizer='L2', weight_decay=1e-4)
        x = tflearn.dropout(x, keep_prob=keep_prob)

        return x

def dense_block(input_x, filters, nb_layers, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)
        
        x = bottleneck_layer(input_x, filters, scope=layer_name+'_bottleN_'+str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer(x, filters, scope=layer_name + '_bottleN_'+str(i+1))
            layers_concat.append(x)

        x = tf.concat(layers_concat, axis=3)
        
        print(layer_name,x)
        return x

def transition_layer(x, scope, reduction=0.5, keep_prob=1):
    out_filters = int(int(x.get_shape()[-1])*reduction)
    with tf.name_scope(scope):
        x = tflearn.batch_normalization(x, scope=scope+'_batch1')
        x = tf.nn.relu(x)
        x = tflearn.conv_2d(x, nb_filter=out_filters, filter_size=1, strides=1, padding='same', 
                            activation='linear', bias=False, scope=scope+'_conv1',
                            regularizer='L2', weight_decay=1e-4)
        x = tflearn.dropout(x, keep_prob=keep_prob)
        x = tflearn.avg_pool_2d(x, kernel_size=2, strides=2, padding='valid')
        print(scope,x)
        return x

def main(args):
    # Data loading
    fid = "/data/srd/data/Image/cifar-100-python/train"
    data = unpickle(fid)
    # print(data.keys())
    n_class = 100

    train_feats = data["data"].astype(np.float64)
    # labs = data["labels"]
    train_labs = data["fine_labels"]
    train_feats = np.reshape(np.transpose(np.reshape(train_feats, [-1 ,3,1024]), (0, 2, 1)), [-1,32,32,3])
    train_labs = tflearn.data_utils.to_categorical(train_labs, n_class)

    fid = "/data/srd/data/Image/cifar-100-python/test"
    data = unpickle(fid)

    test_feats = data["data"].astype(np.float64)
    # labs = data["labels"]
    test_labs = data["fine_labels"]
    test_feats = np.reshape(np.transpose(np.reshape(test_feats, [-1 ,3,1024]), (0, 2, 1)), [-1,32,32,3])
    test_labs = tflearn.data_utils.to_categorical(test_labs, n_class)

    # Real-time data preprocessing
    mean = [129.30416561, 124.0699627, 112.43405006]
    std = [51.20360335, 50.57829831, 51.56057865]
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True, mean=mean)
    img_prep.add_featurewise_stdnorm(per_channel=True, std=std)
    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([32, 32], padding=4)


    # DenseNet
    depth = args.depth
    filters = args.growth_rate
    nb_blocks = 3
    #nb_layers_list = [6,12,48,32]
    nb_layers_list = [(depth - (nb_blocks + 1)) // (2*nb_blocks)  for i in range(nb_blocks)]
    print(nb_layers_list)
    net = tflearn.input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
    print("input",net)
    net = tflearn.conv_2d(net, nb_filter=2*filters, filter_size=3, strides=1, padding='same', activation='linear', bias=False, name='conv0',
                      regularizer='L2', weight_decay=1e-4)
    # net = tflearn.max_pool_2d(net, kernel_size=3, strides=2, padding='valid')
    print("init_layer",net)

    for i in range(nb_blocks-1):
        net = dense_block(net, filters, nb_layers=nb_layers_list[i], layer_name='dense_'+str(i+1))
        net= transition_layer(net, scope='trans_'+str(i+1))

    net = dense_block(net, filters, nb_layers=nb_layers_list[-1], layer_name='dense_final')

    # Global Avg + FC
    net = tflearn.batch_normalization(net, scope='linear_batch')
    net = tf.nn.relu(net)
    net = tflearn.global_avg_pool(net)
    net = tflearn.fully_connected(net, n_class, activation='softmax', regularizer='L2', weight_decay=1e-4)

    # Optimizer
    opt = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.9, use_nesterov=True)
    epsilon = 1e-4
    learning_rate = 1e-4
    # opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)

    # Regression
    net = tflearn.regression(net, optimizer=opt, loss='categorical_crossentropy', restore=False)

    # Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.add_to_collection(tf.GraphKeys.GRAPH_CONFIG, config)
    model = tflearn.DNN(net, checkpoint_path='/data/srd/models/image/model_'+args.model_name+'/model',
                        tensorboard_dir='/data/srd/logs/image/log_'+args.model_name,
                        max_checkpoints=3, tensorboard_verbose=0, clip_gradients=0.0)

    # pre-train model
    if args.pre_train:
        model.load("/data/srd/models/image/"+args.pre_train+"/model.tfl", weights_only=True)
    # model.load("/data/srd/models/image/model_densenet75_cifar100/densenet.tfl")
    # model.load("/data/srd/models/image/model_densenet225_cifar100/densenet.tfl", weights_only=True)
    # model.load("/data/srd/models/image/model_densenet226_cifar100/densenet.tfl")


    try:
        model.fit(train_feats, train_labs, n_epoch=args.epoch, validation_set=(test_feats, test_labs),
                  snapshot_epoch=False, snapshot_step=500, show_metric=True, batch_size=64, shuffle=True,
                  run_id=args.model_name)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

    model.save("/data/srd/models/image/"+args.model_name+"/model.tfl")

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, help='DenseNet\'s Depth', default=190)
    parser.add_argument('--growth_rate', type=int, help='DenseNet\'s growth_rate', default=40)
    parser.add_argument('--model_name', type=str, help='model name', default='test')
    parser.add_argument('--pre_train', type=str, help='pre train model', default=None)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('--epoch', type=int, help='max epoch', default=1000)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


