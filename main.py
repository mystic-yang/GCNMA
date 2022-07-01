import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
#from scipy.io import loadmat as load
def PredictScore(train_dis_mic_matrix, dis_matrix, mic_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_dis_mic_matrix, dis_matrix, mic_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_dis_mic_matrix.sum()
    X = constructNet(train_dis_mic_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_dis_mic_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_dis_mic_matrix.shape[0], name='GCNMA')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_dis_mic_matrix.shape[0], num_v=train_dis_mic_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


def cross_validation_experiment(dis_mic_matrix, dis_matrix, mic_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(dis_mic_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating disease-microbe association...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(dis_mic_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        dis_len = dis_mic_matrix.shape[0]
        mic_len = dis_mic_matrix.shape[1]
        dis_mic_res = PredictScore(
            train_matrix, dis_matrix, mic_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = dis_mic_res.reshape(dis_len, mic_len)
        metric_tmp = cv_model_evaluate(
            dis_mic_matrix, predict_y_proba, train_matrix)

        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric


if __name__ == "__main__":
    dis_sim = np.loadtxt('../data/HMDAD/disease_features.txt')
    #dis_sim = np.loadtxt('../data/Disbiome/disease_features.txt')
    mic_sim = np.loadtxt('../data/HMDAD/microbe_features.txt')
    #mic_sim = np.loadtxt('../data/Disbiome/microbe_features.txt')
    dis_mic_matrix = np.loadtxt('../data/HMDAD/disease-microbe matrix.txt')
    #dis_mic_matrix = np.loadtxt('../data/Disbiome/disease-microbe matrix.txt')
    epoch = 4000
    emb_dim = 128
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
         result += cross_validation_experiment(
             dis_mic_matrix, dis_sim*simw, mic_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time

    ##输出预测的得分矩阵
    # dis_mic_res = PredictScore(
    #     dis_mic_matrix, dis_sim, mic_sim, 1, epoch, emb_dim, dp, lr, adjdp)
    # result1=dis_mic_res.reshape((39,292))
    # np.savetxt("./predicted_score.txt",result1)

    print(average_result)
