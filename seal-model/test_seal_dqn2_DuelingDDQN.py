# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

np.random.seed(1)  # 原来每次运行代码时设置相同的seed，则每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样。
tf.set_random_seed(1)


class SealDQN2_DuelingDDQN:
    def __init__(
            self,
            n_actions,
            n_features,  # 使用features来预测actions的值
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        # consist of [target_net, evaluate_net]
        # saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.sess = tf.Session()
        self._build_net()
        # self.load_model()

        self.t_params = tf.get_collection('dqn2_target_net_params')
        self.e_params = tf.get_collection('dqn2_eval_net_params')
        self.sess.run(tf.global_variables_initializer())
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("dqn2_test_logs/", self.sess.graph)

        # if sess is None:
        #     self.sess = tf.Session()
        #     self.sess.run(tf.global_variables_initializer())
        # else:
        #     self.sess = sess

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='dqn2_s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='dqn2_Q_target')  # for calculating loss
        with tf.variable_scope('dqn2_eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_nodes, w_initializer, b_initializer = \
                ['dqn2_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 90, \
                tf.random_normal_initializer(), tf.constant_initializer()  # config of layers

            # first layer. collections is used later when assign to target net
            # with tf.variable_scope('dqn2_l1'):
            #     w1 = tf.get_variable('dqn2_w1', [self.n_features, n_nodes],
            #                          initializer=w_initializer,
            #                          collections=c_names)  # 当我们需要共享变量的时候，需要使用tf.get_variable()。在其他
            #     # 情况下，tf.Variable() 和tf.get_variable()的用法是一样的.tf.get_variable(name,  shape, initializer)
            #     # 创建一个变量对于get_variable()，来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的
            #     # 话，就创建一个新的。
            #     b1 = tf.get_variable('dqn2_b1', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)  # tf.multiply（）两个矩阵中对应元素各自相乘;
            #
            #     # tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
            #     # 第二三四五层
            # with tf.variable_scope('dqn2_l2'):
            #     w2 = tf.get_variable('dqn2_w2', [n_nodes, n_nodes], initializer=w_initializer, collections=c_names)
            #     b2 = tf.get_variable('dqn2_b2', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            #
            # with tf.variable_scope('dqn2_l3'):
            #     w3 = tf.get_variable('dqn2_w3', [n_nodes, n_nodes], initializer=w_initializer, collections=c_names)
            #     b3 = tf.get_variable('dqn2_b3', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.swish(tf.matmul(l2, w3) + b3)
            # with tf.variable_scope('dqn2_l4'):
            #     w4 = tf.get_variable('dqn2_w4', [n_nodes, n_nodes], initializer=w_initializer,
            #                          collections=c_names)
            #     b4 = tf.get_variable('dqn2_b4', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l4 = tf.nn.swish(tf.matmul(l1, w4) + b4)
            with tf.variable_scope('dqn2_Value'):  # 专门分析 state 的 Value
                w5 = tf.get_variable('dqn2_w5', [self.n_features, 1], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('dqn2_b5', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.nn.relu(tf.matmul(self.s, w5) + b5)

            with tf.variable_scope('dqn2_Advantage'):  # 专门分析每种动作的 Advantage
                w6 = tf.get_variable('dqn2_w6', [self.n_features, self.n_actions], initializer=w_initializer,
                                     collections=c_names)
                b6 = tf.get_variable('dqn2_b6', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.A = tf.nn.relu(tf.matmul(self.s, w6) + b6)

            with tf.variable_scope('dqn2_Q'):  # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值
                self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)


    def load_model(self, r):
        # saver = tf.train.Saver()
        # # self.sess.run(tf.global_variables_initializer())
        # saver.restore(self.sess, "./seal_model_dqn/saved_model_" + str(r))
        # saver = tf.train.import_meta_graph('./model_save_path/saved_model.meta')
        # saver.restore(self.sess, tf.train.latest_checkpoint("./model_save_path"))
        # saver1 = tf.train.Saver()
        # saver2 = tf.train.Saver()
        # saver1 = tf.train.Saver(self.t_params)
        saver2 = tf.train.Saver(self.e_params)
        # saver1.restore(self.sess, "./seal_model_dqn/saved_model_Target_" + str(r))
        saver2.restore(self.sess, "./2_model_dqn2_DuelingDDQN/2_saved_model_Eval_" + str(r))



    def choose_action(self, observation):
        observation_1 = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation_1})
        action = int(np.argmax(actions_value).reshape(-2))
        # if not hasattr(self, 'q'):  # 记录选的 Qmax 值
        #     self.q = []
        #     self.running_q = 0
        # self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        # self.q.append(self.running_q)
        return actions_value,action


