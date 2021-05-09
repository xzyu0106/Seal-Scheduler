# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding('UTF-8')
import numpy as np
import tensorflow as tf


# import matplotlib.pyplot as plt

# np.random.seed(1)  # 原来每次运行代码时设置相同的seed，则每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样。
# tf.set_random_seed(1)


# Deep Q Network off-policy
class OnlyDQN2_DuelingDDQN:
    def __init__(
            self,
            n_actions,
            n_features,  # 使用features来预测actions的值
            # TASK_NUM,
            # initial_learning_rate=0.1,
            # learning_rate_decay=0.5,
            # learning_rate_minimum=0.001,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=10000,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=0.005,
            output_graph=True,
            double_q=True,
            sess=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        # self.TASK_NUM = TASK_NUM
        # self.lr = initial_learning_rate
        # self.learning_rate_decay = learning_rate_decay
        # self.learning_rate_minimum = learning_rate_minimum
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # self.sequential_selection_action = np.zeros([1, self.TASK_NUM])

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.t_params = tf.get_collection('dqn2_target_net_params')
        self.e_params = tf.get_collection('dqn2_eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            # $ tensorboard --logdir=train_logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("dqn2_train_logs/", self.sess.graph)

        # self.double_q = double_q
        # if sess is None:
        #     self.sess = tf.Session()
        # else:
        #     self.sess = sess
        # self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='dqn2_s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='dqn2_Q_target')  # for calculating loss
        with tf.variable_scope('dqn2_eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_nodes, w_initializer, b_initializer = \
                ['dqn2_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 90, \
                tf.ones_initializer(), tf.ones_initializer()  # config of layers

            # first layer. collections is used later when assign to target net
            # with tf.variable_scope('dqn2_l1'):
            #     w1 = tf.get_variable('dqn2_w1', [self.n_features, self.n_actions],
            #                          initializer=w_initializer,
            #                          collections=c_names)  # 当我们需要共享变量的时候，需要使用tf.get_variable()。在其他
            #     # 情况下，tf.Variable() 和tf.get_variable()的用法是一样的.tf.get_variable(name,  shape, initializer)
            #     # 创建一个变量对于get_variable()，来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的
            #     # 话，就创建一个新的。
            #     b1 = tf.get_variable('dqn2_b1', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            #     self.q_eval = tf.nn.relu(tf.matmul(self.s, w1) + b1)  # tf.multiply（）两个矩阵中对应元素各自相乘;

            # tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
            # 第二三四五层
            # with tf.variable_scope('dqn2_l2'):
            #     w2 = tf.get_variable('dqn2_w2', [n_nodes, n_nodes], initializer=w_initializer, collections=c_names)
            #     b2 = tf.get_variable('dqn2_b2', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l2 = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)
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

        with tf.variable_scope('dqn2_loss'):
            # self.loss = tf.reduce_mean(tf.abs(tf.subtract(self.q_target, self.q_eval)))  # 基于Q估计与Q现实，构造
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            # self.loss = tf.nn.l2_loss(tf.subtract(self.q_target, self.q_eval))
            # self.loss = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(self.q_target, self.q_eval))
            # loss-function
        with tf.variable_scope('dqn2_train'):
            # self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # 进行训练
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='2_s_')
        with tf.variable_scope('dqn2_target_net'):
            c_names = ['dqn2_target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            #
            # with tf.variable_scope('dqn2_l1'):
            #     w1 = tf.get_variable('dqn2_w1', [self.n_features, self.n_actions], initializer=w_initializer,
            #                          collections=c_names)
            #     b1 = tf.get_variable('dqn2_b1', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            #     self.q_next = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # with tf.variable_scope('dqn2_l2'):
            #     w2 = tf.get_variable('dqn2_w2', [n_nodes, n_nodes], initializer=w_initializer,
            #                          collections=c_names)
            #     b2 = tf.get_variable('dqn2_b2', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l2 = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)
            # with tf.variable_scope('dqn2_l3'):
            #     w3 = tf.get_variable('dqn2_w3', [n_nodes, n_nodes], initializer=w_initializer,
            #                          collections=c_names)
            #     b3 = tf.get_variable('dqn2_b3', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.swish(tf.matmul(l2, w3) + b3)
            # with tf.variable_scope('dqn2_l4'):
            #     w4 = tf.get_variable('dqn2_w4', [n_nodes, n_nodes], initializer=w_initializer,
            #                          collections=c_names)
            #     b4 = tf.get_variable('dqn2_b4', [1, n_nodes], initializer=b_initializer, collections=c_names)
            #     l4 = tf.nn.swish(tf.matmul(c, w4) + b4)
            with tf.variable_scope('dqn2_Value'):  # 专门分析 state 的 Value
                w5 = tf.get_variable('dqn2_w5', [self.n_features, 1], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('dqn2_b5', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.nn.relu(tf.matmul(self.s_, w5) + b5)

            with tf.variable_scope('dqn2_Advantage'):  # 专门分析每种动作的 Advantage
                w6 = tf.get_variable('dqn2_w6', [self.n_features, self.n_actions], initializer=w_initializer,
                                     collections=c_names)
                b6 = tf.get_variable('dqn2_b6', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.A = tf.nn.relu(tf.matmul(self.s_, w6) + b6)

            with tf.variable_scope('dqn2_Q'):  # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值
                self.q_next = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'dqn2_memory_counter'):
            self.memory_counter = 0

        transition = np.hstack([s, a, r, s_])

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation, current_process, dqn2_node_cpu):
        observation_1 = observation[np.newaxis, :]
        # if np.random.uniform() > self.epsilon and current_process % 5 != 4:
        if np.random.uniform() > 5:
            actions_value = np.random.randint(-1000, 1000, (1, self.n_actions)) / 100
            action = int(np.argmax(actions_value).reshape(-2))
            action = np.argmax(dqn2_node_cpu)
            y = 1
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation_1})
            action = int(np.argmax(actions_value).reshape(-2))
            # print(action)
            y = 0
        return action, y, actions_value, self.epsilon

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # # self.lr = self.lr * self.learning_rate_decay if self.lr > self.learning_rate_minimum else self.learning_rate_minimum
            print('\ndqn2_target_params_replaced\n')
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else \
            self.epsilon_max
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # print("batch_memory = \n" + str(batch_memory))

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, -self.n_features:],  # newest params
            })
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)  # q_eval 得出的最高奖励动作
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN 选择 q_next 依据 q_eval 选出的动作

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        # print("q_target = \n" + str(q_target))

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # self.cost_his.append(self.cost)

        # increasing epsilon
        self.learn_step_counter += 1

        return self.lr, self.cost

    # def plot_cost(self):
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')
    #     plt.savefig("cost.jpg")
    #     plt.show()

    def save_model(self, r):
        self.r = r
        # saver = tf.train.Saver()
        # saver.save(self.sess, "./seal_model_dqn/saved_model_" + str(self.r))

        # saver1 = tf.train.Saver(self.t_params)
        saver2 = tf.train.Saver(self.e_params)
        # saver1.save(self.sess, "./seal_model_dqn/saved_model_Target_" + str(self.r))
        saver2.save(self.sess, "./2_model_dqn2_DuelingDDQN/2_saved_model_Eval_" + str(self.r))
