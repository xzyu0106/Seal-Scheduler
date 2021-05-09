import json
import requests
import numpy as np
import rl_cluster_simulation
import time, random, copy, os
import math
from train_only_dqn2_model_DuelingDDQN import OnlyDQN2_DuelingDDQN

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=400)
data_set = np.loadtxt("batch_task_new.csv", delimiter=",", skiprows=0)
data_set = np.delete(data_set, 1912, axis=0)

data_train = data_set[0:7000, :]
data_test = data_set[7000:, :]


def get_news():
    sumCpuUrl = 'http://192.168.2.104:30200/api/v1/query?query=machine_cpu_cores'
    avaCpuUrl = 'http://192.168.2.104:30200/api/v1/query?query=instance:node_cpu_utilisation:rate1m'
    sumMemUrl = 'http://192.168.2.104:30200/api/v1/query?query=machine_memory_bytes'
    avaMemUrl = 'http://192.168.2.104:30200/api/v1/query?query=node_memory_MemAvailable_bytes'
    sumCpuJson = requests.get(sumCpuUrl)
    m1SumCpu = float(json.loads(sumCpuJson.text).get('data').get('result')[0].get('value')[1])
    n2SumCpu = float(json.loads(sumCpuJson.text).get('data').get('result')[2].get('value')[1])
    n3SumCpu = float(json.loads(sumCpuJson.text).get('data').get('result')[1].get('value')[1])
    n4SumCpu = float(json.loads(sumCpuJson.text).get('data').get('result')[3].get('value')[1])

    avaCpuJson = requests.get(avaCpuUrl)
    m1AvaCpu = m1SumCpu * (1 - float(json.loads(avaCpuJson.text).get('data').get('result')[0].get('value')[1]))
    n2AvaCpu = n2SumCpu * (1 - float(json.loads(avaCpuJson.text).get('data').get('result')[1].get('value')[1]))
    n3AvaCpu = n3SumCpu * (1 - float(json.loads(avaCpuJson.text).get('data').get('result')[2].get('value')[1]))
    n4AvaCpu = n4SumCpu * (1 - float(json.loads(avaCpuJson.text).get('data').get('result')[3].get('value')[1]))

    sumMemJson = requests.get(sumMemUrl)
    m1SumMem = float(json.loads(sumMemJson.text).get('data').get('result')[0].get('value')[1]) / 1024 / 1024 / 1024
    n2SumMem = float(json.loads(sumMemJson.text).get('data').get('result')[2].get('value')[1]) / 1024 / 1024 / 1024
    n3SumMem = float(json.loads(sumMemJson.text).get('data').get('result')[1].get('value')[1]) / 1024 / 1024 / 1024
    n4SumMem = float(json.loads(sumMemJson.text).get('data').get('result')[3].get('value')[1]) / 1024 / 1024 / 1024

    avaMemJson = requests.get(avaMemUrl)
    m1AvaMem = float(json.loads(avaMemJson.text).get('data').get('result')[0].get('value')[1]) / 1024 / 1024 / 1024
    n2AvaMem = float(json.loads(avaMemJson.text).get('data').get('result')[1].get('value')[1]) / 1024 / 1024 / 1024
    n3AvaMem = float(json.loads(avaMemJson.text).get('data').get('result')[2].get('value')[1]) / 1024 / 1024 / 1024
    n4AvaMem = float(json.loads(avaMemJson.text).get('data').get('result')[3].get('value')[1]) / 1024 / 1024 / 1024

    # print("m1的总cpu为：", m1SumCpu, "           n2的总cpu为：", n2SumCpu, "           n3的总cpu为：", n3SumCpu,
    #       "           n4的总cpu为：",
    #       n4SumCpu)
    # print("m1的实时可用cpu为：", m1AvaCpu, "           n2的实时可用cpu为：", n2AvaCpu, "           n3的实时可用cpu为：", n3AvaCpu,
    #       "           n4的实时可用cpu为：", n4AvaCpu)
    # print("m1的总mem为：", m1SumMem, "           n2的总mem为：", n2SumMem, "           n3的总mem为：", n3SumMem,
    #       "           n4的总mem为：",
    #       n4SumMem)
    # print("m1的实时可用mem为：", m1AvaMem, "           n2的实时可用mem为：", n2AvaMem, "           n3的实时可用mem为：", n3AvaMem,
    #       "           n4的实时可用mem为：", n4AvaMem)

    node_cpu_sum = np.array(
        [m1SumCpu, n2SumCpu, n3SumCpu, n4SumCpu], dtype=np.float)
    node_mem_sum = np.array(
        [m1SumMem, n2SumMem, n3SumMem, n4SumMem], dtype=np.float)
    node_cpu_ava = np.array(
        [m1AvaCpu, n2AvaCpu, n3AvaCpu, n4AvaCpu], dtype=np.float)
    node_mem_ava = np.array(
        [m1AvaMem, n2AvaMem, n3AvaMem, n4AvaMem], dtype=np.float)
    return node_cpu_sum, node_mem_sum, node_cpu_ava, node_mem_ava


def seal_trainer_dqn2():
    global data_train
    # learn_gap = 50
    batch_task_num, var = 500, 3
    restore_record = 0
    # s_dim = (len(node_cpu) + 3 + 9) * (batch_task_num + len(node_cpu))
    s_dim = 2 * (len(node_cpu_sum))
    # s_dim = len(node_cpu)
    a_dim = len(node_cpu_sum)
    # a_bound = np.array([1])
    client_record = np.zeros([1, 5])
    dqn2 = OnlyDQN2_DuelingDDQN(a_dim, s_dim)
    learn_num_2 = 0

    # epsilon_max, epsilon = 0.9, 0  # control exploration
    # epsilon_increment = 0.0001

    current_process, total_process, overall_episode = 0, 1500, 0
    while learn_num_2 < 400000:
        print("##############################seal_scheduler_dqn2，进入流程ID：", current_process,
              "中，正在处理，请稍后##############################")
        random_data_set_list = random.sample(overall_job_list, 1000)
        current_data_set = np.zeros([1, data_train.shape[1]])
        for b in random_data_set_list:
            for c in data_train:
                if c[0] == b:
                    current_data_set = np.row_stack((current_data_set, c))
        current_data_set = np.delete(current_data_set, 0, axis=0)
        i = 0
        while i < current_data_set.shape[0]:
            current_data_set[i, 1] = random.randint(1, 20)  # 执行时间
            current_data_set[i, 2] = random.choice([0.5, 1])  # cpu
            current_data_set[i, 3] = random.randint(20, 70) / 100  # mem
            i += 1
        all_job_list = []
        all_job_len = []
        one_job_len = 0
        job_judge = current_data_set[0, 0]
        # 计算data_train中的job列表，和每个job的task个数，开始
        for round_1 in current_data_set:
            if round_1[0] != job_judge:
                all_job_len.append(one_job_len)
                one_job_len = 0
                all_job_list.append(job_judge)
                job_judge = round_1[0]
            one_job_len = one_job_len + 1
        print(current_data_set)

        # 开始随机算法
        print("*************random算法ing")
        # 给data_train中的每个task打上tag
        current_data_train = np.insert(current_data_set, 0, values=0, axis=1)
        episode = 0
        current_node_cpu = copy.deepcopy(node_cpu_sum)
        current_node_mem = copy.deepcopy(node_mem_sum)
        start_time_stamp = int(round(time.time() * 1000))
        env = rl_cluster_simulation.EnvironmentModel(current_node_cpu, current_node_mem, start_time_stamp)
        env.job_list_len(all_job_list, all_job_len)
        t_start = time.time()
        # 开始循环
        while True:
            t1 = time.time()
            wait_solve = np.zeros((batch_task_num, 24))
            cyclic, cyclic_1 = 0, 0
            exit_flag = False
            # 获取要处理的100个task，开始
            while cyclic < current_data_train.shape[0]:
                if current_data_train[cyclic, 0] == 0:
                    round_data = current_data_train[cyclic, 1:]
                    if env.dependence_monitor_1(current_data_train[cyclic, :]) == 1 or round_data[5] == 0:
                        wait_solve[cyclic_1, :] = round_data
                        cyclic_1 = cyclic_1 + 1
                    if cyclic_1 == batch_task_num:
                        exit_flag = True
                        break
                    cyclic = cyclic + 1
                else:
                    cyclic = cyclic + 1
                if exit_flag:
                    break
            # 获取要处理的100个task，结束

            random_allocate = np.random.randint(0, node_cpu_sum.shape[0], (wait_solve.shape[0], 1))
            bound_node_task_1 = np.column_stack((random_allocate, wait_solve))
            bound_node_task = np.array(bound_node_task_1)
            # 执行！

            current_node_cpu, current_node_mem, over_task, average_job_response_time, num_1, accomplish_task, total_job_response_num, total_job_response_time, average_task_response_time, total_task_response_num, ohuo, cpu_sum, mem_sum = env.run(
                bound_node_task, episode)
            current_node_cpu, current_node_mem, over_task, job_list_len_rt = env.get_cluster_state()
            # 给已处理的task打上记号，开始
            if accomplish_task:
                for yy in accomplish_task:
                    bound_node_task[yy, 0] = 0
                    current_data_train[np.where(
                        (current_data_train == bound_node_task[yy, :].reshape(1, 25)[:, None]).all(-1))[1], 0] = 1
            # print('#########################################当前流程ID为：', current_process, '，此刻Episode为:', episode,
            #       '#########################################')
            # # print("本次所处理的任务序号为：", wait_solve[:, 0].tolist())
            # print("当前所有节点的cpu 为:", current_node_cpu, "，所有节点的cpu平均利用率为：", 1 - np.mean(current_node_cpu / node_cpu),
            #       "\n当前所有节点的mem 为:", current_node_mem, "，所有节点的mem平均利用率为：", 1 - np.mean(current_node_mem / node_mem))
            # # # print("当前的job(包含已完成和待完成的)编号、长度、用时情况：\n", job_list_len_rt)
            # print("当前已完成的job任务情况：", over_task)
            # print("当前所有job的平均响应时间为：", average_job_response_time, "，当前已完成job数目为：",
            #       num_1)
            # print("当前所有task的平均响应时间时间为：", average_task_response_time, "，当前已完成task数目为：", total_task_response_num)
            # if r_dqn == -0 and episode > 5:
            # if cyclic_1 == 0 and int(sum(((job_list_len_rt > 0) * 1.0)[:, 2])) == job_list_len_rt.shape[0]:
            # if num_1 > 100:
            if episode == 5:
                t_end = time.time()
                t2 = time.time()
                print('*************本次random调度已完成，总共所用Episode为:', episode)
                print("本流程所选择job为：", random_data_set_list)
                print("当前所有节点的cpu 为:", current_node_cpu, "，所有节点的cpu平均利用率为：",
                      1 - np.mean(current_node_cpu / node_cpu_sum),
                      "\n当前所有节点的mem 为:", current_node_mem, "，所有节点的mem平均利用率为：",
                      1 - np.mean(current_node_mem / node_mem_sum))
                # print("当前的job(包含已完成和待完成的)编号、长度、用时情况：\n", job_list_len_rt)
                # print("当前已完成的job任务情况：", over_task)
                print("本次所有job的平均响应时间为：", average_job_response_time, "，当前已完成job数目为：", num_1)
                print("当前所有task的平均响应时间时间为：", average_task_response_time, "，当前已完成task数目为：", total_task_response_num)
                print("总用时为：", t_end - t_start, "最后一次的t2-t1为：", t2 - t1)
                # random_average_job_response_time = average_job_response_time
                random_average_task_response_time = average_task_response_time
                randon_episode = episode
                random_time = t_end - t_start
                break
            episode = episode + 1
            t2 = time.time()
            if t2 - t1 < 1:
                time.sleep(1 - t2 + t1)

        print("*************seal算法ing")
        # 给data_train中的每个task打上tag
        current_data_train = np.insert(current_data_set, 0, values=0, axis=1)
        episode = 0
        average_job_response_time_, num_1_, s_dqn, s_dqn2, average_task_response_time_, s_dqn_, s_dqn2_, sss_dqn2, sss_dqn2_, a_dqn, a_dqn2, r_dqn, r_dqn2, resource_utilization_, eps2, y_dqn2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        start_time_stamp = int(round(time.time() * 1000))
        current_node_cpu = copy.deepcopy(node_cpu_sum)
        current_node_mem = copy.deepcopy(node_mem_sum)
        node_cpu_seer = copy.deepcopy(node_cpu_sum)
        node_mem_seer = copy.deepcopy(node_mem_sum)
        env = rl_cluster_simulation.EnvironmentModel(current_node_cpu, current_node_mem, start_time_stamp)
        env.job_list_len(all_job_list, all_job_len)
        t_start = time.time()
        # 开始循环
        while True:
            t1 = time.time()
            ohuo = 0
            wait_solve = np.zeros((batch_task_num, 24))
            dqn_store, dqn2_store = [], []
            cyclic, cyclic_1 = 0, 0
            exit_flag = False
            # 获取要处理的100个task，开始
            while cyclic < current_data_train.shape[0]:
                if current_data_train[cyclic, 0] == 0:
                    round_data = current_data_train[cyclic, 1:]
                    if env.dependence_monitor_1(current_data_train[cyclic, :]) == 1 or round_data[5] == 0:
                        wait_solve[cyclic_1, :] = round_data
                        cyclic_1 = cyclic_1 + 1
                    if cyclic_1 == batch_task_num:
                        exit_flag = True
                        break
                    cyclic = cyclic + 1
                else:
                    cyclic = cyclic + 1
                if exit_flag:
                    break
            # 获取要处理的100个task，结束
            current_node_cpu, current_node_mem, over_task, job_list_len_rt = env.get_cluster_state()
            wait_solve_dqn2 = wait_solve
            # observation = np.zeros([2, 1 + 10 + len(node_cpu) * 2])
            # internal_index = 0
            # dqn2_node_cpu = copy.deepcopy(current_node_cpu)
            # dqn2_node_mem = copy.deepcopy(current_node_mem)
            # action_chose_node_copy = np.zeros([batch_task_num, 1])
            # # s_dqn = observation.reshape(-1)
            # node_cpu_seer, node_mem_seer = env.get_node_seer()
            # # print("1:", current_node_cpu, current_node_mem)
            # # print("2:", node_cpu_seer, node_mem_seer)
            # while internal_index < batch_task_num:
            #     # cpu_var_1 = np.var(dqn2_node_cpu)
            #     # mem_var_1 = np.var(dqn2_node_mem)
            #     # print(s_dqn2[:10])
            #     observation[0, 0] = wait_solve_dqn2[internal_index, 2]
            #     observation[1, 0] = wait_solve_dqn2[internal_index, 3]
            #     rl_job = wait_solve_dqn2[internal_index, 0]
            #     rl_task = wait_solve_dqn2[internal_index, 4]
            #     rl_idx = current_data_train[:, 1]
            #     next_task, rr = [], 0
            #     # print(rl_job)
            #     if rl_job != 0:
            #         idx_1 = np.argwhere(rl_idx == rl_job)[0]
            #         idx_2 = np.argwhere(rl_idx == rl_job)[-1]
            #         job_sum_execution_time, job_undone_execution_time, job_undone_task_num = 0, 0, 0
            #         while idx_1 < idx_2 + 1:
            #             if current_data_train[idx_1, 5] not in over_task[rl_job]:
            #                 job_undone_execution_time = job_undone_execution_time + current_data_train[idx_1, 2]
            #                 job_undone_task_num = job_undone_task_num + 1
            #             if rl_task in current_data_train[idx_1, 6:]:
            #                 observation[0, 1 + rr] = current_data_train[idx_1, 3]
            #                 observation[1, 1 + rr] = current_data_train[idx_1, 4]
            #                 rr = rr + 1
            #             if rr == 5:
            #                 break
            #             idx_1 = idx_1 + 1
            #     i = 0
            #     while i < len(node_cpu):
            #         observation[0, 1 + 10 + i] = dqn2_node_cpu[i]
            #         observation[1, 1 + 10 + i] = dqn2_node_mem[i]
            #         observation[0, 1 + 10 + i + len(node_cpu)] = node_cpu_seer[i]
            #         observation[1, 1 + 10 + i + len(node_cpu)] = node_mem_seer[i]
            #         i += 1
            #     # observation_2 = np.delete(observation, 0, axis=1)
            #     s_dqn2_ = observation.reshape(-1)
            #     if episode > 4 and internal_index > 0 and r_dqn2 != 0:
            #         # print(s_dqn2, a_dqn2, r_dqn2, s_dqn2_)
            #         dqn2.store_transition(s_dqn2, a_dqn2, r_dqn2, s_dqn2_)
            #         restore_record += 1
            #     s_dqn2 = copy.deepcopy(s_dqn2_)
            #     a_dqn2, y_dqn2, actions_value2, eps2 = dqn2.choose_action(s_dqn2_, current_process)
            #     if observation[0, 0] != 0:
            #         r_dqn2 = (0.7 * (dqn2_node_cpu[a_dqn2] - np.mean(dqn2_node_cpu)) + 0.3 * (
            #                 dqn2_node_mem[a_dqn2] - np.mean(dqn2_node_mem))) - (
            #                          dqn2_node_cpu[a_dqn2] % observation[0, 0] + dqn2_node_mem[a_dqn2] % observation[
            #                      1, 0])
            #     else:
            #         r_dqn2 = 0
            #     # print(y_dqn2,a_dqn2, r_dqn2)
            #     action_chose_node_copy[internal_index, 0] = a_dqn2
            #     if dqn2_node_cpu[a_dqn2] > observation[0, 0] and dqn2_node_mem[a_dqn2] > \
            #             observation[1, 0]:
            #         # dqn2_node_cpu[a_dqn2] -= observation[len(current_node_cpu), len(node_cpu)]
            #         # dqn2_node_mem[a_dqn2] -= observation[len(current_node_cpu), len(node_cpu) + 1]
            #         dqn2_node_cpu[a_dqn2] -= observation[0, 0]
            #         dqn2_node_mem[a_dqn2] -= observation[1, 0]
            #     else:
            #         r_dqn2 = 0
            #     # print(y_dqn2, a_dqn2, r_dqn2, actions_value2)
            #     # cpu_var_2 = np.var(dqn2_node_cpu)
            #     # mem_var_2 = np.var(dqn2_node_mem)
            #     # print("2", observation)
            #     # r_dqn2 = (cpu_var_1 - cpu_var_2) + (mem_var_1 - mem_var_2)
            #     internal_index += 1
            # 制作rl所需的state，结束
            # 处理获取的action成为bound_node_task
            # print(action_chose_node_copy)

            observation = np.zeros([2, len(node_cpu_sum)])
            internal_index = 0
            dqn2_node_cpu = copy.deepcopy(current_node_cpu)
            dqn2_node_mem = copy.deepcopy(current_node_mem)
            action_chose_node_copy = np.zeros([batch_task_num, 1])
            # s_dqn = observation.reshape(-1)
            actions_value2 = 0
            # r_dqn2 = -2
            cpu_var_2 = 0
            mem_var_2 = 0
            while internal_index < batch_task_num:
                cpu_var_1 = np.std(dqn2_node_cpu)
                mem_var_1 = np.std(dqn2_node_mem)
                # print(s_dqn2[:10])
                # print(dqn2_node_cpu,dqn2_node_mem)
                i = 0
                while i < len(node_cpu_sum):
                    observation[0, i] = dqn2_node_cpu[i]
                    observation[1, i] = dqn2_node_mem[i]
                    i += 1
                observation[0, :] = (observation[0, :] - 0.99 * np.min(observation[0, :])) / (
                        1.01 * np.max(observation[0, :]) - 0.99 * np.min(observation[0, :]))
                observation[1, :] = (observation[1, :] - 0.99 * np.min(observation[1, :])) / (
                        1.01 * np.max(observation[1, :]) - 0.99 * np.min(observation[1, :]))
                s_dqn2_ = observation.reshape(-1)
                # print(observation)

                if episode > 2 and internal_index > 0 and r_dqn2 != 0:
                    # print("\ns:", s_dqn2.reshape(2, len(node_cpu) * 2 + 1 + 1), actions_value2, "动作：", a_dqn2, "奖励：",
                    #       r_dqn2, "是否随机：", y_dqn2, "\ns_:", s_dqn2_.reshape(2, len(node_cpu) * 2 + 1 + 1))
                    dqn2.store_transition(s_dqn2, a_dqn2, r_dqn2, s_dqn2_)
                    restore_record += 1
                    if overall_episode > 20:
                        ri = 0
                        while ri < 5:
                            lr2, loss2 = dqn2.learn()
                            ri += 1
                            if learn_num_2 % 1000 == 0:
                                dqn2.save_model(learn_num_2)
                            learn_num_2 += 1
                        # learn_num_2 += 1
                        # lr2, loss2 = dqn2.learn()
                        # print("*******************************************************loss为：", loss2)
                s_dqn2 = copy.deepcopy(s_dqn2_)
                a_dqn2, y_dqn2, actions_value2, eps2 = dqn2.choose_action(s_dqn2_, current_process, dqn2_node_cpu)
                r_dqn2_cpu, r_dqn2_mem = 0, 0
                if wait_solve_dqn2[internal_index, 2] != 0:
                    # di2 = 0
                    # while di2 < len(node_cpu):
                    #     if dqn2_node_cpu[a_dqn2] >= dqn2_node_cpu[di2]:
                    #         r_dqn2_cpu += 1
                    #     if dqn2_node_mem[a_dqn2] >= dqn2_node_mem[di2]:
                    #         r_dqn2_mem += 1
                    #     di2 += 1

                    # if cpu_var_1 > cpu_var_2:
                    #     r_dqn2_cpu += 1
                    # else:
                    #     r_dqn2_cpu -= 1
                    # if mem_var_1 > mem_var_2:
                    #     r_dqn2_mem += 1
                    # else:
                    #     r_dqn2_mem -= 1
                    # print(dqn2_node_cpu,dqn2_node_mem)
                    # print(r_dqn2_cpu, r_dqn2_mem)
                    # r_dqn2 = 0.1 * (0.8 * r_dqn2_cpu + 0.2 * r_dqn2_mem) - 0.05 * (len(node_cpu))
                    # r_dqn2 = 0.08 * observation[a_dqn2, 2] + 0.02 * observation[a_dqn2, 3]
                    # if dqn2_node_cpu[a_dqn2]
                    # r_dqn2 = (0.8 * (dqn2_node_cpu[a_dqn2] - np.mean(dqn2_node_cpu)) + 0.2 * (
                    #         dqn2_node_mem[a_dqn2] - np.mean(dqn2_node_mem))) - 0.01 * (
                    #                  dqn2_node_cpu[a_dqn2] % observation[0, 0] + dqn2_node_mem[a_dqn2] % observation[
                    #              0, 1])
                    r_dqn2_base = 0.8 * (observation[0, a_dqn2] - np.mean(observation[0, :])) + 0.2 * (
                                observation[1, a_dqn2] - np.mean(observation[1, :])) - 0.008 * observation[
                                      0, a_dqn2] - 0.002 * observation[1, a_dqn2]
                    r_dqn2_state_value = np.var(dqn2_node_cpu) + np.var(dqn2_node_mem)
                    r_dqn2_state_value_1 = np.exp(-(r_dqn2_state_value - 0) ** 2 / (2 * 1 ** 2)) / (
                            math.sqrt(2 * math.pi) * 1)
                    r_dqn2 = (1 + 4 * r_dqn2_state_value_1) * r_dqn2_base
                else:
                    r_dqn2 = 0
                # print(y_dqn2,a_dqn2, r_dqn2)
                action_chose_node_copy[internal_index, 0] = a_dqn2
                if dqn2_node_cpu[a_dqn2] > wait_solve_dqn2[internal_index, 2] and dqn2_node_mem[a_dqn2] > \
                        wait_solve_dqn2[internal_index, 3]:
                    # dqn2_node_cpu[a_dqn2] -= observation[len(current_node_cpu), len(node_cpu)]
                    # dqn2_node_mem[a_dqn2] -= observation[len(current_node_cpu), len(node_cpu) + 1]
                    dqn2_node_cpu[a_dqn2] -= wait_solve_dqn2[internal_index, 2]
                    dqn2_node_mem[a_dqn2] -= wait_solve_dqn2[internal_index, 3]
                else:
                    r_dqn2 = 0
                # print(y_dqn2, a_dqn2, r_dqn2, actions_value2)
                cpu_var_2 = np.std(dqn2_node_cpu)
                mem_var_2 = np.std(dqn2_node_mem)
                # print("2", observation)
                # r_dqn2 = (cpu_var_1 - cpu_var_2) + (mem_var_1 - mem_var_2)
                internal_index += 1
            bound_node_task = np.column_stack((action_chose_node_copy, wait_solve))
            # print(bound_node_task[:,:3])
            # 执行！
            current_node_cpu, current_node_mem, over_task, average_job_response_time, num_1, accomplish_task, total_job_response_num, total_job_response_time, average_task_response_time, total_task_response_num, ohuo, cpu_sum, mem_sum = env.run(
                bound_node_task, episode)
            current_node_cpu, current_node_mem, over_task, job_list_len_rt = env.get_cluster_state()
            # 给已处理的task打上记号，开始
            if accomplish_task:
                for yy in accomplish_task:
                    bound_node_task[yy, 0] = 0
                    current_data_train[np.where(
                        (current_data_train == bound_node_task[yy, :].reshape(1, 25)[:, None]).all(-1))[1], 0] = 1

            if cyclic_1 == 0 and int(sum(((job_list_len_rt > 0) * 1.0)[:, 2])) == job_list_len_rt.shape[0]:
                # if num_1 > 1100:
                # if episode ==100:
                t_end = time.time()
                t2 = time.time()
                print('*************本次seal调度已完成，总共所用Episode为:', episode, "当前overall_episode为：", overall_episode)
                print("本流程所选择job为：", random_data_set_list)
                # print("本次所处理的任务序号为：", wait_solve[:, 0].tolist())
                # print('DQN的Reward: ', r_dqn, '，DQN的Explore为:', eps, '\ndqn2的Reward: ', r_dqn2, '，dqn2的Explore为:',
                #       epsilon)
                print("当前所有节点的cpu 为:", current_node_cpu, "，所有节点的cpu平均空闲率为：", np.mean(current_node_cpu / node_cpu_sum),
                      "\n当前所有节点的mem 为:", current_node_mem, "，所有节点的mem平均空闲率为：",
                      np.mean(current_node_mem / node_mem_sum))
                # print("当前的job(包含已完成和待完成的)编号、长度、用时情况：\n", job_list_len_rt)
                # print("当前已完成的job任务情况：", over_task)
                print("当前所有job的平均响应时间为：", average_job_response_time,
                      "，当前已完成job数目为：", num_1)
                print("总用时为：", t_end - t_start, "最后一次的t2-t1为：", t2 - t1)
                print("截至目前的dqn2的探索率为：", 1 - eps2)
                print("当前所有task的平均响应时间时间为：", average_task_response_time, "，当前已完成task数目为：", total_task_response_num)
                print("restore_record", restore_record, "，学习的次数为：", learn_num_2)
                # seal_average_job_response_time = average_job_response_time
                seal_average_task_response_time = average_task_response_time
                seal_episode = episode
                seal_time = t_end - t_start

                reward = 0.003 * (random_average_task_response_time - seal_average_task_response_time) + 0.7 * (
                        random_time - seal_time)
                # reward = -reward
                print("reward为：", reward)
                break
            episode = episode + 1
            overall_episode = overall_episode + 1
            t2 = time.time()
            if t2 - t1 < 1:
                time.sleep(1 - t2 + t1)
            # print("（t2 - t1）为：", t2 - t1, "，（t4 - t3）为：", t4 - t3)
        current_process += 1
        time.sleep(2)


if __name__ == "__main__":
    node_cpu_sum, node_mem_sum, node_cpu_ava, node_mem_ava = get_news()
    # print(node_cpu_sum, node_mem_sum, node_cpu_ava, node_mem_ava)
    overall_job_list = []
    overall_job_len = []
    overall_one_job_len = 0
    overall_job_judge = data_train[0, 0]
    for overall_round_1 in data_train:
        if overall_round_1[0] != overall_job_judge:
            overall_job_len.append(overall_one_job_len)
            overall_one_job_len = 0
            overall_job_list.append(overall_job_judge)
            overall_job_judge = overall_round_1[0]
        overall_one_job_len = overall_one_job_len + 1

    seal_trainer_dqn2()
