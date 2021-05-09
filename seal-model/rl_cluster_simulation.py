import numpy as np
import time, copy
import threading
from collections import deque


class EnvironmentModel:
    def __init__(self, node_cpu, node_mem, start_time_stamp):
        self.node_cpu = node_cpu
        self.judge = node_cpu
        self.node_mem = node_mem
        self.total_job_response_time = 0
        self.total_task_response_time = 0
        self.total_task_response_num = 0
        self.total_job_response_num = 0
        self.total_task_perform_num = 0
        self.names = locals()
        self.min_time_stamp = 0
        i = 0
        while i < node_cpu.shape[0]:
            self.names['node_queue_%i' % i] = deque([])
            i = i + 1
        self.TWR = np.zeros((1, 25))
        self.job_list_len_rt = []
        self.lock_2 = threading.Lock()
        self.lock_3 = threading.Lock()
        self.lock_4 = threading.Lock()
        self.lock_5 = threading.Lock()
        self.lock_6 = threading.Lock()
        self.lock_7 = threading.Lock()
        self.lock_8 = threading.Lock()
        self.lock_9 = threading.Lock()
        self.task_number_wait_time = np.array([0, 0])
        self.record_1 = 1
        self.start_time_stamp = start_time_stamp
        self.node_cpu_seer = copy.deepcopy(self.node_cpu)
        self.node_mem_seer = copy.deepcopy(self.node_mem)
        self.slas_total_task_response_time = 0
        self.slas_total_task_response_num = 0


    def node_resource_release(self, current_task_time, current_task_cpu, current_task_mem, chose_node, current_task_job,
                              current_task_name):
        # print("self.node_cpu", self.node_cpu)
        # if current_task_time < 1:
        #     self.node_cpu_seer[chose_node] += current_task_cpu
        #     self.node_mem_seer[chose_node] += current_task_mem
        #     time.sleep(current_task_time)
        # else:
        #     time.sleep(current_task_time - 1)
        #     self.node_cpu_seer[chose_node] += current_task_cpu
        #     self.node_mem_seer[chose_node] += current_task_mem
        #     time.sleep(1)

        time.sleep(current_task_time)
        # self.lock_1.acquire()
        self.node_cpu[chose_node] += current_task_cpu
        self.node_mem[chose_node] += current_task_mem
        # print(current_task_job,current_task_name)
        self.over_task[current_task_job].append(current_task_name)
        self.job_time_stamp_collect[current_task_job].append(int(round(time.time() * 1000)))
        self.total_task_response_time += int(round(time.time() * 1000)) - self.start_time_stamp
        self.total_task_response_num += 1
        # self.lock_1.release()

    def dependence_monitor_1(self, current_task):
        current_task_job = current_task[1]
        # if current_task_job == 0:
        #     return -2
        current_task_dependence = current_task[6:]
        if current_task_job in self.over_task:
            i = 0
            for j in current_task_dependence:
                if j in self.over_task[current_task_job]:
                    # print(self.over_task)
                    i = i + 1
            if i == len(current_task_dependence):
                return 1
            else:
                return 0
        else:
            self.over_task[current_task_job] = [0]
            return -1

    def dependence_monitor_2(self, current_task):
        current_task_job = current_task[1]
        if current_task_job == 0:
            return -2
        current_task_dependence = current_task[6:]
        if current_task_job in self.over_task:
            i = 0
            for j in current_task_dependence:
                if j in self.over_task[current_task_job]:
                    i = i + 1
            if i == len(current_task_dependence):
                return 1
            else:
                # print("这个任务 ", current_task_job, " 没准备好，我康康本job已完成的其他任务：", self.over_task[current_task_job])
                return 0
        else:
            return -1

    def time_statistics(self):
        ri, ri_1, sum_1 = 0, 0, 0
        while ri_1 < len(self.job_list_len_rt):
            if self.job_list_len_rt[ri_1][0] in self.job_time_stamp_collect:
                if len(self.job_time_stamp_collect[self.job_list_len_rt[ri_1][0]]) == int(
                        self.job_list_len_rt[ri_1][1]) + 1 and self.job_list_len_rt[ri_1][2] == 0:
                    # self.job_list_len_rt[ri_1][2] = max(
                    #     self.job_time_stamp_collect[self.job_list_len_rt[ri_1][0]]) - min(
                    #     self.job_time_stamp_collect[self.job_list_len_rt[ri_1][0]])
                    self.job_list_len_rt[ri_1][2] = max(
                        self.job_time_stamp_collect[self.job_list_len_rt[ri_1][0]]) - self.start_time_stamp
                    # self.job_list_len_rt[ri_1][3] = max(self.job_time_stamp_collect[self.job_list_len_rt[ri_1][0]]) - min(self.job_time_stamp_collect[self.job_list_len_rt[ri_1][0]])
                    # print("qweqwe",min(self.job_time_stamp_collect[self.job_list_len_rt[ri_1][0]]),"asdasdsa",min_time_stamp)
            ri_1 = ri_1 + 1
        arr = np.array(self.job_list_len_rt)
        exist = (arr[:, 2] != 0)
        sum_1 = sum(arr[:, 2])
        num_1 = exist.sum(axis=0)
        # print("sum_1", sum_1, "num_1", num_1)
        if num_1 != 0:
            average_job_response_time = sum_1 / num_1
        else:
            average_job_response_time = 0
        # print("self.total_job_response_time", self.total_job_response_time, "num_1", num_1,"self.total_job_response_num",self.total_job_response_num)
        # print("self.job_list_len_rt", self.job_list_len_rt)
        # 总体执行时间
        # job_list_len_rt_array = np.array(self.job_list_len_rt)
        # print(max_time_stamp, min_time_stamp)
        # total_response_time = max_time_stamp - self.min_time_stamp
        # measure_response_time = sum(job_list_len_rt_array[:, 2]) / len(job_list_len_rt_array[:, 2])
        # measure_wait_time = sum(job_list_len_rt_array[:, 3]) / len(job_list_len_rt_array[:, 3])
        return average_job_response_time, num_1

    def job_list_len(self, all_job_list, all_job_len):
        # time.sleep(2)
        self.job_list_len_rt = []
        self.job_time_stamp_collect = {}
        self.over_task = {}
        zero_supplement = np.zeros((len(all_job_list), 1))
        # zero_supplement_copy = np.zeros((len(all_job_list), 1))
        all_job_list_len = np.column_stack((np.array(all_job_list).reshape(len(all_job_list), 1),
                                            np.array(all_job_len).reshape(len(all_job_len), 1),
                                            zero_supplement))
        all_job_list_len_copy = all_job_list_len.tolist()
        self.job_list_len_rt = self.job_list_len_rt + all_job_list_len_copy
        # print(self.job_list_len_rt)

    def run(self, bound_node_task, episode):
        self.node_cpu_seer = copy.deepcopy(self.node_cpu)
        self.node_mem_seer = copy.deepcopy(self.node_mem)
        average_job_response_time, num_1, accomplish_task = 0, 0, []
        i, first_stop_num = 0, False
        ohuo = 0
        mem_sum = 0
        cpu_sum = 0
        self.lock_8.acquire()
        self.min_time_stamp = int(round(time.time() * 1000))
        # print("bound_node_task.shape[0]", bound_node_task.shape[0])
        # print(bound_node_task)
        while i < bound_node_task.shape[0]:
            current_task = bound_node_task[i, :]
            chosen_node = int(current_task[0])
            if chosen_node == -1:
                i += 1
                continue
            perform_task_job = current_task[1]
            perform_task_time = current_task[2]
            perform_task_cpu = current_task[3]
            perform_task_mem = current_task[4]
            perform_task_name = current_task[5]
            # print(perform_task_cpu, self.node_cpu[chosen_node], perform_task_mem, self.node_mem[chosen_node])
            if perform_task_cpu <= self.node_cpu[chosen_node] and perform_task_mem <= self.node_mem[
                chosen_node]:
                if perform_task_job != 0:
                    ohuo += 1
                    cpu_sum += perform_task_cpu
                    mem_sum += perform_task_mem
                    if perform_task_job not in self.job_time_stamp_collect.keys():
                        self.job_time_stamp_collect[perform_task_job] = [self.min_time_stamp]
                        xx = int(round(time.time() * 1000)) - self.start_time_stamp
                        # print(perform_task_job,xx)
                        self.total_job_response_time = self.total_job_response_time + xx
                        self.total_job_response_num += 1
                    accomplish_task.append(i)
                    # print("所处理的job为：", perform_task_job)
                    self.dependence_monitor_1(current_task)
                    self.total_task_perform_num += 1
                    # xx = int(round(time.time() * 1000)) - self.start_time_stamp
                    # self.total_task_wait_time = self.total_task_wait_time + xx
                    self.node_cpu[chosen_node] = self.node_cpu[chosen_node] - perform_task_cpu
                    self.node_mem[chosen_node] = self.node_mem[chosen_node] - perform_task_mem
                    threading.Thread(target=self.node_resource_release,
                                     args=(
                                         perform_task_time, perform_task_cpu, perform_task_mem, chosen_node,
                                         perform_task_job,
                                         perform_task_name)).start()
            else:
                break
            i = i + 1

        average_job_response_time = 0
        if episode > 4:
            average_job_response_time, num_1 = self.time_statistics()
        if self.total_task_response_num > 0:
            average_task_response_time = self.total_task_response_time / self.total_task_response_num
        else:
            average_task_response_time = 0

        self.lock_8.release()
        return self.node_cpu, self.node_mem, self.over_task, average_job_response_time, num_1, accomplish_task, self.total_job_response_num, self.total_job_response_time, average_task_response_time, self.total_task_response_num, ohuo, cpu_sum, mem_sum

    def get_cluster_state(self):
        return self.node_cpu, self.node_mem, self.over_task, np.array(self.job_list_len_rt)

    def get_node_seer(self):
        return self.node_cpu_seer, self.node_mem_seer
