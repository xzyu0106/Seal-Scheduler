import json
import requests
import numpy as np
from redis import StrictRedis
import rl_cluster_simulation
import time, random, copy, os
import math
from test_seal_dqn2_DuelingDDQN import SealDQN2_DuelingDDQN

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=400)


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

    node_cpu_sum = np.array(
        [m1SumCpu, n2SumCpu, n3SumCpu, n4SumCpu], dtype=np.float)
    node_mem_sum = np.array(
        [m1SumMem, n2SumMem, n3SumMem, n4SumMem], dtype=np.float)
    node_cpu_ava = np.array(
        [m1AvaCpu, n2AvaCpu, n3AvaCpu, n4AvaCpu], dtype=np.float)
    node_mem_ava = np.array(
        [m1AvaMem, n2AvaMem, n3AvaMem, n4AvaMem], dtype=np.float)

    print("1.本次获取集群资源信息：\n        集群cpu和内存总量：", node_cpu_sum, node_mem_sum, " \n       集群cpu和内存可用量：", node_cpu_ava,
          node_mem_ava)
    return node_cpu_sum, node_mem_sum, node_cpu_ava, node_mem_ava


def scheduler():
    # s_dim = 2 * (len(node_cpu_sum))
    # a_dim = len(node_cpu_sum)
    # 获取要处理的100个task，结束
    observation = np.zeros([2, len(node_cpu_sum)])
    i = 0
    while i < len(node_cpu_sum):
        observation[0, i] = node_cpu_ava[i]
        observation[1, i] = node_mem_ava[i]
        i += 1
    observation[0, :] = (observation[0, :] - 0.99 * np.min(observation[0, :])) / (
            1.01 * np.max(observation[0, :]) - 0.99 * np.min(observation[0, :]))
    observation[1, :] = (observation[1, :] - 0.99 * np.min(observation[1, :])) / (
            1.01 * np.max(observation[1, :]) - 0.99 * np.min(observation[1, :]))
    s_dqn2_ = observation.reshape(-1)
    # print(observation)
    actions_value2, a_dqn2 = dqn2.choose_action(s_dqn2_)
    # print(actions_value2)
    return actions_value2


def update_redis():
    redis = StrictRedis(host='192.168.2.104', port=30246, db=0, password='root123456')
    m1_value = int((actions_value2[0][0] + 10) * 1000)
    n2_value = int((actions_value2[0][1] + 10) * 1000)
    n3_value = int((actions_value2[0][2] + 10) * 1000)
    n4_value = int((actions_value2[0][3] + 10) * 1000)
    redis.set('gzk8s-master1', m1_value)
    redis.set('gzk8s-node2', n2_value)
    redis.set('gzk8s-node3', n3_value)
    redis.set('gzk8s-node4', n4_value)
    print("2.DQN模型已更新四个节点分值分别为：", m1_value, "  ", n2_value, "  ", n3_value, "  ", n4_value)

    # print(redis.get('gzk8s-master1'))
    # print(redis.get('gzk8s-node2'))
    # print(redis.get('gzk8s-node3'))
    # print(redis.get('gzk8s-node4'))


if __name__ == "__main__":
    s_dim = 8
    a_dim = 4
    dqn2 = SealDQN2_DuelingDDQN(a_dim, s_dim)
    model_id = 132000
    dqn2.load_model(model_id)
    try:
        while True:
            print("##########正常运行###########")
            node_cpu_sum, node_mem_sum, node_cpu_ava, node_mem_ava = get_news()
            actions_value2 = scheduler()
            update_redis()
            time.sleep(1)
    except:
        print("##########错误：网络出现异常，获取或传输信息失败##########")
