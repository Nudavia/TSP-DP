import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from time import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank()
Size = Comm.Get_size()


class GA:
    def __init__(self, filepath):
        self.filepath = filepath  # 数据源
        self.pos = []  # 坐标
        self.cityNum = 0  # 城市数量，染色体长度
        self.citySet = 0  # 剩余城市集合
        self.dist = []  # 距离矩阵
        self.bestDistance = 0  # 最佳长度
        self.bestPath = []  # 最佳路径
        self.DP = []  # 动态规划网格
        self.next_city = []  # 记录dp过程中每一个状态所选择的下一个城市，为了获取最优路径

    # 读取文件
    def readfile(self, filepath):
        infile = open(filepath)
        i = 0
        for line in infile:
            linedata = line.strip().split()
            self.pos.append([float(linedata[1]), float(linedata[2])])
            i = i + 1
            if i > 15:  # 控制城市数量，动态规划不能解决大规模TSP问题
                break
        infile.close()

    # 初始化dist矩阵
    def init(self):
        self.cityNum = len(self.pos)
        self.citySet = 1 << (self.cityNum - 1)
        self.next_city = np.zeros([self.cityNum, self.citySet], dtype=int)
        self.dist = np.zeros([self.cityNum, self.cityNum], dtype=int)
        self.DP = np.zeros([self.cityNum, self.citySet], dtype=int)
        for i in range(self.cityNum):
            for j in range(i, self.cityNum):
                self.dist[i][j] = self.dist[j][i] = self.distance(self.pos[i], self.pos[j])

    # 计算欧氏距离矩阵
    @staticmethod
    def distance(pos1, pos2):
        return np.around(np.sqrt(np.sum(np.power(np.array(pos1) - np.array(pos2), 2))))

    # 开始DP
    def run(self):
        self.readfile(self.filepath)
        self.init()
        for cur_city in range(self.cityNum):  # 处理剩余城市为空集的情况，此时直接回到0号城市即可
            self.DP[cur_city][0] = self.dist[cur_city][0]
        for last_cities in range(1, self.citySet):  # 选定剩余城市集合
            for cur_city in range(self.cityNum):  # 选定当前城市
                self.DP[cur_city][last_cities] = -1  # 唯一确定了状态,初值为-1，表示最大
                if cur_city != 0 and ((last_cities >> (cur_city - 1)) & 1) == 1:  # 如果当前城市在剩余城市里直接continue
                    continue  # 因为从0开始，所以0肯定不在
                min_k = 1
                for next_city in range(1, self.cityNum):  # 选择下一个城市,从1开始
                    if ((last_cities >> (next_city - 1)) & 1) == 0:  # 如果所选下一个城市不再剩余城市continue
                        continue
                    next_last_cities = last_cities ^ (1 << (next_city - 1))  # 去除了所选城市之后的剩余城市集合
                    if self.DP[cur_city][last_cities] == -1 or self.DP[cur_city][last_cities] > self.dist[cur_city][
                        next_city] + self.DP[next_city][next_last_cities]:
                        self.DP[cur_city][last_cities] = self.dist[cur_city][next_city] + self.DP[next_city][
                            next_last_cities]  # 状态转移方程
                        min_k = next_city
                    self.next_city[cur_city][last_cities] = min_k

        self.bestDistance = self.DP[0][self.citySet - 1]
        cur_city = 0
        last_cities = self.citySet - 1
        self.bestPath.append(cur_city)
        while len(self.bestPath) < self.cityNum:
            cur_city = self.next_city[cur_city][last_cities]
            self.bestPath.append(cur_city)
            last_cities = last_cities ^ (1 << (cur_city - 1))

        print("最佳路径:" + str(self.bestPath))
        print("最短距离:" + str(self.bestDistance))

    # 显示结果
    def show(self):
        plt.title('TSP-GA')
        ax1 = plt.subplot(211)
        ax1.set_title('原始坐标')
        ax1.set_xlabel('x坐标')
        ax1.set_ylabel('y坐标')
        for point in self.pos:
            plt.plot(point[0], point[1], marker='o', color='k')

        ax2 = plt.subplot(212)
        ax2.set_title('线路')
        ax2.set_xlabel('x坐标')
        ax2.set_ylabel('y坐标')
        for point in self.pos:
            plt.plot(point[0], point[1], marker='o', color='k')
        for i in range(1, self.cityNum):
            plt.plot([self.pos[self.bestPath[i]][0], self.pos[self.bestPath[i - 1]][0]],
                     [self.pos[self.bestPath[i]][1], self.pos[self.bestPath[i - 1]][1]], color='g')
        plt.plot([self.pos[self.bestPath[0]][0], self.pos[self.bestPath[self.cityNum - 1]][0]],
                 [self.pos[self.bestPath[0]][1], self.pos[self.bestPath[self.cityNum - 1]][1]], color='g')
        plt.show()


def main():
    ga = GA(filepath="data/bayg29.txt")
    t1 = time()
    ga.run()
    t2 = time()
    print("耗时:" + str(t2 - t1))
    ga.show()


if __name__ == '__main__':
    main()
