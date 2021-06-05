import sys
import time
import math
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio',
                               'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
rcParams["font.size"] = 14
startTime = time.time()
NStart = 7200


def calculate_total_distance(order, distance_matrix):
    """Calculate total distance traveled for given visit order"""
    idx_from = np.array(order)
    idx_to = np.array(order[1:] + [order[0]])
    distance_arr = distance_matrix[idx_from, idx_to]
    return np.sum(distance_arr)


def visualize_visit_order(starcharts, order=None, distance=None, savename="starchart", nstart=0,
                          ngorder=None, ngdist=None):
    stepstr = "：Count="+str(nstart).zfill(5)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor('black')
    if order == None:
        for star in range(nstar):
            ax.plot(starcharts.loc[star, "RA"], starcharts.loc[star, "DE"],
                    marker="o", color="yellow", linestyle=None)
            ax.annotate("[00]"+starcharts.loc[star, "NAME"],
                        xy=(starcharts.loc[star, "RA"],
                            starcharts.loc[star, "DE"]),
                        size=9, color="white")
    else:
        if not ngorder == None:
            # add point of departure
            ngroute = np.array(ngorder + [ngorder[0]])
            xng_arr = np.array(starcharts["RA"])[ngroute]
            yng_arr = np.array(starcharts["DE"])[ngroute]
            ax.plot(xng_arr, yng_arr, '-', color="orange", alpha=0.5)

        route = np.array(order + [order[0]])  # add point of departure
        x_arr = np.array(starcharts["RA"])[route]
        y_arr = np.array(starcharts["DE"])[route]
        n_arr = np.array(starcharts["NAME"])[route]

        ax.plot(x_arr, y_arr, 'o-', color="cyan", markerfacecolor="yellow")
        for star in range(len(order)):
            ax.annotate("["+str(star).zfill(2)+"]"+n_arr[star],
                        xy=(x_arr[star], y_arr[star]), size=9, color="white")

    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    ax.set_title("1等星による巡回セールスマン座"+stepstr)
    if distance is not None:
        ax.text(20, -105, "総延長:"+str(distance))
    plt.savefig("./export/"+savename+"@NS"+str(nstart)+".png")
    plt.clf()
    plt.close()


def apply_2opt_exchange(visit_order, i, j):
    """Apply 2-opt exhanging on visit order"""

    tmp = visit_order[i + 1: j + 1]
    tmp.reverse()
    visit_order[i + 1: j + 1] = tmp


def calculate_2opt_exchange_cost(visit_order, i, j, distance_matrix):
    """Calculate the difference of cost by applying given 2-opt exchange"""
    n_cities = len(visit_order)
    a, b = visit_order[i], visit_order[(i + 1) % n_cities]
    c, d = visit_order[j], visit_order[(j + 1) % n_cities]

    cost_before = distance_matrix[a, b] + distance_matrix[c, d]
    cost_after = distance_matrix[a, c] + distance_matrix[b, d]
    return cost_after - cost_before


def improve_with_2opt(visit_order, distance_matrix):
    """Check all 2-opt neighbors and improve the visit order"""
    n_cities = len(visit_order)
    cost_diff_best = 0.0
    i_best, j_best = None, None

    for i in range(0, n_cities - 2):
        for j in range(i + 2, n_cities):
            if i == 0 and j == n_cities - 1:
                continue

            cost_diff = calculate_2opt_exchange_cost(
                visit_order, i, j, distance_matrix)

            if cost_diff < cost_diff_best:
                cost_diff_best = cost_diff
                i_best, j_best = i, j

    if cost_diff_best < 0.0:
        visit_order_new = apply_2opt_exchange(visit_order, i_best, j_best)
        return visit_order_new
    else:
        return None


def local_search(visit_order, distance_matrix, improve_func):
    """Main procedure of local search"""
    cost_total = calculate_total_distance(visit_order, distance_matrix)

    while True:
        improved = improve_func(visit_order, distance_matrix)
        if not improved:
            break

        visit_order = improved

    return visit_order

def get_distance_m(lat1, lon1, lat2, lon2):
    """
    ２点間の距離(m)
    球面三角法を利用した簡易的な距離計算
    GoogleMapAPIのgeometory.computeDistanceBetweenのロジック
    https://www.suzu6.net/posts/167-php-spherical-trigonometry/
    """
    R = 1.00  # 赤道半径
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    diff_lon = lon1 - lon2
    dist = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(diff_lon)
    return R * math.acos(min(max(dist, -1.0), 1.0))


if __name__ == "__main__":
    # 星図データを読み込み
    starcharts = pd.read_csv("./data/hip_lite_major.csv")
    # 一等星の抽出
    starcharts = starcharts[starcharts["Magnitude"] < 1.5]
    # 恒星名の結合
    starcharts = pd.merge(starcharts, pd.read_csv(
        "./data/hip_proper_name.csv"), on="HIP", how="left").fillna({"NAME": ""})
    # 赤経の時、分、秒を結合して、degに変換
    starcharts["RA"] = starcharts["RA.h"] + \
        starcharts["RA.m"]/60+starcharts["RA.s"]/3600
    starcharts["RA"] = 360 * starcharts["RA"] / 24
    # 赤緯の度、分、秒を結合して、degに変換
    starcharts["DE"] = starcharts["DE.deg"] + \
        starcharts["DE.m"]/60+starcharts["DE.s"]/3600
    starcharts["DE"].mask(starcharts["DE.c"] == 0, -
                          starcharts["DE"], inplace=True)
    nstar = len(starcharts)

    # 総移動距離の計算
    star_x = np.array(starcharts["RA"])
    star_y = np.array(starcharts["DE"])
    distance_matrix = np.zeros([len(star_x),len(star_x)])
    for i in range(len(star_x)):
        for j in range(len(star_y)):
            sdis = get_distance_m(lat1 = star_y[j], lon1= star_x[j], lat2= star_y[i], lon2= star_x[i])
            if sdis < 1e-5:
                continue
            else:
                distance_matrix[i,j] = sdis

    print(np.round(distance_matrix))
    sys,exit()
   

    # 初期状態の星図を作成
    visualize_visit_order(starcharts)

    oeder_best = None
    score_best = sys.float_info.max

    for ncount in range(1, NStart+1):
        print("--------第", str(ncount), "回目の実行--------")
        print("初期解")
        rdorder = list(np.random.permutation(nstar))
        print('>訪問順序 = {}'.format(rdorder))
        total = calculate_total_distance(rdorder, distance_matrix)
        print('>総移動距離 = {}'.format(total))
        #visualize_visit_order(starcharts, rdorder, distance=total)

        improved = local_search(rdorder, distance_matrix, improve_with_2opt)
        total_distance = calculate_total_distance(improved, distance_matrix)
        #visualize_visit_order(starcharts, improved, distance=total_distance)
        print('近傍探索適用後の総移動距離 = {}'.format(total_distance))

        if total_distance < score_best:
            print("最短経路がアップデートされました")
            score_best = total_distance
            order_best = improved

        visualize_visit_order(starcharts, order_best, distance=score_best,
                              savename="starchart", nstart=ncount,
                              ngorder=improved)
        print("計算時間[sec]", time.time()-startTime)
        print("1データセットの平均時間", (time.time()-startTime)/ncount+1)

    print("--------最終結果--------")
    visualize_visit_order(starcharts, order_best, distance=score_best,
                          savename="starchart-best", nstart=NStart)
    print('近傍探索適用後の総移動距離 = {}'.format(score_best))
    print("総計算時間[sec]", time.time()-startTime)
    print("1データセットの平均時間", (time.time()-startTime)/NStart)
