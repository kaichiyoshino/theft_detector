#%%
import polars as pl 
import numpy as np 
import dataclasses
import argparse


def generate_voltage(house_count: int,  mu: float) -> list[float]:
    '''
    一定の抵抗値に対する電圧を生成する関数

    引数:
        house_count: 家の数
        mu: 正規分布の標準偏差(Vがどれだけばらけるか)

    戻り値:
        list[float]: 電圧値のリスト
    '''

    voltages = np.array([np.random.default_rng().normal(100, mu) for _ in range(house_count)])
    return voltages

#ツリー形状の簡単なモデルだけ実装
def generate_current(V: list[float], R: list[int]) -> list[float]:
    '''
    一定の電圧値に対する電流を生成する関数

    引数:
        V: 電圧値
        R: 抵抗値

    戻り値:
        list[float]: 電流値のリスト
    '''

    currents = []
    for point in range(len(V)):
        if point == 0:
            currents.append((V[point + 1] - V[point])/R[point])
        elif point == len(V) - 1:
            currents.append((V[point] - V[point - 1])/R[point - 1])
        else:
            currents.append((V[point + 1] - V[point])/R[point] + (V[point] - V[point - 1])/R[point - 1])
    
    #電流の合計が0になるように調整
    currents_sum = sum(currents)
    currents[0] -= currents_sum
    V[0] += R[0] * currents_sum
    return np.array(currents) 

def generate_row(I: list[float], V: list[float]) -> list[float]:
    '''
    I, Vをまとめて一行分のデータを生成する関数
    '''

    row = []
    for i in range(len(V)):
        row.append(I[i])
        row.append(V[i])
    
    return np.array(row)

def generate_data(house_count: int, time_count: int, mu: float, step: int = 1) -> list[list[float]]:

    '''
    ツリー形状の電力網モデルにしたがってI,Vデータを生成する関数

    引数:
        house_count: 家の数
        time_count: 何秒分のデータを生成するか
        steps(default = 1s): 何秒値でデータを生成するか
        mu: 正規分布の標準偏差(Vがどれだけばらけるか)

    戻り値:
        Record: I,Vデータを含むRecordオブジェクト
    
    例:
        2件の家のデータを10秒分生成する場合
        generate_data(2, 10, 1, 1) -> Record(data)
    
    注意:
        生成されるデータは以下の形式
        | I1 | V1 | I2 | V2 | ... | In | Vn |
    '''

    data_label = [f"I{i//2 + 1}" if i%2 == 0 else f"V{i//2 + 1}" for i in range(house_count * 2)]
    _voltage = generate_voltage(house_count, mu = mu)
    _current = generate_current(_voltage, [1 for _ in range(house_count)])
    origin_data = np.array([generate_row(_current, _voltage)])

    for _ in range(0,time_count - 1,step):
        voltage = generate_voltage(house_count, mu = mu)
        current = generate_current(voltage, [1 for _ in range(house_count)])
        origin_data = np.vstack((origin_data, generate_row(current, voltage)))

    # data = pl.DataFrame(
    #     origin_data,
    #     schema = data_label
    # )

    return origin_data


def conver_pldataframe(data: list, house_count: int) -> pl.DataFrame:
    data_label = [f"I{i//2 + 1}" if i%2 == 0 else f"V{i//2 + 1}" for i in range(house_count * 2)]
    created_data = pl.DataFrame(data, schema = data_label).transpose()
    return created_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Electricity Data')
    parser.add_argument("house_count", help="Number of houses", type=int)
    parser.add_argument("time_count", help="Number of seconds", type=int)
    parser.add_argument("mu", help="Standard deviation of normal distribution", type=float)
    args = parser.parse_args()

    original_data = generate_data(args.house_count, args.time_count, args.mu)
    data = conver_pldataframe(original_data, args.house_count)
    
    print(data)


    
