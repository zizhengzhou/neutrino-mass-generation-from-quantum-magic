import numpy as np
import matplotlib.pyplot as plt


def check_pmns_structure():
    # PDG NO 参数
    t12 = np.deg2rad(33.41)
    t13 = np.deg2rad(8.58)
    t23 = np.deg2rad(49.1)

    c12, s12 = np.cos(t12), np.sin(t12)
    c13, s13 = np.cos(t13), np.sin(t13)
    c23, s23 = np.cos(t23), np.sin(t23)

    delta_range = np.linspace(0, 360, 361)
    costs = []

    for deg in delta_range:
        dcp = np.deg2rad(deg)
        eid = np.exp(1j * dcp)

        # 构建 U 的关键元素 (这里只写几个关键项，或者构建全矩阵)
        # 简单起见，构建全矩阵
        U = np.zeros((3, 3), dtype=complex)
        U[0, 0], U[0, 1], U[0, 2] = c12 * c13, s12 * c13, s13 * np.exp(-1j * dcp)
        U[1, 0] = -s12 * c23 - c12 * s23 * s13 * eid
        U[1, 1] = c12 * c23 - s12 * s23 * s13 * eid
        U[1, 2] = s23 * c13
        U[2, 0] = s12 * s23 - c12 * c23 * s13 * eid
        U[2, 1] = -c12 * s23 - s12 * c23 * s13 * eid
        U[2, 2] = c23 * c13

        # 计算 IPR (sum |U|^4)
        # 这个值越小，Magic 越大
        ipr = np.sum(np.abs(U) ** 4)
        costs.append(ipr)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(delta_range, costs, 'k-', linewidth=2)
    plt.title("Inverse Participation Ratio of PMNS Matrix vs $\delta_{CP}$")
    plt.xlabel(r"$\delta_{CP}$ (Degree)")
    plt.ylabel(r"Sum of $|U_{ij}|^4$ (Lower is Higher Magic)")

    # 标出极值点
    min_idx = np.argmin(costs)  # IPR 最小 -> Magic 最大
    max_idx = np.argmax(costs)  # IPR 最大 -> Magic 最小

    # 寻找局部极值
    from scipy.signal import argrelextrema
    costs = np.array(costs)
    local_min_indices = argrelextrema(costs, np.less)[0]  # IPR 局部最小 -> Magic 局部最大

    print(f"Global Minimum Magic (Max IPR) at: {delta_range[max_idx]} deg")
    print(f"Local Maxima Magic (Min IPR) at: {delta_range[local_min_indices]} deg")

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    check_pmns_structure()