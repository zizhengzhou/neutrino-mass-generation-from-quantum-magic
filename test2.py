import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


# ==========================================
# 1. 全局预计算 (Global Pre-computation)
# ==========================================

def get_pauli_group_d3():
    """生成 9 个广义泡利矩阵 (3x3), Stack 成 (9, 3, 3) 张量"""
    omega = np.exp(2j * np.pi / 3)
    X = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    Z = np.array([[1, 0, 0], [0, omega, 0], [0, 0, omega ** 2]], dtype=complex)
    ops = []
    for a in range(3):
        for b in range(3):
            ops.append(np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b))
    return np.array(ops)  # Shape: (9, 3, 3)


def get_12_stabilizers():
    """生成 12 个初始态, Stack 成 (12, 3) 张量"""
    stabs = []
    omega = np.exp(2j * np.pi / 3)
    norm = 1.0 / np.sqrt(3)
    # Z basis
    for k in range(3):
        v = np.zeros(3, dtype=complex);
        v[k] = 1.0
        stabs.append(v)
    # X, XZ, XZ^2 bases
    for r in range(3):
        for k in range(3):
            v = np.zeros(3, dtype=complex)
            for n in range(3):
                v[n] = omega ** (r * (n ** 2) + k * n)
            stabs.append(v * norm)
    return np.array(stabs).T  # Transpose to Shape: (3, 12) -> Column vectors


# 全局变量，利用 Shared Memory 避免 Pickling 开销
PAULI_TENSOR = get_pauli_group_d3()  # (9, 3, 3)
STAB_MATRIX = get_12_stabilizers()  # (3, 12)


# ==========================================
# 2. 向量化计算核心 (Vectorized Core)
# ==========================================

def fast_magic_integral(args):
    """
    单个进程的工作函数: 计算指定 deltaCP 下的积分
    """
    dcp_deg, ordering = args
    dcp_rad = np.deg2rad(dcp_deg)

    # --- A. 参数设定 (PDG 2024) ---
    t12 = np.deg2rad(33.41)
    dm2_21 = 7.41e-5

    if ordering == 'NO':
        t13 = np.deg2rad(8.58)
        t23 = np.deg2rad(49.1)
        dm2_31 = 2.507e-3
    else:  # IO
        t13 = np.deg2rad(8.63)
        t23 = np.deg2rad(49.5)
        # IO转换: dm2_31 = dm2_32 + dm2_21
        dm2_31 = -2.486e-3 + 7.41e-5

    # --- B. 构建 PMNS 矩阵 U ---
    c12, s12 = np.cos(t12), np.sin(t12)
    c13, s13 = np.cos(t13), np.sin(t13)
    c23, s23 = np.cos(t23), np.sin(t23)
    eid = np.exp(1j * dcp_rad)

    U = np.array([
        [c12 * c13, s12 * c13, s13 * np.exp(-1j * dcp_rad)],
        [-s12 * c23 - c12 * s23 * s13 * eid, c12 * c23 - s12 * s23 * s13 * eid, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * eid, -c12 * s23 - s12 * c23 * s13 * eid, c23 * c13]
    ], dtype=complex)

    # --- C. 空间设置 (5个太阳周期) ---
    E_GeV = 1e-9  # 1 eV
    # 太阳振荡波长 L_sol (km)
    L_sol = 5 * np.pi * E_GeV / (2.534 * dm2_21) * 1.973e-16 * 1e18  # 手动单位换算或使用简易系数
    # 简易系数公式: Phase = 1.267 * dm2 * L / E
    # 2pi = 1.267 * dm2_21 * L / E  => L = 2pi * E / (1.267 * dm2)
    L_cycle = (2 * np.pi * E_GeV) / (1.267 * dm2_21)

    L_max = 10.0 * L_cycle
    steps = 5000
    L_vals = np.linspace(0, L_max, steps)  # Shape (N,)
    dx = L_vals[1] - L_vals[0]

    # --- D. 向量化演化 (在质量基下计算) ---
    # 1. 将所有 Stabilizer 转换到 质量本征基: psi_mass = U_dagger @ psi_flavor
    # STAB_MATRIX shape: (3, 12)
    psi_mass_0 = U.conj().T @ STAB_MATRIX  # Shape (3, 12)

    # 2. 计算演化相位因子 (Vectorized over L)
    factor = 1.267 / E_GeV
    # phi matrix: shape (3, N)
    # m1 (idx 0) phase is 0 (relative), m2 (idx 1), m3 (idx 2)
    phases = np.zeros((3, steps), dtype=complex)
    phases[0, :] = 1.0
    phases[1, :] = np.exp(-1j * factor * dm2_21 * L_vals)
    phases[2, :] = np.exp(-1j * factor * dm2_31 * L_vals)

    # 3. 演化: psi_mass(L) = phases * psi_mass_0
    # 利用广播: (3, N, 1) * (3, 1, 12) -> (3, N, 12)
    psi_mass_L = phases[:, :, np.newaxis] * psi_mass_0[:, np.newaxis, :]
    # Transpose to (N, 3, 12) for matrix mult
    psi_mass_L = psi_mass_L.transpose(1, 0, 2)

    # 4. 转回味本征基: psi_flavor(L) = U @ psi_mass(L)
    # (3, 3) @ (N, 3, 12) -> (N, 3, 12)
    # 需要对 N 进行广播矩阵乘法
    psi_L = U @ psi_mass_L
    # Current shape: (N, 3, 12) -> N moments, 3 dim, 12 states

    # --- E. 向量化 Magic 计算 (Einsum) ---

    # 目标: 计算 |<psi|P|psi>|^4
    # PAULI_TENSOR: (9, 3, 3) -> (P, i, j)
    # psi_L: (N, i, S) where S=12
    # Expectation E[n, s, p] = sum_{i,j} conj(psi[n,i,s]) * Pauli[p,i,j] * psi[n,j,s]

    # einsum string: 'nis, pij, njs -> nsp'
    # n: time step, s: stabilizer index, p: pauli index, i/j: vector dim
    exp_vals = np.einsum('nis, pij, njs -> nsp', psi_L.conj(), PAULI_TENSOR, psi_L)

    # 计算概率分布 (mod squared, then squared again for Renyi-2 sum)
    # P_vals = |<P>|^4
    probs_4 = np.abs(exp_vals) ** 4

    # Sum over Pauli index (p) -> (N, S)
    sum_p = np.sum(probs_4, axis=2)

    # Magic M2 = -ln(sum / 3)
    # Handle numerical stability where sum ~ 3.0
    val = sum_p / 3.0
    # Create mask for valid logs
    magic_vals = np.zeros_like(val)
    mask = val < 0.99999999
    magic_vals[mask] = -np.log(val[mask])

    # Average over 12 stabilizers -> (N,)
    mean_magic_L = np.mean(magic_vals, axis=1)

    # --- F. 积分 ---
    # cumulative integral is not needed for the plot, just the final total integral
    total_integral = np.trapz(mean_magic_L, dx=dx)

    return total_integral


# ==========================================
# 3. 主控制程序
# ==========================================

if __name__ == "__main__":
    # 扫描设置
    dcp_range = np.arange(0, 361, 5)  # 0, 5, ..., 360 (73 points)

    tasks_no = [(d, 'NO') for d in dcp_range]
    tasks_io = [(d, 'IO') for d in dcp_range]

    start_t = time.time()

    # 并行计算
    # 根据你的CPU核心数调整 processes，建议 8-16
    with Pool(processes=60) as pool:
        res_no = pool.map(fast_magic_integral, tasks_no)
        res_io = pool.map(fast_magic_integral, tasks_io)

    end_t = time.time()
    print(f"Calculation done in {end_t - start_t:.2f} seconds.")

    # ==========================================
    # 4. 绘图 (修正版)
    # ==========================================
    # 强制关闭外部 LaTeX 依赖，防止解析错误
    plt.rcParams['text.usetex'] = False

    plt.figure(figsize=(10, 7))

    # 转换成 numpy array 方便操作
    y_no = np.array(res_no)
    y_io = np.array(res_io)
    x = dcp_range

    # 绘制 NO 和 IO 曲线
    plt.plot(x, y_no, 'b-o', markersize=4, label='Normal Ordering (NO)', linewidth=2)
    plt.plot(x, y_io, 'r-s', markersize=4, label='Inverted Ordering (IO)', linewidth=2)

    # 标注 Best Fit Points
    idx_no_best = np.abs(x - 197).argmin()
    idx_io_best = np.abs(x - 286).argmin()

    plt.scatter(x[idx_no_best], y_no[idx_no_best], color='blue', s=150, zorder=5, marker='*',
                label='NO Best Fit (197 deg)')
    plt.scatter(x[idx_io_best], y_io[idx_io_best], color='red', s=150, zorder=5, marker='*',
                label='IO Best Fit (286 deg)')

    # 修改标题和标签，确保使用原始字符串 r""
    plt.title(r"Cumulative Magic Action (5 Solar Cycles) vs $\delta_{CP}$", fontsize=14)
    plt.xlabel(r"CP Violation Phase $\delta_{CP}$ (Degree)", fontsize=12)
    plt.ylabel(r"Total Integrated Magic", fontsize=12)

    # 设置 X 轴刻度
    plt.xticks(np.arange(0, 361, 45))
    plt.xlim(0, 360)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper center')  # 调整图例位置防止遮挡

    # 额外：画一个小插图显示差值 (NO - IO)
    # 手动定义插图位置 [left, bottom, width, height] (0-1 归一化坐标)
    left, bottom, width, height = [0.2, 0.55, 0.35, 0.25]
    ax2 = plt.axes([left, bottom, width, height])

    diff = y_no - y_io
    ax2.plot(x, diff, 'k-', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # 填充颜色
    ax2.fill_between(x, diff, 0, where=(diff < 0), color='green', alpha=0.3)
    ax2.fill_between(x, diff, 0, where=(diff > 0), color='red', alpha=0.3)

    ax2.set_title("Difference (NO - IO)", fontsize=10)
    ax2.set_xticks(np.arange(0, 361, 90))
    # 稍微调小插图字体
    ax2.tick_params(axis='both', which='major', labelsize=8)

    # !!! 关键修改：删除 plt.tight_layout()，使用手动调整 !!!
    # plt.tight_layout()  <-- 这行会导致报错，已删除
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95)

    plt.show()