import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. 基础物理模块 (复用)
# ==========================================
def get_pauli_group_d3():
    omega = np.exp(2j * np.pi / 3)
    X = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    Z = np.array([[1, 0, 0], [0, omega, 0], [0, 0, omega ** 2]], dtype=complex)
    ops = []
    for a in range(3):
        for b in range(3):
            ops.append(np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b))
    return ops


PAULI_OPS = get_pauli_group_d3()


def generate_12_stabilizers():
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
    return stabs


STABILIZERS = generate_12_stabilizers()


# 计算单个态的 Magic
def calc_magic_m2(psi):
    psi = psi / np.linalg.norm(psi)
    psi_dag = psi.conj().T
    sum_val = 0.0
    for P in PAULI_OPS:
        sum_val += np.abs(psi_dag @ P @ psi) ** 4
    val = sum_val / 3.0
    if val >= 0.99999999: return 0.0
    return -np.log(val)


# PDG 2024 参数
def rad(deg): return deg * np.pi / 180


PARAMS_NO = {
    't12': rad(33.41), 't13': rad(8.58), 't23': rad(49.1), 'dcp': rad(180),
    'dm2_21': 7.41e-5, 'dm2_31': 2.507e-3
}
PARAMS_IO = {
    't12': rad(33.41), 't13': rad(8.63), 't23': rad(49.5), 'dcp': rad(180),
    'dm2_21': 7.41e-5, 'dm2_31': -2.486e-3 + 7.41e-5
}


def get_pmns(p):
    c12, s12 = np.cos(p['t12']), np.sin(p['t12'])
    c13, s13 = np.cos(p['t13']), np.sin(p['t13'])
    c23, s23 = np.cos(p['t23']), np.sin(p['t23'])
    eid = np.exp(1j * p['dcp'])
    U = np.zeros((3, 3), dtype=complex)
    U[0, 0], U[0, 1], U[0, 2] = c12 * c13, s12 * c13, s13 * np.exp(-1j * p['dcp'])
    U[1, 0] = -s12 * c23 - c12 * s23 * s13 * eid
    U[1, 1] = c12 * c23 - s12 * s23 * s13 * eid
    U[1, 2] = s23 * c13
    U[2, 0] = s12 * s23 - c12 * c23 * s13 * eid
    U[2, 1] = -c12 * s23 - s12 * c23 * s13 * eid
    U[2, 2] = c23 * c13
    return U


# ==========================================
# 2. 计算主逻辑
# ==========================================
def calculate_integral_curves(E_eV):
    E_GeV = E_eV * 1e-9

    # 设定距离：覆盖 3 个太阳振荡周期 (3 * Solar Cycle)
    # L_sol_km = 4*pi*E / dm2_21 (approx)
    # Precise: Phase = 1.267 * dm2 * L / E. Cycle -> Phase=2pi
    # L_cycle = 2pi * E / (1.267 * dm2)
    L_sol_km = (2 * np.pi * E_GeV) / (1.267 * PARAMS_NO['dm2_21'])
    L_max = 5.0 * L_sol_km

    # 采样点：需要极高分辨率以捕捉大气振荡
    # Solar ~ 33 * Atm. We calculate 3 Solar cycles.
    # Total Atm cycles ~ 100. Need ~20 points per Atm cycle.
    # Total points ~ 2000 - 3000 is safe.
    steps = 5000
    L_vals = np.linspace(0, L_max, steps)
    dx = L_vals[1] - L_vals[0]

    # 准备矩阵
    U_NO = get_pmns(PARAMS_NO)
    U_IO = get_pmns(PARAMS_IO)
    factor = 1.267 / E_GeV

    density_no = []
    density_io = []

    # 循环计算 Magic 密度 (Instantaneous Average Magic)
    print(f"Computing for E = {E_eV} eV over {L_max:.2e} km...")

    for L in L_vals:
        # --- NO ---
        D_NO = np.diag([1.0,
                        np.exp(-1j * factor * PARAMS_NO['dm2_21'] * L),
                        np.exp(-1j * factor * PARAMS_NO['dm2_31'] * L)])
        S_NO = U_NO @ D_NO @ U_NO.conj().T

        # Avg over 12 stabilizers
        m_sum = 0
        for psi0 in STABILIZERS:
            m_sum += calc_magic_m2(S_NO @ psi0)
        density_no.append(m_sum / 12.0)

        # --- IO ---
        D_IO = np.diag([1.0,
                        np.exp(-1j * factor * PARAMS_IO['dm2_21'] * L),
                        np.exp(-1j * factor * PARAMS_IO['dm2_31'] * L)])
        S_IO = U_IO @ D_IO @ U_IO.conj().T

        m_sum = 0
        for psi0 in STABILIZERS:
            m_sum += calc_magic_m2(S_IO @ psi0)
        density_io.append(m_sum / 12.0)

    # 计算积分 (Cumulative Integral)
    # cumsum 近似积分
    integral_no = np.cumsum(density_no) * dx
    integral_io = np.cumsum(density_io) * dx

    # 计算差值
    diff = integral_no - integral_io

    return L_vals, integral_no, integral_io, diff


# ==========================================
# 3. 绘图
# ==========================================
if __name__ == "__main__":
    E_target = 1.0  # 1 eV
    L, I_no, I_io, Diff = calculate_integral_curves(E_target)

    # 创建双面板图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 图 1: 累积 Magic (积分值)
    # 这将是两条斜率略有不同的直线
    ax1.plot(L, I_no, 'b-', label='Normal Ordering (NO)', linewidth=2)
    ax1.plot(L, I_io, 'r--', label='Inverted Ordering (IO)', linewidth=2)
    ax1.set_ylabel(r"Cumulative Magic $\int \langle \mathcal{M}_2 \rangle dL$", fontsize=12)
    ax1.set_title(f"Cumulative Magic Action for E = {E_target} eV\n(Range: 3 Solar Oscillation Cycles)", fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 图 2: 累积差值 (NO - IO)
    # 这是最关键的图
    ax2.plot(L, Diff, 'k-', linewidth=2, label=r'Difference $\Delta = \mathcal{A}_{NO} - \mathcal{A}_{IO}$')

    # 填充颜色：绿色表示 NO 更好 (Diff < 0)，红色表示 IO 更好 (Diff > 0)
    ax2.fill_between(L, Diff, 0, where=(Diff < 0), color='green', alpha=0.1, label='NO Preferred (Lower Cost)')
    ax2.fill_between(L, Diff, 0, where=(Diff > 0), color='red', alpha=0.1, label='IO Preferred')

    # 画一条 0 线
    ax2.axhline(0, color='gray', linestyle='--')

    ax2.set_ylabel(r"Difference (NO - IO)", fontsize=12)
    ax2.set_xlabel("Distance L (km)", fontsize=12)
    ax2.legend(loc='lower left', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 强制让 X 轴使用科学计数法 (因为 1eV 对应的 km 很小)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()