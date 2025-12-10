import numpy as np


# ==========================================
# Part 1: 构建 d=3 广义泡利群
# ==========================================
def get_pauli_group_d3():
    """
    生成 d=3 的广义泡利算符基底 (9个矩阵)。
    P_ab = X^a * Z^b
    """
    # 定义三次单位根
    omega = np.exp(2j * np.pi / 3)

    # 定义基础矩阵 X (Shift) 和 Z (Clock)
    X = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=complex)

    Z = np.array([[1, 0, 0],
                  [0, omega, 0],
                  [0, 0, omega ** 2]], dtype=complex)

    ops = []
    # 生成所有 9 个组合: X^a Z^b
    for a in range(3):
        for b in range(3):
            # 矩阵幂和乘法
            Xa = np.linalg.matrix_power(X, a)
            Zb = np.linalg.matrix_power(Z, b)
            P = Xa @ Zb
            ops.append(P)

    return ops


# 为了避免重复计算，我们在全局生成一次泡利群
PAULI_OPS_D3 = get_pauli_group_d3()


# ==========================================
# Part 2: 计算 Magic (SRE M2)
# ==========================================
def calculate_magic_m2(psi):
    """
    输入: psi (归一化的 3维复数向量)
    输出: 2-Renyi Stabilizer Entropy (Magic)
    公式: M2 = -ln( 1/3 * sum( |<psi|P|psi>|^4 ) )
    """
    # 确保输入是 numpy array
    psi = np.array(psi, dtype=complex)

    # 简单的归一化检查 (防止传入非归一化态导致计算错误)
    norm = np.vdot(psi, psi).real
    if not np.isclose(norm, 1.0):
        # 如果没归一化，帮它归一化
        psi = psi / np.sqrt(norm)

    psi_dag = psi.conj().T
    total_sum = 0.0

    # 遍历 9 个泡利算符
    for P in PAULI_OPS_D3:
        # 计算期望值 <psi|P|psi>
        exp_val = psi_dag @ P @ psi

        # 关键步骤: 取模的4次方
        # 注意: exp_val 是复数
        val = np.abs(exp_val) ** 4
        total_sum += val

    # 计算 M2
    # 理论上 Stabilizer 态的 total_sum = 3.0
    # 所以 log 里面是 1.0，结果是 0

    # 防止数值误差导致 log(负数)
    argument = total_sum / 3.0

    # 如果 argument 非常接近 1 (误差范围内)，直接返回 0
    if np.isclose(argument, 1.0):
        return 0.0

    return -np.log(argument)


# ==========================================
# Part 3: 生成 Stabilizer 态 (复用之前的代码)
# ==========================================
def generate_qutrit_stabilizers():
    stabilizers = []
    omega = np.exp(2j * np.pi / 3)
    norm_factor = 1.0 / np.sqrt(3)

    # Group 1: Z-basis
    basis_Z = np.eye(3, dtype=complex)
    for k in range(3):
        stabilizers.append(basis_Z[:, k])

    # Group 2, 3, 4: Generalized bases
    for r in range(3):
        for k in range(3):
            vec = np.zeros(3, dtype=complex)
            for n in range(3):
                vec[n] = omega ** (r * (n ** 2) + k * n)
            stabilizers.append(vec * norm_factor)

    return stabilizers


# ==========================================
# Part 4: 主程序 - 验证
# ==========================================
if __name__ == "__main__":

    print(f"{'State Type':<25} | {'Raw Sum':<10} | {'Magic (M2)':<15} | {'Check'}")
    print("-" * 65)

    # 1. 验证所有 12 个 Stabilizer 态
    stabs = generate_qutrit_stabilizers()
    for i, psi in enumerate(stabs):
        magic = calculate_magic_m2(psi)

        # 我们可以顺便算出 sum(|<P>|^4)，理论值应该是 3.0
        # 仅仅为了展示 debug 信息
        psi_dag = psi.conj().T
        raw_sum = sum([np.abs(psi_dag @ P @ psi) ** 4 for P in PAULI_OPS_D3])

        # 判断是否为 0 (允许微小误差 1e-15)
        is_zero = np.isclose(magic, 0.0, atol=1e-10)
        status = "PASS" if is_zero else "FAIL"

        print(f"Stabilizer #{i + 1:<2}             | {raw_sum:.5f}    | {magic:.10f}    | {status}")

    print("-" * 65)

    # 2. 验证一个非 Stabilizer 态 (Magic 态)
    # 例如 T 态的类比，或者随机态
    # 构造一个随机归一化态
    np.random.seed(42)
    random_vec = np.random.rand(3) + 1j * np.random.rand(3)
    random_vec = random_vec / np.linalg.norm(random_vec)

    magic_rand = calculate_magic_m2(random_vec)
    psi_dag = random_vec.conj().T
    raw_sum_rand = sum([np.abs(psi_dag @ P @ random_vec) ** 4 for P in PAULI_OPS_D3])

    print(f"Random Magic State        | {raw_sum_rand:.5f}    | {magic_rand:.10f}    | {'Non-Zero (Expected)'}")

    # 3. 验证最大 Magic 态 (N=3 的 Hesse function 最大值点)
    # 一个已知的 High Magic State 是 "Strange state": (1, 1, 0) / sqrt(2) ?
    # 或者简单的叠加态，如 (1, 1, 0) 在某些基下可能有 Magic

    # 试一个简单的二分叠加态 (1, 1, 0)
    psi_super = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    magic_super = calculate_magic_m2(psi_super)
    raw_sum_super = sum([np.abs(psi_super.conj().T @ P @ psi_super) ** 4 for P in PAULI_OPS_D3])
    print(f"Superpos [1,1,0]/sqrt(2)  | {raw_sum_super:.5f}    | {magic_super:.10f}    | {'Non-Zero (Expected)'}")