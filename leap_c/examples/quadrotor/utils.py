import casadi as cs
import numpy as np
import quaternion
import yaml


def quaternion_multiply_casadi(q, r):
    """
    Computes the quaterion product of two quaternions using CasADi symbolic vectors.

    Parameters:
    q1, q2: casadi.SX or casadi.MX of shape (4,)

    Returns:
    casadi.SX or casadi.MX: The resulting quaternion (4,)
    """
    t0 = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t1 = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t2 = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t3 = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]

    return cs.vertcat(t0, t1, t2, t3)


import casadi as ca


def quaternion_inverse_casadi(q):
    return cs.vertcat(q[0], -q[1], -q[2], -q[3]) / cs.sum1(q ** 2)


def quaternion_rotate_vector_casadi(q, v):
    """
    Rotates a 3D vector using a quaternion in CasADi.

    Parameters:
    q : casadi.SX or casadi.MX (4,)
    v : casadi.SX or casadi.MX (3,)

    Returns:
    casadi.SX or casadi.MX: Rotated vector (3,)
    """
    norm_fac = cs.sqrt(cs.sum1(q ** 2))
    #q= q/norm_fac
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = cs.horzcat(
        cs.vertcat(1 - 2 * q2 ** 2  - 2 * q3 ** 2,  2 * q1 * q2       - 2 * q0 * q3     , 2 * q1 * q3     - 2 * q0 * q2),
        cs.vertcat(2 * q1 * q2      - 2 * q0 * q3,  1 - 2 * q1 ** 2   - 2 * q3 ** 2     , 2 * q2 * q3     + 2 * q0 * q1),
        cs.vertcat(2 * q1 * q3      + 2 * q0 * q2,  2 * q2 * q3       - 2 * q0 * q1     , 1 - 2 * q1 ** 2 - 2 * q2 ** 2)
    )


    return R@v

# Save dictionary to YAML file
def save_to_yaml(data, filename="model_params.yaml"):
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)

# Read dictionary from YAML file
def read_from_yaml(filename="./model_params.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


# if called as main, test functions
if __name__ == "__main__":
    q = cs.SX.sym("q", 4)
    r = cs.SX.sym("r", 4)
    fun_multiply = cs.Function("fun_multiply", [q, r], [quaternion_multiply_casadi(q, r)])

    q = [1, 0, 3, 4]
    r = [5, 6, 4, 8]
    print(f"multiply ego:{fun_multiply(q, r).full().flatten()}")  # expect [-60.  12.  30.  24.]
    print(f"multiply expected:{np.quaternion(*q) * np.quaternion(*r)}")  # expect [-60.  12.  30.  24.]

    q = cs.SX.sym("q", 4)
    v = cs.SX.sym("v", 3)
    fun_rotate = cs.Function("fun_rotate", [q, v], [quaternion_rotate_vector_casadi(q, v)])

    from scipy.spatial.transform import Rotation as R

    #r = R.from_euler('y', 45, degrees=True)
    #q = list(r.as_quat())
    from pyquaternion import Quaternion
    q = [1,2,3,4]
    v = [3., 4, 5]
    print(f"rotated:{fun_rotate(q, v).full().flatten()}")
    print(f"rotated expected:{quaternion.rotate_vectors(np.quaternion(*q), np.array(v))}")
    qbar = Quaternion(q)
    print(f"rotated expected:{qbar.rotate(v)}")


