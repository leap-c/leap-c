from acados_template import AcadosOcpSolver

from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMPC

if __name__ == "__main__":

    env = QuadrotorStop()
    mpc = QuadrotorMPC()
    solver = mpc.ocp_solver

    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        # Set initial state
        solver.set(0, "lbx", obs)
        solver.set(0, "ubx", obs)
        status = solver.solve()
        action = solver.get(0, "u")
        obs, reward, done, _, _ = env.step(action)
        print(f"reward:{reward}")
    env.render()
