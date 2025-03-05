import imageio
from matplotlib import pyplot as plt

from acados_template import AcadosOcpSolver

from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMPC

if __name__ == "__main__":

    env = QuadrotorStop(render_mode="rgb_array")
    mpc = QuadrotorMPC()
    solver = mpc.ocp_solver

    obs, _ = env.reset(seed=5)
    image = env.render()
    done = False

    # create fig and axis with given size
    fig, ax = plt.subplots(figsize=(15, 15))
    img_display = ax.imshow(image)
    plt.axis("off")  # Hide axes
    frames = []
    while not done:
        # Set initial state
        solver.set(0, "lbx", obs)
        solver.set(0, "ubx", obs)
        status = solver.solve()
        action = solver.get(0, "u")
        obs, reward, done, _, _ = env.step(action)
        print(f"reward:{reward}")
        env.render_mode = "rgb_array"
        image= env.render()
        img_display.set_data(image)
        frames.append(image)
        plt.pause(0.01)
    imageio.mimsave("drone_simulation.mp4", frames, fps=1/env.sim_params["dt"])
    plt.close(fig)

    env.render_mode = "human"
    env.render()
