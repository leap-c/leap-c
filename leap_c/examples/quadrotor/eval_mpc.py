import imageio
from matplotlib import pyplot as plt
import random

from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMpc

if __name__ == "__main__":

    env = QuadrotorStop(render_mode="rgb_array", scale_disturbances=0.001)#0.001)
    mpc = QuadrotorMpc(N_horizon=8)
    solver = mpc.ocp_solver
    render_movie = True
    record_iterate = False

    obs, _ = env.reset(seed=random.randint(0, 1000))
    image = env.render()
    done = False

    rewards_sum = 0

    # create fig and axis with given size
    if render_movie:
        fig, ax = plt.subplots(figsize=(15, 15))
        img_display = ax.imshow(image)
        plt.axis("off")  # Hide axes
        frames = []
    while not done:
        # Set initial state
        solver.set(0, "lbx", obs)
        solver.set(0, "ubx", obs)
        status = solver.solve()
        #print(f"SQP Iterations: {solver.get_stats('sqp_iter')}")
        if env.t > 0.8 and record_iterate:
            solver.store_iterate("./examples/quadrotor/init_iterate.json")
            break

        action = solver.get(0, "u")
        obs, reward, done, _, _ = env.step(action)
        rewards_sum += reward
        print(f"reward:{reward}")
        if render_movie:
            env.render_mode = "rgb_array"
            image= env.render()
            img_display.set_data(image)
            frames.append(image)
            plt.pause(0.01)
    print(f"Total reward: {rewards_sum}")
    if render_movie:
        imageio.mimsave("drone_simulation.mp4", frames, fps=1/env.sim_params["dt"])
        plt.close(fig)

    env.render_mode = "human"
    env.render()
