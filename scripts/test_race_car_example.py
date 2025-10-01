#!/usr/bin/env python3
"""
Simple test script to check if the race car example works.
"""

import sys
import os

# Add the race car example path to sys.path
race_car_path = os.path.join(os.path.dirname(__file__), '../external/acados/examples/acados_python/race_cars')
if race_car_path not in sys.path:
    sys.path.insert(0, race_car_path)

try:
    from bicycle_model import bicycle_model
    from tracks.readDataFcn import getTrack
    from plotFcn import plotTrackProj
    
    print("✓ Successfully imported race car modules")
    
    # Test loading track data
    track_data = getTrack("LMS_Track.txt")
    Sref, Xref, Yref, Psiref, kapparef = track_data
    print(f"✓ Successfully loaded track data: {len(Sref)} points, length {Sref[-1]:.2f}m")
    
    # Test loading bicycle model
    model, constraint = bicycle_model("LMS_Track.txt")
    print(f"✓ Successfully loaded bicycle model: {model.name}")
    
    print("\n🎉 Race car example is working properly!")
    print("You can now run the SAC training script for Q matrix learning.")
    
except ImportError as e:
    print(f"❌ Failed to import race car modules: {e}")
    print("Make sure the acados race car example is available in external/acados/examples/acados_python/race_cars/")
except Exception as e:
    print(f"❌ Error testing race car example: {e}")
    
try:
    # Test our wrapper
    from leap_c.examples.race_cars.env import RaceCarEnv, RaceCarEnvConfig

    print("\n✓ Testing our race car environment wrapper...")
    env = RaceCarEnv(cfg=RaceCarEnvConfig())  # kappa_ref will be auto-loaded
    obs, info = env.reset()
    print(f"✓ Environment reset successful. State shape: {obs.shape}")
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space: {env.action_space}")

    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        s, n, alpha, v, D, delta = obs
        print(f"  Step {i+1}: s={s:.2f}, n={n:.3f}, v={v:.2f}, reward={reward:.3f}")

        if terminated or truncated:
            print(f"  Episode ended: {info}")
            break

    # Test visualization methods
    print("\n✓ Testing visualization methods...")
    try:
        env.plot_trajectory()
        print("✓ plot_trajectory() works")
    except Exception as e:
        print(f"⚠ plot_trajectory() error: {e}")

    try:
        env.plot_results()
        print("✓ plot_results() works")
    except Exception as e:
        print(f"⚠ plot_results() error: {e}")

    env.close()
    print("✓ Race car environment wrapper is working!")

except Exception as e:
    print(f"❌ Error with race car environment wrapper: {e}")
    import traceback
    traceback.print_exc()