# trailer_ctrl -- controller for the TurtleBot3 with off-hooked trailers (no ROS)

Pure Python (numpy; scipy only for `sim.py` and the tests). It is the Python twin of the
MATLAB reference `1 trailer/trailer1.m` and `2 trailers/trailer2.m`. The ROS 2 node
(next step) only calls this package, so the controller can be tested without ROS, Gazebo or the robot.

## Files

| file | what |
|---|---|
| `geometry.py` | `Geometry`, coefficients alpha/kappa/theta/beta, state map `x -> z` (1 and 2 trailers) |
| `controller.py` | `Params`, `TrailerController` (amplitudes, `u(tau, z)`, worst-case bound, `gamma_max`), `SampledController` (fixed interval grid, freezing of `z`, saturation) |
| `sim.py` | offline closed loop on the approximation and on the exact kinematics (for tests/plots) |
| `configs/1tr.json`, `configs/2tr.json` | the design of thesis Chapters 6-7 (gamma, epsilon, k, Lyapunov function, limits) |
| `experiment.py` | everything around the controller that does not need ROS: pose input, goal frame, angle unwrapping, watchdog, safety stops, logging/plots |
| `ros_node.py` | ROS 2 Humble node (thin shell around `experiment.py`; Gazebo Classic or NatNet) |
| `tests/test_vs_matlab.py` | compares everything with the reference numbers of the MATLAB selftest |
| `tests/test_experiment.py` | closed loop with a simulated plant: goal frame, wrapping, axle offset, stops, export (needs numpy, scipy, matplotlib) |

## Test

From the folder that contains `trailer_ctrl/`:

    python3 -m trailer_ctrl.tests.test_vs_matlab
    (or simply: python3 trailer_ctrl/tests/test_vs_matlab.py)

It must end with `PORT TEST PASSED`. Reference numbers come from `selftest` in `trailer1.m` /
`trailer2.m` (18 Oct 2026, thesis design). If you change the design in `configs/*.json` on purpose,
the test fails on purpose: update the numbers from a fresh MATLAB selftest.

## Use (what the ROS node will do)

    import json
    from trailer_ctrl import Params, TrailerController, SampledController

    ctrl = TrailerController(Params.from_dict(json.load(open("configs/1tr.json"))))
    sc = SampledController(ctrl)
    sc.start(t_now)                       # t = 0 of the experiment
    ...
    cmd = sc.command(t_now, x_measured)   # x = [x1, x2, phi, theta1(, theta2)], UNWRAPPED angles
    publish(cmd.u1, cmd.u2)               # already saturated; cmd.clipped tells if that was active

## Rules that matter on the robot

* **Interval starts on a fixed grid** `t0 + j*epsilon`, phase counted from the start of the interval.
  Then all cosines equal 1 at the start (the peak of `|u1|`, which the bound of Chapter 6 refers to) and
  every oscillation completes whole periods. Do not restart "when at least epsilon has elapsed".
* **Angles must be continuous (unwrapped).** The privileged coordinates are linear in the angles and do not
  know that an angle is defined modulo 2 pi; wrapping them would make `z` jump.
* **Gain is fixed.** The design gain is tuned along the straight ray to the goal on the nilpotent
  approximation; on the exact kinematics `|u1|` can exceed the limit (thesis Chapter 7, two trailers:
  up to 1.12x after 30 minutes). Set `clip_fraction` below 1 for a margin and watch `cmd.clipped`.
* `cmd.skipped` (`SampledController.skipped`) counts intervals missed because the loop was late.

## Relation to S. Eisner's `controller_main.py`

His node (ROS 2, NatNet, `/cmd_vel`, unwrap of yaw, CSV/plot export) is the model for the node structure;
it is for the car alone (r = 2) and tracks one rigid body. Not taken over: his restart-when-elapsed timing,
the missing saturation. Used under the permission he gave in September 2026; do not publish it.

## ROS 2 node (Humble)

    source /opt/ros/humble/setup.bash
    cd <folder that contains trailer_ctrl/>
    # simulation (Gazebo Classic with road_train.urdf.xacro running)
    python3 -m trailer_ctrl.ros_node --config trailer_ctrl/configs/2tr.json --source sim --ros-args -p use_sim_time:=true
    # real robot, first without driving
    python3 -m trailer_ctrl.ros_node --config trailer_ctrl/configs/2tr.json --source real --dry_run \
        --natnet_dir "<folder with Eisner's NatNetClient.py>" --rb_car 40 --rb_t1 41 --rb_t2 42

`python3 -m trailer_ctrl.ros_node --help` lists all options (goal, rate, gain, clip fraction, tolerances, topics, IPs).
Results land in `results/<timestamp>/` (log.csv, params.json, summary.txt, PNGs). Ctrl+C stops the robot and still writes them.

Points to check before the first run (not testable without ROS, so they are YOUR checks):

* **Frames in simulation.** The trailer poses come from `libgazebo_ros_p3d` in the WORLD frame; `/odom` of the
  TurtleBot starts at (0, 0, 0) at its spawn pose. They agree only if the car is spawned at the world origin with
  yaw 0 (otherwise add a p3d plugin for the car and pass its topic as `--car_topic`).
* **Rigid body IDs** 41 and 42 for the trailers are placeholders; 40 is Eisner's default for the car.
  Motive must give Z-up (yaw about z), as in his code.
* **Tracked point of the car.** The controller needs the midpoint of the axle. If the rigid body origin is elsewhere,
  give `--car_axle_offset` (positive = axle ahead of the tracked point, in metres).
* **Zero-order hold.** The command is held between ticks. Measured on the exact model after 30 s: error of ||z|| is
  2.7 % of ||z0|| at 20 Hz, 1 % at 50 Hz, 0.5 % at 100 Hz (default), 0.1 % at 400 Hz.
* **Safety stops** (all publish zero velocity and write the log): pose older than `--stale` (0.2 s; resumes when fresh),
  hitch angle above `--hitch_limit` (1.35 rad), `--t_max` (1800 s), goal reached.

## Known problem: hitch angles of the design path

The path of the thesis design (start x0 = (-0.2, -0.2, 45 deg, 45 deg[, 45 deg])) folds the hitches far beyond the
URDF limit of +-1.40 rad (checked 22 Oct 2026, exact model, independent of the node):

| robot | max hitch angle | first time above 1.40 rad | share of the first 120 s above 1.40 rad |
|---|---|---|---|
| 1 trailer | 2.51 rad (144 deg) | 9.4 s | 54 % |
| 2 trailers | 2.60 rad between trailers 1 and 2 (car-trailer 1: 0.87) | 32.1 s | 44 % |

With the start scaled by 0.5 the maxima are 1.65 / 1.84 rad, with 0.25 they are 0.66 / 1.03 rad. In Gazebo and on the real robot
the hitch stops (and collisions) will therefore change the motion, and the node aborts at 1.35 rad. This has to be decided
(smaller start, other gains, limit as constraint) before comparing Gazebo or robot runs with MATLAB.
