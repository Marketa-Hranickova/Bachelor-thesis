#!/usr/bin/env python3
"""ROS 2 (Humble) node: closed loop for the TurtleBot3 with off-hooked trailers.

All the control logic lives in controller.py / experiment.py (tested without ROS);
this file only connects them to ROS topics and, for the real robot, to NatNet.

Run from the folder that contains trailer_ctrl/ (after `source /opt/ros/humble/setup.bash`):

  # Gazebo Classic (simulation), two trailers, goal at the origin
  python3 -m trailer_ctrl.ros_node --config trailer_ctrl/configs/2tr.json --source sim \
      --ros-args -p use_sim_time:=true

  # real robot: OptiTrack via NatNet (as in S. Eisner's controller_main.py)
  python3 -m trailer_ctrl.ros_node --config trailer_ctrl/configs/2tr.json --source real \
      --natnet_dir "/path/to/Eisner/src" --client_ip 192.168.1.147 --server_ip 192.168.1.171 \
      --rb_car 40 --rb_t1 41 --rb_t2 42

First run on the real robot with --dry_run: the node computes everything and logs it but
publishes no velocity, so the poses and the frame/offset settings can be checked by hand.

Poses
  sim : car   nav_msgs/Odometry on --car_topic (default /odom; must be in the WORLD frame, see README),
        trailers nav_msgs/Odometry on --t1_topic / --t2_topic (libgazebo_ros_p3d, world frame)
  real: NatNet rigid bodies --rb_car / --rb_t1 / --rb_t2 (positions in m, quaternions, yaw about z)
Output: geometry_msgs/Twist on --cmd_topic (default /cmd_vel); linear.x = u1, angular.z = u2.
Results (log.csv, params.json, summary.txt, PNGs) go to --out_dir/<timestamp>/.
"""
import argparse
import json
import math
import os
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from .controller import Params
from .experiment import Experiment, yaw_from_quat

HERE = os.path.dirname(os.path.abspath(__file__))


class TrailerNode(Node):
    def __init__(self, args, params: Params):
        super().__init__("trailer_controller")
        self.args = args
        self.p = params
        self.exp = Experiment(params, goal=tuple(args.goal), car_axle_offset=args.car_axle_offset,
                              stale_s=args.stale, hitch_limit=args.hitch_limit,
                              pos_tol=args.pos_tol, ang_tol=args.ang_tol, t_max=args.t_max)
        self.pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.done = False
        self.exported = None
        self._ready_since = None
        self._last_print = -1e9
        self.client = None

        if args.source == "sim":
            self.create_subscription(Odometry, args.car_topic, lambda m: self._on_odom("car", m), 10)
            self.create_subscription(Odometry, args.t1_topic, lambda m: self._on_odom("t1", m), 10)
            if params.n_trailers == 2:
                self.create_subscription(Odometry, args.t2_topic, lambda m: self._on_odom("t2", m), 10)
        else:
            self._start_natnet()

        self.timer = self.create_timer(1.0 / args.rate, self._tick)
        self.get_logger().info(
            "trailers=%d gamma=%.4g epsilon=%.3g s k=%s clip=%.2f rate=%.0f Hz source=%s%s"
            % (params.n_trailers, params.gamma, params.epsilon, list(params.k), params.clip_fraction,
               args.rate, args.source, "  DRY RUN (no velocity is published)" if args.dry_run else ""))

    # ------------------------------------------------------------------ time
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ input
    def _on_odom(self, name, msg: Odometry):
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        self.exp.set_pose(name, pos.x, pos.y, yaw_from_quat(q.x, q.y, q.z, q.w), self._now())

    def _start_natnet(self):
        a = self.args
        if a.natnet_dir:
            sys.path.insert(0, a.natnet_dir)
        from NatNetClient import NatNetClient   # S. Eisner's copy; see README
        ids = {a.rb_car: "car", a.rb_t1: "t1"}
        if self.p.n_trailers == 2:
            ids[a.rb_t2] = "t2"
        if len(ids) != (3 if self.p.n_trailers == 2 else 2):
            raise ValueError("rigid body IDs must be different: %s" % ids)
        self._ids = ids

        def on_rigid_body(rb_id, position, rotation):
            name = ids.get(rb_id)
            if name is not None:
                qx, qy, qz, qw = rotation
                self.exp.set_pose(name, position[0], position[1],
                                  yaw_from_quat(qx, qy, qz, qw), self._now())

        c = NatNetClient()
        c.set_client_address(a.client_ip)
        c.set_server_address(a.server_ip)
        c.set_use_multicast(a.use_multicast)
        c.rigid_body_listener = on_rigid_body
        ok = False
        for attempt in range(3):
            try:
                if c.run('d'):
                    ok = True
                    break
            except Exception as e:  # noqa: BLE001
                self.get_logger().warning("NatNet attempt %d failed: %r" % (attempt + 1, e))
        if not ok:
            raise RuntimeError("NatNet connection failed")
        self.client = c
        self.get_logger().info("NatNet connected, rigid bodies %s" % ids)

    # ------------------------------------------------------------------ output
    def _publish(self, u1, u2):
        if self.args.dry_run:
            return
        m = Twist()
        m.linear.x = float(u1)
        m.angular.z = float(u2)
        self.pub.publish(m)

    def _tick(self):
        if self.done:
            return
        now = self._now()
        if not self.exp.ready():
            if now - self._last_print > 2.0:
                self._last_print = now
                self.get_logger().info("waiting for poses of all bodies %s ..." % self.exp.names)
            return
        if self._ready_since is None:
            self._ready_since = now
            self.get_logger().info("all poses received; starting in %.1f s" % self.args.arm_delay)
        if now - self._ready_since < self.args.arm_delay:
            return
        st = self.exp.step(now)
        if st.status in ("goal", "abort"):
            self.get_logger().info("%s: %s" % (st.status.upper(), st.reason))
            self._finish()
            return
        if st.status == "stale":
            self._publish(0.0, 0.0)
            if now - self._last_print > 1.0:
                self._last_print = now
                self.get_logger().warning("stale pose: robot stopped (%s)" % st.reason)
            return
        if st.status != "run":
            return
        self._publish(st.u1, st.u2)
        if now - self._last_print > 2.0:
            self._last_print = now
            r = self.exp.rows[-1]
            nz = r[self.exp.header().index("norm_z")]
            self.get_logger().info("t=%6.1f s  ||z||=%.4f  u1=%+.3f m/s  u2=%+.3f rad/s"
                                   % (r[0], nz, st.u1, st.u2))

    # ------------------------------------------------------------------ end
    def _finish(self):
        if self.done:
            return
        self.done = True
        for _ in range(5):                     # make sure the robot gets a zero command
            self._publish(0.0, 0.0)
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception:                  # noqa: BLE001
                pass
        meta = dict(source=self.args.source, dry_run=self.args.dry_run, config=self.args.config,
                    rate=self.args.rate, argv=sys.argv[1:])
        self.exported = self.exp.export(self.args.out_dir, meta)
        self.get_logger().info("results saved in %s" % self.exported)
        self.get_logger().info("\n" + self.exp.summary())


def parse_args(argv):
    d = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d.add_argument("--config", default=os.path.join(HERE, "configs", "2tr.json"))
    d.add_argument("--source", choices=["sim", "real"], default="sim")
    d.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "PSI"),
                   help="goal pose of the car axle midpoint [m, m, rad], trailers aligned behind")
    d.add_argument("--rate", type=float, default=100.0, help="control loop rate [Hz]; the zero-order hold "
                   "error shrinks like 1/rate (about 0.5 percent of ||z0|| at 100 Hz)")
    d.add_argument("--gamma", type=float, default=None, help="override gain of the config")
    d.add_argument("--clip_fraction", type=float, default=None, help="saturate at this fraction of u1/u2 limits")
    d.add_argument("--car_axle_offset", type=float, default=0.0,
                   help="tracked point -> axle midpoint along the heading [m], positive = axle ahead")
    d.add_argument("--stale", type=float, default=0.2, help="stop if a pose is older than this [s]")
    d.add_argument("--hitch_limit", type=float, default=1.35, help="abort above this hitch angle [rad]")
    d.add_argument("--pos_tol", type=float, default=0.01)
    d.add_argument("--ang_tol", type=float, default=0.05)
    d.add_argument("--t_max", type=float, default=1800.0, help="abort after this time [s]")
    d.add_argument("--arm_delay", type=float, default=3.0, help="wait this long after the first complete pose [s]")
    d.add_argument("--dry_run", action="store_true", help="compute and log, publish nothing")
    d.add_argument("--out_dir", default=os.path.join(os.getcwd(), "results"))
    d.add_argument("--cmd_topic", default="/cmd_vel")
    d.add_argument("--car_topic", default="/odom")
    d.add_argument("--t1_topic", default="/trailer1/odom")
    d.add_argument("--t2_topic", default="/trailer2/odom")
    d.add_argument("--natnet_dir", default="", help="folder with NatNetClient.py (Eisner's src)")
    d.add_argument("--client_ip", default="192.168.1.147")
    d.add_argument("--server_ip", default="192.168.1.171")
    d.add_argument("--use_multicast", action="store_true")
    d.add_argument("--rb_car", type=int, default=40)
    d.add_argument("--rb_t1", type=int, default=41)
    d.add_argument("--rb_t2", type=int, default=42)
    return d.parse_args(argv)


def load_params(args) -> Params:
    with open(args.config) as f:
        cfg = json.load(f)
    if args.gamma is not None:
        cfg["gamma"] = args.gamma
    if args.clip_fraction is not None:
        cfg["clip_fraction"] = args.clip_fraction
    return Params.from_dict(cfg)


def main():
    args = parse_args(remove_ros_args(sys.argv)[1:])
    params = load_params(args)
    rclpy.init()
    node = None
    try:
        node = TrailerNode(args, params)
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._finish()                     # zero velocity + export, also after Ctrl+C
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
