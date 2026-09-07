"""ROS 2 node that hosts the runtime. Deliberately thin.

Everything that can be wrong is decided in `runtime.py` and tested without ROS.
This file only moves messages: subscribe to Vicon, call `Runtime.step()`,
publish what it returns. If you find yourself adding a decision here, it
probably belongs in the runtime where it can be tested.

    ros2 run ... or:
    python3 -m splat_hitl.ros_node \\
        --topic /vicon/crazyflie2/crazyflie2 \\
        --prefix /cf1 \\
        --sensor config/sensor_model.example.json \\
        --transform scene_a/vicon_to_splat.json \\
        --gates scene_a/gates.json \\
        --esdf scene_a/arena_cleaned_pc_esdf.npy \\
        --policy my_module:MyPolicy \\
        --fake-room 4 3 2.5            # or --worker "<python> splat_rendering.py ..."

NOT YET RUN AGAINST HARDWARE. The logic underneath has 240 tests; this shell
has none, because none of it is testable without a live ROS graph. Bring it up
the first time with `--dry-run`, which does everything except publish.

SAFETY, EXPLICITLY
------------------
This node commands the drone. `cf_core`'s guards still sit downstream in
`crazyflie_server` and are what actually protect the room -- this adds the
simulation's verdict on top. If the node dies, the driver's 300 ms command
timeout stops the drone; that is the backstop, not the plan. The plan is the
runtime's own LANDING state, which happens first and on our terms.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import threading
import time
from typing import Optional

import numpy as np

from .collision import CollisionMonitor, ESDF
from .frames import SplatTransform
from .gates import GateCourse
from .policy import GateSeekPolicy, HoverPolicy, Policy
from .recorder import RunRecorder
from .renderer import FakeRenderer, SplatWorkerClient
from .runtime import (FINISHED, PoseSample, PoseSource, Runtime, RuntimeConfig,
                      TransformedPoseSource)
from .sensor import SensorModel


class ViconPoseSource(PoseSource):
    """Latest pose off a ROS topic. Thread-safe, non-blocking.

    `latest()` never waits: a subscriber that blocks the control loop is worse
    than a stale pose, because the runtime can see and handle staleness and
    cannot see a stall. The stamp is the message's own header time, so
    `pose_age_s` measures capture-to-use once the bridge's capture-time
    stamping is in place -- not the age of our last callback.
    """

    def __init__(self, node, topic: str):
        from rosidl_runtime_py.utilities import get_message
        self._lock = threading.Lock()
        self._latest: Optional[PoseSample] = None
        mtype = None
        for _ in range(50):
            for name, types in node.get_topic_names_and_types():
                if name == topic and types:
                    mtype = types[0]
                    break
            if mtype:
                break
            time.sleep(0.1)
        if not mtype:
            raise RuntimeError("topic %s not found -- is the bridge running?" % topic)
        node.get_logger().info("pose from %s [%s]" % (topic, mtype))
        node.create_subscription(get_message(mtype), topic, self._cb, 20)

    def _cb(self, msg):
        p = getattr(msg, "pose", None)
        if p is not None and hasattr(p, "position"):
            o, q = p.position, p.orientation
        else:
            t = getattr(msg, "transform", None)
            if t is None:
                return
            o, q = t.translation, t.rotation
        h = msg.header.stamp
        # t_s is a LOCAL MONOTONIC receipt time, because that is what the
        # staleness watchdog must compare against and it cannot be fooled by
        # clock skew with the Vicon PC. The header stamp -- which the bridge
        # sets to CAPTURE time -- rides along separately, for latency.
        s = PoseSample(time.monotonic(),
                       np.array([o.x, o.y, o.z], dtype=float),
                       (q.x, q.y, q.z, q.w),
                       capture_t_s=h.sec + h.nanosec * 1e-9)
        with self._lock:
            self._latest = s

    def latest(self) -> Optional[PoseSample]:
        with self._lock:
            return self._latest


def load_policy(spec: str) -> Policy:
    """`module:ClassName`, or one of the built-in reference names."""
    builtin = {"hover": HoverPolicy, "gate_seek": GateSeekPolicy}
    if spec in builtin:
        return builtin[spec]()
    if ":" not in spec:
        raise ValueError("policy must be 'module:ClassName' or one of %s"
                         % sorted(builtin))
    mod, cls = spec.split(":", 1)
    obj = getattr(importlib.import_module(mod), cls)()
    if not isinstance(obj, Policy):
        raise TypeError("%s is not a Policy subclass" % spec)
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--prefix", default="/cf1", help="Crazyflie command namespace")
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--transform", default=None,
                    help="vicon_to_splat.json; omit only for a fake-room dry run")
    ap.add_argument("--gates", default=None)
    ap.add_argument("--esdf", default=None)
    ap.add_argument("--clearance-m", type=float, default=0.10)
    ap.add_argument("--policy", default="hover")
    ap.add_argument("--worker", default=None,
                    help="shell command that starts splat_rendering.py")
    ap.add_argument("--fake-room", nargs=3, type=float, default=None,
                    metavar=("W", "D", "H"))
    ap.add_argument("--hold-altitude-m", type=float, default=0.60)
    ap.add_argument("--rate-hz", type=float, default=None)
    ap.add_argument("--yaw-sign", type=int, choices=[1, -1], default=1)
    ap.add_argument("--log", default=None, help="write the run log here")
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything, publish nothing")
    a = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from crazyflie_interfaces.msg import Hover

    sensor = SensorModel.load(a.sensor)
    transform = SplatTransform.load(a.transform) if a.transform else None
    course = GateCourse.load(a.gates) if a.gates else None
    monitor = None
    if a.esdf:
        monitor = CollisionMonitor(ESDF.load_npy(a.esdf), a.clearance_m)

    if a.worker:
        import shlex
        renderer = SplatWorkerClient(sensor, shlex.split(a.worker))
    elif a.fake_room:
        renderer = FakeRenderer(sensor, tuple(a.fake_room))
    else:
        raise SystemExit("need --worker or --fake-room")

    rclpy.init()
    node = Node("splat_hitl")
    source: PoseSource = ViconPoseSource(node, a.topic)
    if transform is not None:
        source = TransformedPoseSource(source, transform)
    elif a.worker:
        raise SystemExit("--worker without --transform would render the wrong "
                         "part of the scene; supply the calibration")

    cfg = RuntimeConfig(hold_altitude_m=a.hold_altitude_m, yaw_sign=a.yaw_sign)
    rt = Runtime(source, renderer, load_policy(a.policy), course, monitor, cfg)
    rec = RunRecorder(time.strftime("%Y%m%d-%H%M%S"), rt.policy.name,
                      sensor.fingerprint(), transform)
    pub = node.create_publisher(Hover, "%s/cmd_hover" % a.prefix.rstrip("/"), 10)
    if monitor:
        for w in monitor.warnings:
            node.get_logger().warn(w)

    period = 1.0 / (a.rate_hz or sensor.rate_hz)
    node.get_logger().info("stepping at %.1f Hz%s"
                           % (1.0 / period, "  [DRY RUN]" if a.dry_run else ""))
    done = threading.Event()

    def tick():
        r = rt.step()
        s = source.latest()
        if s is not None:
            if course is not None:
                rec.note_gates(course.passed)
            rec.add(s.t_s, s, r)
        for e in r.events:
            node.get_logger().info(e)
        if r.command is not None and not a.dry_run:
            m = Hover()
            m.vx, m.vy = r.command.vx, r.command.vy
            m.yaw_rate = r.command.yaw_rate
            m.z_distance = r.command.z_distance
            pub.publish(m)
        if r.finished:
            rec.finish(r.reason)
            done.set()

    node.create_timer(period, tick)
    try:
        while rclpy.ok() and not done.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        rt.stop("operator_stop")
        rec.finish("operator_stop")
    finally:
        renderer.close()
        print("\n" + rt.summary() + "\n" + rec.summary() + "\n")
        if a.log:
            rec.save_json(a.log)
            rec.save_csv(str(a.log).rsplit(".", 1)[0] + ".csv")
            print("  wrote %s" % a.log)
        # Ctrl-C reaches rclpy's own signal handler first, which shuts the
        # context down before this block runs; calling shutdown() again raises.
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
