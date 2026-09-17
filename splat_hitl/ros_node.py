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
import math
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
from .frames import matrix_to_rpy, quat_to_matrix
from .runtime import (PoseSample, PoseSource, Runtime, RuntimeConfig,
                      TransformedPoseSource)
from .contract import PolicyContract
from .sensor import SensorModel


# --- mocap solver flips ------------------------------------------------------
# A rotationally ambiguous rigid body makes Tracker switch between two equally
# good fits of the same marker constellation. Yaw jumps ~90 or ~180 deg in a
# single frame while the drone has not moved at all: the rotation is
# FICTITIOUS, and the right response is to disbelieve it and keep the heading
# we already had -- not to average it, not to track it.
#
# The driver refuses to feed such a quaternion to the onboard EKF
# (cf_core.flip_check) and lands if they persist. Nothing protected the POLICY,
# which is the more dangerous gap: a velocity policy rotates its world-frame
# command into the body frame with this yaw on every step, so a yaw that is
# 90 deg wrong does not degrade the flight, it aims it at the wall at full
# commanded speed.
#
# Same test as cf_core, deliberately, so the two cannot disagree about what a
# flip is: BOTH an absolute step over MIN_FLIP_DEG and an implied rate over
# MAX_YAW_RATE_DPS. Ordinary solver jitter is 1-2 deg, so the step test alone
# would also reject real rotation; the rate test alone fires on that jitter
# whenever two frames arrive close together, which is why the divisor is
# capped -- uncapped, one rejected frame makes the next look slow and lets the
# flip through.
MIN_FLIP_DEG = 45.0
MAX_YAW_RATE_DPS = 720.0
FLIP_GAP_CAP_S = 0.05


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def is_solver_flip(yaw_deg, last_yaw_deg, last_yaw_t, now):
    """(is_flip, step_deg, rate_dps). Pure, so it can be tested without ROS.

    Mirrors cf_core.flip_check. Kept as a free function rather than inlined in
    the callback for exactly that reason: the one piece of logic standing
    between a mirrored solution and a full-speed sideways command should be
    testable without a live ROS graph and a physical drone.
    """
    if last_yaw_deg is None or last_yaw_t is None:
        return (False, 0.0, 0.0)
    d = abs(_wrap180(yaw_deg - last_yaw_deg))
    gap = max(now - last_yaw_t, 1e-6)
    rate = d / min(gap, FLIP_GAP_CAP_S)
    return (d > MIN_FLIP_DEG and rate > MAX_YAW_RATE_DPS, d, rate)


def _yaw_deg_of(quat) -> float:
    """Yaw exactly as PoseSample.yaw_rad computes it.

    Derived through the same matrix_to_rpy(quat_to_matrix(...)) path rather
    than a hand-rolled atan2, because a filter that measures a different yaw
    than the policy consumes is worse than no filter: it would pass the frames
    it should catch and catch the frames it should pass.
    """
    return math.degrees(matrix_to_rpy(quat_to_matrix(*quat))[2])


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
        self._node = node
        self._last_quat = None          # last orientation we believed
        self._last_yaw_deg = None
        self._last_yaw_t = None
        self._quat_ok_t = None
        self.yaw_rejects = 0
        self._warned = False
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
        now = time.monotonic()
        quat = (q.x, q.y, q.z, q.w)
        yaw_deg = _yaw_deg_of(quat)

        flip, d, rate = is_solver_flip(yaw_deg, self._last_yaw_deg,
                                      self._last_yaw_t, now)
        if flip:
            # Hold the last orientation we believed. _last_yaw_deg is NOT
            # advanced: if the solver stays in the wrong solution, every
            # following frame is measured against the heading we trust and is
            # rejected too, so quat_age_s grows and the watchdog lands us.
            # Advancing it here would let a latched flip walk the heading
            # across in two steps that are each individually too small to catch.
            self.yaw_rejects += 1
            quat = self._last_quat
            if not self._warned:
                self._warned = True
                self._node.get_logger().error(
                    "yaw jumped %.0f deg (%.0f deg/s) -- rigid body is "
                    "rotationally ambiguous. Holding the last good heading; "
                    "the policy will not see the flip." % (d, rate))
        else:
            self._last_yaw_deg, self._last_yaw_t = yaw_deg, now
            self._last_quat, self._quat_ok_t = quat, now

        s = PoseSample(now,
                       np.array([o.x, o.y, o.z], dtype=float),
                       quat,
                       capture_t_s=h.sec + h.nanosec * 1e-9)
        with self._lock:
            self._latest = s

    def latest(self) -> Optional[PoseSample]:
        with self._lock:
            return self._latest

    def quat_age_s(self) -> float:
        """Seconds since the last ACCEPTED orientation.

        Deliberately not pose age. A solver latched onto the wrong solution
        keeps delivering fresh, smooth, low-latency POSITION while every
        orientation is rejected, and a position watchdog sees that as healthy.
        """
        t = self._quat_ok_t
        return 1e9 if t is None else (time.monotonic() - t)


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
    ap.add_argument("--sensor", default=None,
                    help="sensor model JSON. Omit when --contract is given: "
                         "the contract carries its own sensor and loading two "
                         "is how they drift apart.")
    ap.add_argument("--contract", default=None,
                    help="policy contract JSON (config/policy_contract.example"
                         ".json). Required to fly a policy that emits "
                         "accelerations or expects a stacked depth history.")
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

    contract = PolicyContract.load(a.contract) if a.contract else None
    if contract is not None:
        if a.sensor:
            raise SystemExit(
                "--sensor and --contract both given. The contract carries its "
                "own sensor model; loading a second one is exactly how the two "
                "drift apart. Drop --sensor.")
        sensor = contract.observation.sensor
        want = contract.control.rate_hz
        if a.rate_hz is not None and abs(a.rate_hz - want) > 1e-9:
            raise SystemExit(
                "--rate-hz %.3f disagrees with the contract's %.3f Hz.\n"
                "A policy that integrates its own action carries its timestep "
                "inside its behaviour: running it faster does not make it "
                "smoother, it makes it faster. Drop --rate-hz, or fly a "
                "contract that says %.3f." % (a.rate_hz, want, a.rate_hz))
        a.rate_hz = want
    elif a.sensor:
        sensor = SensorModel.load(a.sensor)
    else:
        raise SystemExit("need --sensor or --contract")
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
    vicon = ViconPoseSource(node, a.topic)
    source: PoseSource = vicon
    if transform is not None:
        source = TransformedPoseSource(source, transform)
    elif a.worker:
        raise SystemExit("--worker without --transform would render the wrong "
                         "part of the scene; supply the calibration")

    cfg = RuntimeConfig(hold_altitude_m=a.hold_altitude_m, yaw_sign=a.yaw_sign,
                        contract=contract)
    rt = Runtime(source, renderer, load_policy(a.policy), course, monitor, cfg)
    rec = RunRecorder(time.strftime("%Y%m%d-%H%M%S"), rt.policy.name,
                      sensor.fingerprint(), transform,
                      contract_fingerprint=None if contract is None
                      else contract.fingerprint())
    pub = node.create_publisher(Hover, "%s/cmd_hover" % a.prefix.rstrip("/"), 10)
    if monitor:
        for w in monitor.warnings:
            node.get_logger().warn(w)

    period = 1.0 / (a.rate_hz or sensor.rate_hz)
    node.get_logger().info("stepping at %.1f Hz%s"
                           % (1.0 / period, "  [DRY RUN]" if a.dry_run else ""))
    done = threading.Event()

    # Longer than the driver's own QUAT_DEAD_S would be pointless -- it lands
    # first. This exists so the HITL node stops COMMANDING at the same moment,
    # instead of streaming setpoints at a drone that has started an auto-land.
    QUAT_DEAD_S = 0.5

    def tick():
        if vicon.quat_age_s() > QUAT_DEAD_S:
            rt.stop("mocap orientation unusable for %.0f ms (%d yaw rejects)"
                    % (vicon.quat_age_s() * 1000.0, vicon.yaw_rejects))
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
        print("\n" + rt.summary() + "\n" + rec.summary()
              + "\n  yaw rejects: %d" % vicon.yaw_rejects + "\n")
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
