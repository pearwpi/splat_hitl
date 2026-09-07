"""splat_hitl -- hardware-in-the-loop bridge between a Vicon-flown Crazyflie
and a metric Gaussian-splat scene.

Nothing in this package imports torch, gsplat or nerfstudio. The renderer is
reached through a client interface so the entire runtime is testable, and
teachable, on a laptop with no GPU.
"""
__version__ = "0.1.0"
