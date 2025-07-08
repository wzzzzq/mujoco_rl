import numpy as np
import pytest

from pyroboplan.trajectory.trapezoidal_velocity import TrapezoidalVelocityTrajectory


def test_single_dof_trajectory():
    q = np.array([1.0, 2.0, 2.0, 2.4, 5.0])
    qd_max = 1.0
    qdd_max = 1.0

    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    # Check that the segments are as follows:
    # 2-segment, 0-segment (no motion), 2-segment, 3-segment
    assert len(traj.segment_times) == 4
    traj_0 = traj.single_dof_trajectories[0]
    assert len(traj_0.times) == 8
    assert not np.any(np.abs(traj_0.velocities) > qd_max)
    assert not np.any(np.abs(traj_0.accelerations) > qdd_max)


def test_multi_dof_trajectory_scalar_limits():
    q = np.array(
        [
            [1.0, 2.0, 2.0, 2.4, 5.0],
            [1.0, 0.0, -1.0, 0.5, 2.0],
            [0.0, 1.0, 0.0, -1.0, 0.0],
        ]
    )
    qd_max = 1.0
    qdd_max = 1.0

    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    assert len(traj.segment_times) == 5
    for sub_traj in traj.single_dof_trajectories:
        assert not np.any(np.abs(sub_traj.velocities) > qd_max)
        assert not np.any(np.abs(sub_traj.accelerations) > qdd_max)


def test_multi_dof_trajectory_vector_limits():
    q = np.array(
        [
            [1.0, 2.0, 2.0, 2.4, 5.0],
            [1.0, 0.0, -1.0, 0.5, 2.0],
            [0.0, 1.0, 0.0, -1.0, 0.0],
        ]
    )
    qd_max = np.array([1.5, 0.5, 0.7])
    qdd_max = np.array([1.0, 1.5, 0.9])

    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    assert len(traj.segment_times) == 5
    for dim, sub_traj in enumerate(traj.single_dof_trajectories):
        assert not np.any(np.abs(sub_traj.velocities) > qd_max[dim])
        assert not np.any(np.abs(sub_traj.accelerations) > qdd_max[dim])


def test_evaluate_bad_time_values():
    q = np.array([1.0, 2.0, 2.0, 2.4, 5.0])
    qd_max = 1.5
    qdd_max = 1.0
    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    with pytest.warns(UserWarning):
        assert traj.evaluate(-1.0) is None

    with pytest.warns(UserWarning):
        assert traj.evaluate(100.0) is None


def test_evaluate_single_dof():
    q = np.array([1.0, 2.0, 2.0, 2.4, 5.0])
    qd_max = 1.5
    qdd_max = 1.0
    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    q, qd, qdd = traj.evaluate(0.0)
    assert q == pytest.approx(1.0)
    assert qd == pytest.approx(0.0)
    assert qdd == pytest.approx(1.0)

    q, qd, qdd = traj.evaluate(1.0)
    assert q == pytest.approx(1.5)
    assert qd == pytest.approx(1.0)
    assert qdd == pytest.approx(1.0)


def test_evaluate_multi_dof():
    q = np.array(
        [
            [1.0, 2.0, 2.0, 2.4, 5.0],
            [1.0, 0.0, -1.0, 0.5, 2.0],
            [0.0, 1.0, 0.0, -1.0, 0.0],
        ]
    )
    qd_max = 1.0
    qdd_max = 1.0
    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    q, qd, qdd = traj.evaluate(0.0)

    assert len(q) == 3
    assert len(qd) == 3
    assert len(qd) == 3

    assert q[0] == pytest.approx(1.0)
    assert qd[0] == pytest.approx(0.0)
    assert qdd[0] == pytest.approx(1.0)

    assert q[1] == pytest.approx(1.0)
    assert qd[1] == pytest.approx(0.0)
    assert qdd[1] == pytest.approx(-1.0)

    assert q[2] == pytest.approx(0.0)
    assert qd[2] == pytest.approx(0.0)
    assert qdd[2] == pytest.approx(1.0)


def test_generate_single_dof():
    q = np.array([1.0, 2.0, 2.0, 2.4, 5.0])
    qd_max = 1.5
    qdd_max = 1.0
    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    t, q, qd, qdd = traj.generate(dt=0.01)
    num_pts = len(t)
    assert q.shape == (1, num_pts)
    assert qd.shape == (1, num_pts)
    assert qdd.shape == (1, num_pts)


def test_generate_multi_dof():
    q = np.array(
        [
            [1.0, 2.0, 2.0, 2.4, 5.0],
            [1.0, 0.0, -1.0, 0.5, 2.0],
            [0.0, 1.0, 0.0, -1.0, 0.0],
        ]
    )
    qd_max = np.array([1.5, 0.5, 0.7])
    qdd_max = np.array([1.0, 1.5, 0.9])
    traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)

    t, q, qd, qdd = traj.generate(dt=0.01)
    num_pts = len(t)
    assert q.shape == (3, num_pts)
    assert qd.shape == (3, num_pts)
    assert qdd.shape == (3, num_pts)


if __name__ == "__main__":
    pass
