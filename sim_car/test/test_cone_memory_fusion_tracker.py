from sim_car.cones.tracking.fusion import (
    choose_position_source,
    class_from_probs,
    resolve_boundary_color_by_lateral_position,
    update_class_probs,
)
from sim_car.cones.tracking.pose import convert_odom_child_pose_to_base_frame
from sim_car.cones.tracking.tracker import (
    TRACK_STATE_STALE,
    GlobalConeMemory,
    LocalConeTracker,
    TrackUpdate,
)


def _u(
    assoc_x=0.0,
    assoc_y=0.0,
    update_x=0.0,
    update_y=0.0,
    source='lidar',
    has_lidar=True,
    has_camera=False,
    camera_label=None,
    camera_confidence=0.0,
):
    return TrackUpdate(
        assoc_x=assoc_x,
        assoc_y=assoc_y,
        update_x=update_x,
        update_y=update_y,
        update_z=0.0,
        update_source=source,
        has_lidar=has_lidar,
        has_camera=has_camera,
        camera_label=camera_label,
        camera_confidence=camera_confidence,
        range_m=1.0,
    )


def test_choose_position_source_camera_range_split_and_fallbacks():
    # camera_range_m=5 => lidar in [0,15], camera in (15,20]
    assert choose_position_source(
        range_m=10.0,
        camera_range_m=5.0,
        has_lidar_position=True,
        has_camera_position=True,
        prefer_lidar_if_camera_missing_far=True,
        allow_camera_fallback_near=False,
    ) == 'lidar'

    assert choose_position_source(
        range_m=16.0,
        camera_range_m=5.0,
        has_lidar_position=True,
        has_camera_position=True,
        prefer_lidar_if_camera_missing_far=True,
        allow_camera_fallback_near=False,
    ) == 'camera'

    assert choose_position_source(
        range_m=18.0,
        camera_range_m=5.0,
        has_lidar_position=True,
        has_camera_position=False,
        prefer_lidar_if_camera_missing_far=True,
        allow_camera_fallback_near=False,
    ) == 'lidar'

    assert choose_position_source(
        range_m=5.0,
        camera_range_m=5.0,
        has_lidar_position=False,
        has_camera_position=True,
        prefer_lidar_if_camera_missing_far=True,
        allow_camera_fallback_near=True,
    ) == 'camera'


def test_class_probability_update_from_camera_label():
    probs = [1.0, 0.0, 0.0, 0.0]
    probs = update_class_probs(probs, label='blue', confidence=0.8)
    label, conf = class_from_probs(probs)
    assert label == 'blue'
    assert conf > 0.5


def test_resolve_boundary_color_by_lateral_position_maps_orange_to_track_side():
    assert resolve_boundary_color_by_lateral_position('orange', 1.0) == 'blue'
    assert resolve_boundary_color_by_lateral_position('orange', -1.0) == 'yellow'
    assert resolve_boundary_color_by_lateral_position('unknown', 1.0) == 'blue'
    assert resolve_boundary_color_by_lateral_position('unknown', -1.0) == 'yellow'


def test_local_tracker_association_and_confirmation():
    tracker = LocalConeTracker()

    stats1 = tracker.update(
        updates=[_u(assoc_x=1.0, assoc_y=2.0, update_x=1.0, update_y=2.0)],
        now_sec=1.0,
        gate_radius_m=0.5,
        spawn_radius_m=0.5,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=2,
    )
    assert stats1.new_tracks == 1
    assert len(tracker.confirmed_tracks(2)) == 0

    stats2 = tracker.update(
        updates=[_u(assoc_x=1.1, assoc_y=2.1, update_x=1.1, update_y=2.1)],
        now_sec=1.1,
        gate_radius_m=0.5,
        spawn_radius_m=0.5,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=2,
    )
    assert stats2.matched_updates == 1
    assert len(tracker.confirmed_tracks(2)) == 1


def test_local_tracker_pruning_by_ttl_and_range_and_behind():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[
            _u(assoc_x=0.0, assoc_y=0.0, update_x=0.0, update_y=0.0),
            _u(assoc_x=30.0, assoc_y=0.0, update_x=30.0, update_y=0.0),
            _u(assoc_x=-10.0, assoc_y=0.0, update_x=-10.0, update_y=0.0),
        ],
        now_sec=1.0,
        gate_radius_m=0.01,
        spawn_radius_m=0.01,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )

    # one track old, one too far, one behind
    tracker.tracks[0].last_seen_sec = -10.0
    positions = [
        (0.0, 0.0),
        (30.0, 0.0),
        (-10.0, 0.0),
    ]
    pruned = tracker.prune(
        now_sec=5.0,
        ttl_sec=2.0,
        max_range_m=25.0,
        behind_drop_m=8.0,
        unknown_drop_frames=15,
        track_positions_in_base=positions,
    )
    assert pruned == 3
    assert len(tracker.tracks) == 0


def test_local_tracker_pruning_unknown_filter():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[_u(assoc_x=0.0, assoc_y=0.0, update_x=0.0, update_y=0.0, source='lidar', has_lidar=True, has_camera=False)],
        now_sec=1.0,
        gate_radius_m=0.5,
        spawn_radius_m=0.5,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )
    # Keep matching this track with lidar-only updates => unknown streak grows.
    for i in range(2, 18):
        tracker.update(
            updates=[_u(assoc_x=0.0, assoc_y=0.0, update_x=0.0, update_y=0.0, source='lidar', has_lidar=True, has_camera=False)],
            now_sec=float(i),
            gate_radius_m=0.5,
            spawn_radius_m=0.5,
            alpha_lidar=0.4,
            alpha_camera=0.2,
            min_seen_count=1,
        )
    assert len(tracker.tracks) == 1
    pruned = tracker.prune(
        now_sec=18.0,
        ttl_sec=100.0,
        max_range_m=100.0,
        behind_drop_m=100.0,
        unknown_drop_frames=15,
        track_positions_in_base=[(0.0, 0.0)],
    )
    assert pruned == 1
    assert len(tracker.tracks) == 0


def test_global_cone_memory_merge_and_centerline():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[
            _u(assoc_x=0.0, assoc_y=2.0, update_x=0.0, update_y=2.0, source='camera', has_lidar=False, has_camera=True, camera_label='blue', camera_confidence=0.9),
            _u(assoc_x=0.0, assoc_y=-2.0, update_x=0.0, update_y=-2.0, source='camera', has_lidar=False, has_camera=True, camera_label='yellow', camera_confidence=0.9),
        ],
        now_sec=1.0,
        gate_radius_m=0.2,
        spawn_radius_m=0.2,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )

    memory = GlobalConeMemory()
    memory.update_from_tracks(
        tracks=tracker.confirmed_tracks(1),
        now_sec=1.0,
        merge_radius_m=0.6,
        max_cones=2000,
    )
    # second update near same spots should merge, not duplicate
    tracker.update(
        updates=[
            _u(assoc_x=0.1, assoc_y=2.1, update_x=0.1, update_y=2.1, source='camera', has_lidar=False, has_camera=True, camera_label='blue', camera_confidence=0.8),
            _u(assoc_x=0.1, assoc_y=-2.1, update_x=0.1, update_y=-2.1, source='camera', has_lidar=False, has_camera=True, camera_label='yellow', camera_confidence=0.8),
        ],
        now_sec=2.0,
        gate_radius_m=0.5,
        spawn_radius_m=0.5,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )
    memory.update_from_tracks(
        tracks=tracker.confirmed_tracks(1),
        now_sec=2.0,
        merge_radius_m=0.6,
        max_cones=2000,
    )

    assert len(memory.cones) == 2

    left, right, center = memory.infer_boundaries_and_centerline(
        min_hits=1,
        vehicle_x=0.0,
        vehicle_y=0.0,
        heading_x=1.0,
        heading_y=0.0,
    )
    assert len(left) == 1
    assert len(right) == 1
    assert len(center) == 1


def test_global_cone_memory_infers_orange_cones_by_vehicle_side():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[
            _u(
                assoc_x=0.0,
                assoc_y=2.0,
                update_x=0.0,
                update_y=2.0,
                source='camera',
                has_lidar=False,
                has_camera=True,
                camera_label='orange',
                camera_confidence=0.9,
            ),
            _u(
                assoc_x=0.0,
                assoc_y=-2.0,
                update_x=0.0,
                update_y=-2.0,
                source='camera',
                has_lidar=False,
                has_camera=True,
                camera_label='orange',
                camera_confidence=0.9,
            ),
        ],
        now_sec=1.0,
        gate_radius_m=0.2,
        spawn_radius_m=0.2,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )

    memory = GlobalConeMemory()
    memory.update_from_tracks(
        tracks=tracker.confirmed_tracks(1),
        now_sec=1.0,
        merge_radius_m=0.6,
        max_cones=2000,
    )

    left, right, center = memory.infer_boundaries_and_centerline(
        min_hits=1,
        vehicle_x=0.0,
        vehicle_y=0.0,
        heading_x=1.0,
        heading_y=0.0,
    )
    assert len(left) == 1
    assert len(right) == 1
    assert len(center) == 1


def test_global_cone_memory_can_filter_boundary_hypothesis_by_confidence():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[
            _u(
                assoc_x=0.0,
                assoc_y=2.0,
                update_x=0.0,
                update_y=2.0,
                source='camera',
                has_lidar=False,
                has_camera=True,
                camera_label='blue',
                camera_confidence=0.95,
            ),
            _u(
                assoc_x=0.0,
                assoc_y=-2.0,
                update_x=0.0,
                update_y=-2.0,
                source='camera',
                has_lidar=False,
                has_camera=True,
                camera_label='yellow',
                camera_confidence=0.55,
            ),
        ],
        now_sec=1.0,
        gate_radius_m=0.2,
        spawn_radius_m=0.2,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )

    memory = GlobalConeMemory()
    memory.update_from_tracks(
        tracks=tracker.confirmed_tracks(1),
        now_sec=1.0,
        merge_radius_m=0.6,
        max_cones=2000,
    )

    left, right, center = memory.infer_boundaries_and_centerline(
        min_hits=1,
        min_confidence=0.6,
        vehicle_x=0.0,
        vehicle_y=0.0,
        heading_x=1.0,
        heading_y=0.0,
    )
    assert len(left) == 1
    assert len(right) == 0
    assert len(center) == 0


def test_local_tracker_alpha_reduces_step_size():
    tracker_fast = LocalConeTracker()
    tracker_slow = LocalConeTracker()

    init = [_u(assoc_x=0.0, assoc_y=0.0, update_x=0.0, update_y=0.0)]
    tracker_fast.update(
        updates=init,
        now_sec=1.0,
        gate_radius_m=1.0,
        spawn_radius_m=1.0,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )
    tracker_slow.update(
        updates=init,
        now_sec=1.0,
        gate_radius_m=1.0,
        spawn_radius_m=1.0,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        min_seen_count=1,
    )

    step = [_u(assoc_x=1.0, assoc_y=0.0, update_x=1.0, update_y=0.0)]
    tracker_fast.update(
        updates=step,
        now_sec=2.0,
        gate_radius_m=1.0,
        spawn_radius_m=1.0,
        alpha_lidar=0.4,
        alpha_camera=0.2,
        min_seen_count=1,
    )
    tracker_slow.update(
        updates=step,
        now_sec=2.0,
        gate_radius_m=1.0,
        spawn_radius_m=1.0,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        min_seen_count=1,
    )

    assert tracker_fast.tracks[0].x > tracker_slow.tracks[0].x


def test_local_tracker_suppresses_near_duplicate_track_creation_outside_gate():
    tracker = LocalConeTracker()

    tracker.update(
        updates=[_u(assoc_x=10.0, assoc_y=1.5, update_x=10.0, update_y=1.5, camera_label='blue', has_camera=True)],
        now_sec=1.0,
        gate_radius_m=0.5,
        spawn_radius_m=0.85,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        min_seen_count=1,
    )

    stats = tracker.update(
        updates=[_u(assoc_x=10.58, assoc_y=1.62, update_x=10.58, update_y=1.62, camera_label='blue', has_camera=True)],
        now_sec=2.0,
        gate_radius_m=0.5,
        spawn_radius_m=0.85,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        min_seen_count=1,
    )

    assert len(tracker.tracks) == 1
    assert stats.new_tracks == 0
    assert stats.suppressed_new_tracks == 1


def test_local_tracker_merges_nearby_same_color_tracks():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[
            _u(assoc_x=5.0, assoc_y=1.2, update_x=5.0, update_y=1.2, camera_label='yellow', has_camera=True),
            _u(assoc_x=5.65, assoc_y=1.0, update_x=5.65, update_y=1.0, camera_label='yellow', has_camera=True),
        ],
        now_sec=1.0,
        gate_radius_m=0.2,
        spawn_radius_m=0.2,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        min_seen_count=1,
    )

    merged = tracker.merge_nearby_tracks(merge_radius_m=0.85)

    assert merged == 1
    assert len(tracker.tracks) == 1


def test_local_tracker_merges_longitudinal_same_side_aliases_in_base_frame():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[
            _u(assoc_x=10.0, assoc_y=1.50, update_x=10.0, update_y=1.50, camera_label='blue', has_camera=True),
            _u(assoc_x=11.45, assoc_y=1.56, update_x=11.45, update_y=1.56, camera_label='blue', has_camera=True),
        ],
        now_sec=1.0,
        gate_radius_m=0.2,
        spawn_radius_m=0.2,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        min_seen_count=1,
    )

    merged = tracker.merge_nearby_tracks(
        merge_radius_m=0.85,
        track_positions_in_base=[(10.0, 1.50), (11.45, 1.56)],
        longitudinal_tolerance_m=1.75,
        lateral_tolerance_m=0.25,
    )

    assert merged == 1
    assert len(tracker.tracks) == 1


def test_local_tracker_keeps_confirmed_track_as_stale_before_pruning():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[_u(assoc_x=1.0, assoc_y=1.0, update_x=1.0, update_y=1.0, camera_label='blue', has_camera=True)],
        now_sec=1.0,
        gate_radius_m=0.5,
        spawn_radius_m=0.5,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        confirm_hits=1,
    )

    pruned = tracker.prune(
        now_sec=1.8,
        tentative_ttl_sec=0.5,
        stale_after_sec=0.25,
        confirmed_prune_after_sec=3.0,
        max_range_m=100.0,
        behind_drop_m=100.0,
        unknown_drop_frames=0,
        confirm_hits=1,
        track_positions_in_base=[(1.0, 1.0)],
    )

    assert pruned == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_state == TRACK_STATE_STALE
    assert len(tracker.planner_tracks(now_sec=1.8, confirm_hits=1, publish_stale_tracks=True, stale_planner_ttl_sec=2.0)) == 1


def test_local_tracker_color_hysteresis_resists_single_bad_flip():
    tracker = LocalConeTracker()
    tracker.update(
        updates=[_u(assoc_x=0.0, assoc_y=0.0, update_x=0.0, update_y=0.0, camera_label='blue', has_camera=True, camera_confidence=0.95)],
        now_sec=1.0,
        gate_radius_m=0.5,
        spawn_radius_m=0.5,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        confirm_hits=1,
        color_switch_margin=0.25,
    )
    tracker.update(
        updates=[_u(assoc_x=0.0, assoc_y=0.0, update_x=0.0, update_y=0.0, camera_label='yellow', has_camera=True, camera_confidence=0.55)],
        now_sec=1.1,
        gate_radius_m=0.5,
        spawn_radius_m=0.5,
        alpha_lidar=0.25,
        alpha_camera=0.15,
        confirm_hits=1,
        color_switch_margin=0.25,
    )

    label, _conf = tracker.tracks[0].class_label()
    assert label == 'blue'


def test_convert_odom_child_pose_to_front_axle_applies_wheelbase_offset():
    pose = convert_odom_child_pose_to_base_frame(
        child_frame='base_footprint',
        base_frame='front_axle',
        tx=-20.0,
        ty=1.0,
        yaw=0.0,
        wheelbase_m=1.65,
        is_alias=lambda a, b: a == b,
    )

    assert pose is not None
    assert abs(pose[0] - (-19.175)) < 1e-9
    assert abs(pose[1] - 1.0) < 1e-9
