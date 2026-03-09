use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use rand::Rng;
use std::f32::consts::PI;

// ─── Configuration ───────────────────────────────────────────────────────────────
const NUM_PLAYERS: usize = 2; // Change to 1-5

// ─── Constants ──────────────────────────────────────────────────────────────────
const CAR_Y: f32 = 0.35;
const CAR_RADIUS: f32 = 0.9;

// Road
const ROAD_LENGTH: f32 = 3000.0;
const ROAD_SEGMENTS: usize = 150;
const NUM_LANES: usize = 3;
const LANE_WIDTH: f32 = 3.5;

// Longitudinal
const CAR_MAX_SPEED: f32 = 38.0;
const CAR_REVERSE_MAX: f32 = 10.0;
const CAR_ACCELERATION: f32 = 42.0;
const CAR_BRAKE_FORCE: f32 = 55.0;
const ENGINE_BRAKING: f32 = 2.5;
const THROTTLE_RAMP: f32 = 5.0;

// Steering
const TURN_SPEED_BASE: f32 = 4.0;
const STEERING_SPEED_DAMPING: f32 = 0.15;
const STEER_RESPONSE: f32 = 12.0;

// Grip / Drift
const NORMAL_GRIP: f32 = 1.0;
const DRIFT_GRIP: f32 = 0.35;
const LATERAL_GRIP_STRENGTH: f32 = 12.0;
const DRIFT_ENTER_ANGLE: f32 = 0.45;
const DRIFT_EXIT_ANGLE: f32 = 0.15;
const DRIFT_STEER_MULT: f32 = 1.4;
const DRAG_COEFFICIENT: f32 = 0.25;
const DRAG_QUADRATIC: f32 = 0.008;

// Weight transfer
const WEIGHT_TRANSFER_PITCH: f32 = 0.03;
const WEIGHT_TRANSFER_ROLL: f32 = 0.025;
const WEIGHT_TRANSFER_LERP: f32 = 6.0;

// Traffic
const TRAFFIC_MIN_SPEED: f32 = 14.0;
const TRAFFIC_MAX_SPEED: f32 = 24.0;
const TRAFFIC_SPAWN_AHEAD: f32 = 180.0;
const TRAFFIC_DESPAWN_BEHIND: f32 = 80.0;
const TRAFFIC_WAVE_SPACING: f32 = 14.0;
const TRAFFIC_FRICTION: f32 = 0.4;
const ROPE_TRAFFIC_PUSH: f32 = 18.0;

// Rope
const ROPE_PARTICLES: usize = 20;
const STRING_MAX: f32 = 14.0;
const STRING_STIFFNESS: f32 = 14.0;
const ROPE_GRAVITY: f32 = 40.0;
const ROPE_ITERATIONS: usize = 10;

// Collision
const CAR_BOUNCE_FACTOR: f32 = 1.5;
const CAR_SPIN_FACTOR: f32 = 3.0;
const CAR_COLLISION_RADIUS: f32 = 1.0;

// Camera
const CAM_HEIGHT: f32 = 7.5;
const CAM_BEHIND: f32 = 5.0;
const CAM_LOOKAHEAD: f32 = 18.0;
const CAM_POS_LERP: f32 = 7.0;
const CAM_SHAKE_DECAY: f32 = 8.0;
const CAM_SHAKE_COLLISION: f32 = 0.6;

const FINISH_Z: f32 = ROAD_LENGTH - 200.0;

// ─── Components ─────────────────────────────────────────────────────────────────
#[derive(Component)]
struct Car {
    velocity: Vec3,
    facing: f32,
    speed: f32,
    throttle: f32,
    angular_velocity: f32,
    drift_angle: f32,
    is_drifting: bool,
    drift_timer: f32,
    body_pitch: f32,
    body_roll: f32,
}

#[derive(Component)]
struct PlayerIdx(usize);

#[derive(Component)]
struct TrafficCar {
    speed: f32,
    lane: usize,
    vtype: usize, // 0=compact, 1=sedan, 2=truck
    hit: bool,
    velocity: Vec3,
    spin: f32,
    tumble: Vec3, // 3D angular velocity for flipping
    half_extents: Vec3,
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct TimerText;

// ─── Resources ──────────────────────────────────────────────────────────────────
#[derive(Resource)]
struct Road {
    centerline: Vec<Vec3>,
    forward_dirs: Vec<Vec3>,
    right_dirs: Vec<Vec3>,
}

#[derive(Resource)]
struct RopeState {
    segments: Vec<Vec<Vec3>>,
    prev_segments: Vec<Vec<Vec3>>,
}

#[derive(Resource)]
struct CameraState {
    shake_intensity: f32,
    shake_timer: f32,
}

#[derive(Resource)]
struct GameTimer {
    elapsed: f32,
    started: bool,
    finished: bool,
}

#[derive(Resource)]
struct TrafficManager {
    next_wave_z: f32,
}

#[derive(Resource)]
struct TrafficAssets {
    body_meshes: [Handle<Mesh>; 3],
    cabin_meshes: [Handle<Mesh>; 3],
    body_colors: Vec<Handle<StandardMaterial>>,
    cabin_mat: Handle<StandardMaterial>,
    half_extents: [Vec3; 3],
    cabin_y_offsets: [f32; 3],
}

// ─── Helpers ────────────────────────────────────────────────────────────────────
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut diff = b - a;
    while diff > PI { diff -= 2.0 * PI; }
    while diff < -PI { diff += 2.0 * PI; }
    a + diff * t
}

fn generate_road_centerline() -> Vec<Vec3> {
    let seg = ROAD_LENGTH / ROAD_SEGMENTS as f32;
    (0..=ROAD_SEGMENTS)
        .map(|i| {
            let z = i as f32 * seg;
            let t = i as f32 / ROAD_SEGMENTS as f32;
            let x = (t * PI * 3.0).sin() * 25.0
                + (t * PI * 7.0).sin() * 8.0
                + (t * PI * 13.0).sin() * 3.0;
            Vec3::new(x, 0.0, z)
        })
        .collect()
}

fn road_center_at_z(road: &Road, z: f32) -> (Vec3, Vec3, Vec3) {
    let cl = &road.centerline;
    let z = z.clamp(cl[0].z, cl[cl.len() - 1].z);
    let seg = (cl[1].z - cl[0].z).max(0.01);
    let idx = ((z - cl[0].z) / seg).floor() as usize;
    let idx = idx.min(cl.len() - 2);
    let t = ((z - cl[idx].z) / (cl[idx + 1].z - cl[idx].z)).clamp(0.0, 1.0);
    let center = cl[idx].lerp(cl[idx + 1], t);
    (center, road.forward_dirs[idx], road.right_dirs[idx])
}

fn lane_x_offset(lane: usize) -> f32 {
    let half_w = NUM_LANES as f32 * LANE_WIDTH * 0.5;
    (lane as f32 + 0.5) * LANE_WIDTH - half_w
}

fn get_controls(idx: usize, keys: &ButtonInput<KeyCode>) -> (bool, bool, bool, bool, bool) {
    match idx {
        0 => (
            keys.pressed(KeyCode::KeyW),
            keys.pressed(KeyCode::KeyS),
            keys.pressed(KeyCode::KeyA),
            keys.pressed(KeyCode::KeyD),
            keys.pressed(KeyCode::Space),
        ),
        1 => (
            keys.pressed(KeyCode::ArrowUp),
            keys.pressed(KeyCode::ArrowDown),
            keys.pressed(KeyCode::ArrowLeft),
            keys.pressed(KeyCode::ArrowRight),
            keys.pressed(KeyCode::ShiftRight),
        ),
        2 => (
            keys.pressed(KeyCode::KeyI),
            keys.pressed(KeyCode::KeyK),
            keys.pressed(KeyCode::KeyJ),
            keys.pressed(KeyCode::KeyL),
            keys.pressed(KeyCode::KeyU),
        ),
        3 => (
            keys.pressed(KeyCode::Numpad8),
            keys.pressed(KeyCode::Numpad5),
            keys.pressed(KeyCode::Numpad4),
            keys.pressed(KeyCode::Numpad6),
            keys.pressed(KeyCode::Numpad0),
        ),
        4 => (
            keys.pressed(KeyCode::KeyT),
            keys.pressed(KeyCode::KeyG),
            keys.pressed(KeyCode::KeyF),
            keys.pressed(KeyCode::KeyH),
            keys.pressed(KeyCode::KeyV),
        ),
        _ => (false, false, false, false, false),
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────────
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Traffic Dash — Get to the finish!".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.52, 0.72, 0.88)))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                player_input,
                traffic_spawner,
                traffic_movement,
                player_traffic_collision,
                player_player_collision,
                car_road_bounds,
                rope_chain_physics,
                rope_traffic_collision,
                game_timer_system,
                traffic_despawn,
                camera_follow,
                draw_ropes,
                draw_road_markings,
                draw_speed_lines,
                update_ui,
            )
                .chain(),
        )
        .run();
}

// ─── Setup ──────────────────────────────────────────────────────────────────────
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let centerline = generate_road_centerline();
    let n = centerline.len();

    // Precompute directions
    let mut forward_dirs = Vec::with_capacity(n);
    let mut right_dirs = Vec::with_capacity(n);
    for i in 0..n - 1 {
        let fwd = (centerline[i + 1] - centerline[i]).normalize_or_zero();
        forward_dirs.push(fwd);
        right_dirs.push(Vec3::new(fwd.z, 0.0, -fwd.x));
    }
    forward_dirs.push(*forward_dirs.last().unwrap());
    right_dirs.push(*right_dirs.last().unwrap());

    // ─── Camera ─────────────────────────────────────────────────────────────
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 0.9,
            ..default()
        }),
        Transform::from_xyz(0.0, CAM_HEIGHT, -CAM_BEHIND),
        MainCamera,
        Bloom { intensity: 0.2, ..default() },
        Tonemapping::AcesFitted,
        DistanceFog {
            color: Color::srgb(0.52, 0.72, 0.88),
            falloff: FogFalloff::Exponential { density: 0.004 },
            ..default()
        },
    ));

    // ─── Lighting ───────────────────────────────────────────────────────────
    commands.spawn((
        DirectionalLight {
            illuminance: 20000.0,
            shadows_enabled: true,
            color: Color::srgb(1.0, 0.95, 0.85),
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.65, 0.4, 0.0)),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 4000.0,
            shadows_enabled: false,
            color: Color::srgb(0.7, 0.8, 1.0),
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, -1.8, 0.0)),
    ));
    commands.spawn(AmbientLight {
        color: Color::srgb(0.55, 0.6, 0.75),
        brightness: 350.0,
        affects_lightmapped_meshes: true,
    });

    // ─── Ground ─────────────────────────────────────────────────────────────
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(400.0, 0.05, ROAD_LENGTH + 400.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.5, 0.2),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.08, ROAD_LENGTH * 0.5),
    ));

    // ─── Road surface ───────────────────────────────────────────────────────
    let road_width = NUM_LANES as f32 * LANE_WIDTH;
    let seg_len = ROAD_LENGTH / ROAD_SEGMENTS as f32 + 1.5;
    let road_mesh = meshes.add(Cuboid::new(road_width, 0.05, seg_len));
    let road_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.18, 0.2),
        perceptual_roughness: 0.7,
        metallic: 0.05,
        reflectance: 0.4,
        ..default()
    });

    // Gravel shoulder
    let gravel_mesh = meshes.add(Cuboid::new(road_width + 4.0, 0.04, seg_len));
    let gravel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.45, 0.35),
        perceptual_roughness: 0.95,
        ..default()
    });

    for i in 0..ROAD_SEGMENTS {
        let a = centerline[i];
        let b = centerline[i + 1];
        let mid = (a + b) * 0.5;
        let fwd = forward_dirs[i];
        let angle = f32::atan2(fwd.x, fwd.z);

        commands.spawn((
            Mesh3d(gravel_mesh.clone()),
            MeshMaterial3d(gravel_mat.clone()),
            Transform::from_xyz(mid.x, -0.04, mid.z)
                .with_rotation(Quat::from_rotation_y(angle)),
        ));
        commands.spawn((
            Mesh3d(road_mesh.clone()),
            MeshMaterial3d(road_mat.clone()),
            Transform::from_xyz(mid.x, -0.01, mid.z)
                .with_rotation(Quat::from_rotation_y(angle)),
        ));
    }

    // ─── Barriers (guardrails) ──────────────────────────────────────────────
    let barrier_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.7, 0.72),
        metallic: 0.6,
        perceptual_roughness: 0.4,
        ..default()
    });
    let barrier_mesh = meshes.add(Cuboid::new(0.15, 0.6, seg_len));
    let half_w = road_width * 0.5 + 0.3;

    for i in (0..ROAD_SEGMENTS).step_by(2) {
        let a = centerline[i];
        let b = centerline[i + 1];
        let mid = (a + b) * 0.5;
        let fwd = forward_dirs[i];
        let right = right_dirs[i];
        let angle = f32::atan2(fwd.x, fwd.z);

        for sign in [-1.0_f32, 1.0] {
            let pos = mid + right * sign * half_w;
            commands.spawn((
                Mesh3d(barrier_mesh.clone()),
                MeshMaterial3d(barrier_mat.clone()),
                Transform::from_xyz(pos.x, 0.3, pos.z)
                    .with_rotation(Quat::from_rotation_y(angle)),
            ));
        }
    }

    // ─── Start/finish checkerboards ─────────────────────────────────────────
    let check_black = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.08, 0.08),
        ..default()
    });
    let check_white = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.95),
        ..default()
    });

    for &line_z in &[30.0_f32, FINISH_Z] {
        let (center, fwd, right) = road_center_at_z_raw(&centerline, &forward_dirs, &right_dirs, line_z);
        let angle = f32::atan2(fwd.x, fwd.z);
        for row in 0..2 {
            for col in 0..10 {
                let is_white = (row + col) % 2 == 0;
                let offset_along = (row as f32 - 0.5) * 1.0;
                let offset_perp = (col as f32 - 4.5) * 1.0;
                let pos = center + fwd * offset_along + right * offset_perp + Vec3::Y * 0.02;
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::new(1.0, 0.02, 1.0))),
                    MeshMaterial3d(if is_white { check_white.clone() } else { check_black.clone() }),
                    Transform::from_translation(pos)
                        .with_rotation(Quat::from_rotation_y(angle)),
                ));
            }
        }
    }

    // ─── Roadside trees ─────────────────────────────────────────────────────
    let trunk_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.22, 0.1),
        perceptual_roughness: 0.95,
        ..default()
    });
    let foliage_colors = [
        Color::srgb(0.12, 0.45, 0.1),
        Color::srgb(0.18, 0.52, 0.14),
        Color::srgb(0.1, 0.4, 0.08),
        Color::srgb(0.2, 0.5, 0.18),
    ];
    let trunk_mesh = meshes.add(Cylinder { radius: 0.3, half_height: 1.5 });
    let foliage_mesh = meshes.add(Sphere::new(2.0));
    let mut rng = rand::thread_rng();

    for i in (0..ROAD_SEGMENTS).step_by(3) {
        let mid = (centerline[i] + centerline[i + 1]) * 0.5;
        let right = right_dirs[i];
        let sign = if i % 6 < 3 { 1.0 } else { -1.0 };
        let offset = half_w + rng.gen_range(3.0..8.0);
        let pos = mid + right * sign * offset;
        let color = foliage_colors[i % foliage_colors.len()];

        commands.spawn((
            Mesh3d(trunk_mesh.clone()),
            MeshMaterial3d(trunk_mat.clone()),
            Transform::from_xyz(pos.x, 1.5, pos.z),
        ));
        commands.spawn((
            Mesh3d(foliage_mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: 0.9,
                ..default()
            })),
            Transform::from_xyz(pos.x, 4.5, pos.z),
        ));
    }

    // ─── Finish banner posts ────────────────────────────────────────────────
    let banner_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.1, 0.1),
        emissive: LinearRgba::new(4.0, 0.3, 0.3, 1.0),
        ..default()
    });
    let (fc, _ff, fr) = road_center_at_z_raw(&centerline, &forward_dirs, &right_dirs, FINISH_Z);
    for sign in [-1.0_f32, 1.0] {
        let pos = fc + fr * sign * (half_w + 1.0);
        commands.spawn((
            Mesh3d(meshes.add(Cylinder { radius: 0.15, half_height: 3.0 })),
            MeshMaterial3d(banner_mat.clone()),
            Transform::from_xyz(pos.x, 3.0, pos.z),
        ));
    }
    // Banner crossbar
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(road_width + 2.0, 0.3, 0.15))),
        MeshMaterial3d(banner_mat),
        Transform::from_xyz(fc.x, 5.8, fc.z)
            .with_rotation(Quat::from_rotation_y(f32::atan2(_ff.x, _ff.z))),
    ));

    // ─── Player cars ────────────────────────────────────────────────────────
    let player_colors = [
        (Color::srgb(0.85, 0.12, 0.1), Color::srgb(0.55, 0.08, 0.06)),
        (Color::srgb(0.1, 0.3, 0.85), Color::srgb(0.06, 0.18, 0.55)),
        (Color::srgb(0.1, 0.75, 0.2), Color::srgb(0.06, 0.5, 0.12)),
        (Color::srgb(0.9, 0.8, 0.1), Color::srgb(0.6, 0.55, 0.06)),
        (Color::srgb(0.65, 0.15, 0.8), Color::srgb(0.4, 0.1, 0.55)),
    ];

    let start_z = 35.0;
    let (_sc, sf, _sr) = road_center_at_z_raw(&centerline, &forward_dirs, &right_dirs, start_z);
    let start_facing = f32::atan2(sf.x, sf.z);

    for i in 0..NUM_PLAYERS {
        let lane_off = lane_x_offset(i % NUM_LANES);
        let z_off = -(i as f32) * 4.0;
        let (pc, _pf, pr) = road_center_at_z_raw(&centerline, &forward_dirs, &right_dirs, start_z + z_off);
        let pos = pc + pr * lane_off + Vec3::Y * CAR_Y;

        let (bc, cc) = player_colors[i % player_colors.len()];
        spawn_player_car(
            &mut commands,
            &mut meshes,
            &mut materials,
            bc,
            cc,
            pos,
            start_facing,
            i,
        );
    }

    // ─── Rope init ──────────────────────────────────────────────────────────
    let mut rope_segments = Vec::new();
    let mut rope_prev = Vec::new();
    for seg in 0..NUM_PLAYERS.saturating_sub(1) {
        let z_a = start_z - seg as f32 * 4.0;
        let z_b = start_z - (seg + 1) as f32 * 4.0;
        let (ca, _, ra) = road_center_at_z_raw(&centerline, &forward_dirs, &right_dirs, z_a);
        let (cb, _, rb) = road_center_at_z_raw(&centerline, &forward_dirs, &right_dirs, z_b);
        let a_pos = ca + ra * lane_x_offset(seg % NUM_LANES) + Vec3::Y * CAR_Y;
        let b_pos = cb + rb * lane_x_offset((seg + 1) % NUM_LANES) + Vec3::Y * CAR_Y;

        let mut particles = Vec::with_capacity(ROPE_PARTICLES);
        for j in 0..ROPE_PARTICLES {
            let t = j as f32 / (ROPE_PARTICLES - 1) as f32;
            particles.push(a_pos.lerp(b_pos, t));
        }
        rope_prev.push(particles.clone());
        rope_segments.push(particles);
    }

    // ─── Traffic assets ─────────────────────────────────────────────────────
    let traffic_sizes: [(f32, f32, f32); 3] = [
        (1.2, 0.4, 2.0),  // compact
        (1.4, 0.45, 2.4), // sedan
        (2.0, 0.9, 4.0),  // truck
    ];
    let cabin_sizes: [(f32, f32, f32); 3] = [
        (0.9, 0.3, 0.8),
        (1.1, 0.32, 1.0),
        (1.8, 0.45, 1.5),
    ];

    let body_meshes = [
        meshes.add(Cuboid::new(traffic_sizes[0].0, traffic_sizes[0].1, traffic_sizes[0].2)),
        meshes.add(Cuboid::new(traffic_sizes[1].0, traffic_sizes[1].1, traffic_sizes[1].2)),
        meshes.add(Cuboid::new(traffic_sizes[2].0, traffic_sizes[2].1, traffic_sizes[2].2)),
    ];
    let cabin_meshes = [
        meshes.add(Cuboid::new(cabin_sizes[0].0, cabin_sizes[0].1, cabin_sizes[0].2)),
        meshes.add(Cuboid::new(cabin_sizes[1].0, cabin_sizes[1].1, cabin_sizes[1].2)),
        meshes.add(Cuboid::new(cabin_sizes[2].0, cabin_sizes[2].1, cabin_sizes[2].2)),
    ];

    let car_palette = [
        Color::srgb(0.92, 0.92, 0.92),
        Color::srgb(0.12, 0.12, 0.12),
        Color::srgb(0.65, 0.65, 0.67),
        Color::srgb(0.8, 0.15, 0.1),
        Color::srgb(0.1, 0.3, 0.75),
        Color::srgb(0.15, 0.5, 0.15),
        Color::srgb(0.75, 0.65, 0.1),
        Color::srgb(0.4, 0.2, 0.12),
    ];
    let body_colors: Vec<Handle<StandardMaterial>> = car_palette
        .iter()
        .map(|&c| {
            materials.add(StandardMaterial {
                base_color: c,
                metallic: 0.5,
                perceptual_roughness: 0.4,
                ..default()
            })
        })
        .collect();
    let cabin_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.15, 0.18),
        metallic: 0.3,
        perceptual_roughness: 0.6,
        ..default()
    });

    let half_extents = [
        Vec3::new(traffic_sizes[0].0 * 0.5, traffic_sizes[0].1 * 0.5, traffic_sizes[0].2 * 0.5),
        Vec3::new(traffic_sizes[1].0 * 0.5, traffic_sizes[1].1 * 0.5, traffic_sizes[1].2 * 0.5),
        Vec3::new(traffic_sizes[2].0 * 0.5, traffic_sizes[2].1 * 0.5, traffic_sizes[2].2 * 0.5),
    ];
    let cabin_y_offsets = [
        traffic_sizes[0].1 * 0.5 + cabin_sizes[0].1 * 0.5,
        traffic_sizes[1].1 * 0.5 + cabin_sizes[1].1 * 0.5,
        traffic_sizes[2].1 * 0.5 + cabin_sizes[2].1 * 0.5,
    ];

    // ─── UI ─────────────────────────────────────────────────────────────────
    commands.spawn((
        Text::new("Time: 0.0s | Dist: 0m"),
        TextFont { font_size: 32.0, ..default() },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(14.0),
            left: Val::Px(14.0),
            ..default()
        },
        TimerText,
    ));

    // Controls hint
    let controls_text = match NUM_PLAYERS {
        1 => "P1: WASD+Space",
        2 => "P1: WASD+Space | P2: Arrows+RShift",
        3 => "P1: WASD+Space | P2: Arrows+RShift | P3: IJKL+U",
        4 => "P1: WASD | P2: Arrows | P3: IJKL | P4: Numpad",
        _ => "P1: WASD | P2: Arrows | P3: IJKL | P4: Numpad | P5: TFGH",
    };
    commands.spawn((
        Text::new(controls_text),
        TextFont { font_size: 16.0, ..default() },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(14.0),
            left: Val::Px(14.0),
            ..default()
        },
    ));

    // ─── Insert resources ───────────────────────────────────────────────────
    commands.insert_resource(Road { centerline, forward_dirs, right_dirs });
    commands.insert_resource(RopeState { segments: rope_segments, prev_segments: rope_prev });
    commands.insert_resource(CameraState { shake_intensity: 0.0, shake_timer: 0.0 });
    commands.insert_resource(GameTimer { elapsed: 0.0, started: false, finished: false });
    commands.insert_resource(TrafficManager { next_wave_z: 100.0 });
    commands.insert_resource(TrafficAssets {
        body_meshes,
        cabin_meshes,
        body_colors,
        cabin_mat,
        half_extents,
        cabin_y_offsets,
    });
}

// Helper that works before Road resource exists
fn road_center_at_z_raw(
    centerline: &[Vec3],
    forward_dirs: &[Vec3],
    right_dirs: &[Vec3],
    z: f32,
) -> (Vec3, Vec3, Vec3) {
    let z = z.clamp(centerline[0].z, centerline[centerline.len() - 1].z);
    let seg = (centerline[1].z - centerline[0].z).max(0.01);
    let idx = ((z - centerline[0].z) / seg).floor() as usize;
    let idx = idx.min(centerline.len() - 2);
    let t = ((z - centerline[idx].z) / (centerline[idx + 1].z - centerline[idx].z)).clamp(0.0, 1.0);
    let center = centerline[idx].lerp(centerline[idx + 1], t);
    (center, forward_dirs[idx], right_dirs[idx])
}

// ─── Spawn Player Car ───────────────────────────────────────────────────────────
fn spawn_player_car(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    body_color: Color,
    cabin_color: Color,
    position: Vec3,
    facing: f32,
    player_index: usize,
) {
    let body_mat = materials.add(StandardMaterial {
        base_color: body_color,
        metallic: 0.7,
        perceptual_roughness: 0.25,
        reflectance: 0.6,
        ..default()
    });
    let cabin_mat = materials.add(StandardMaterial {
        base_color: cabin_color,
        metallic: 0.4,
        perceptual_roughness: 0.4,
        ..default()
    });
    let wheel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.1, 0.1),
        metallic: 0.8,
        perceptual_roughness: 0.5,
        ..default()
    });
    let chrome_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.8, 0.85),
        metallic: 0.95,
        perceptual_roughness: 0.1,
        reflectance: 0.9,
        ..default()
    });

    let tip_colors = [
        Color::srgb(1.0, 0.3, 0.3),
        Color::srgb(0.3, 0.5, 1.0),
        Color::srgb(0.3, 1.0, 0.4),
        Color::srgb(1.0, 0.9, 0.2),
        Color::srgb(0.8, 0.3, 1.0),
    ];
    let tip_color = tip_colors[player_index % tip_colors.len()];
    let tip_lin = LinearRgba::from(tip_color);
    let tip_mat = materials.add(StandardMaterial {
        base_color: tip_color,
        emissive: LinearRgba::new(tip_lin.red * 12.0, tip_lin.green * 12.0, tip_lin.blue * 12.0, 1.0),
        ..default()
    });
    let headlight_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 1.0, 0.9),
        emissive: LinearRgba::new(10.0, 10.0, 7.0, 1.0),
        ..default()
    });
    let taillight_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.05, 0.0),
        emissive: LinearRgba::new(8.0, 0.3, 0.1, 1.0),
        ..default()
    });

    let body_mesh = meshes.add(Cuboid::new(1.2, 0.3, 2.0));
    let cabin_mesh = meshes.add(Cuboid::new(0.9, 0.25, 0.8));
    let wheel_mesh = meshes.add(Cylinder { radius: 0.2, half_height: 0.08 });
    let axle_mesh = meshes.add(Cylinder { radius: 0.06, half_height: 0.55 });
    let antenna_mesh = meshes.add(Cylinder { radius: 0.02, half_height: 0.3 });
    let tip_mesh = meshes.add(Sphere::new(0.06));
    let spoiler_mesh = meshes.add(Cuboid::new(1.0, 0.12, 0.05));
    let spoiler_arm = meshes.add(Cuboid::new(0.06, 0.15, 0.06));
    let headlight_mesh = meshes.add(Sphere::new(0.09));
    let taillight_mesh = meshes.add(Cuboid::new(0.15, 0.08, 0.03));
    let bumper_mesh = meshes.add(Cuboid::new(1.1, 0.1, 0.08));
    let windshield_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.4, 0.5, 0.7, 0.3),
        alpha_mode: AlphaMode::Blend,
        metallic: 0.9,
        perceptual_roughness: 0.05,
        ..default()
    });
    let windshield_mesh = meshes.add(Cuboid::new(0.85, 0.22, 0.02));

    commands
        .spawn((
            Mesh3d(body_mesh),
            MeshMaterial3d(body_mat),
            Transform::from_translation(position).with_rotation(Quat::from_rotation_y(facing)),
            Car {
                velocity: Vec3::ZERO,
                facing,
                speed: 0.0,
                throttle: 0.0,
                angular_velocity: 0.0,
                drift_angle: 0.0,
                is_drifting: false,
                drift_timer: 0.0,
                body_pitch: 0.0,
                body_roll: 0.0,
            },
            PlayerIdx(player_index),
        ))
        .with_children(|parent| {
            parent.spawn((Mesh3d(cabin_mesh), MeshMaterial3d(cabin_mat), Transform::from_xyz(0.0, 0.275, 0.15)));
            parent.spawn((Mesh3d(windshield_mesh), MeshMaterial3d(windshield_mat), Transform::from_xyz(0.0, 0.32, 0.55)));
            parent.spawn((Mesh3d(spoiler_mesh), MeshMaterial3d(chrome_mat.clone()), Transform::from_xyz(0.0, 0.35, -0.85)));
            parent.spawn((Mesh3d(spoiler_arm.clone()), MeshMaterial3d(chrome_mat.clone()), Transform::from_xyz(0.35, 0.24, -0.85)));
            parent.spawn((Mesh3d(spoiler_arm), MeshMaterial3d(chrome_mat.clone()), Transform::from_xyz(-0.35, 0.24, -0.85)));
            parent.spawn((Mesh3d(bumper_mesh.clone()), MeshMaterial3d(chrome_mat.clone()), Transform::from_xyz(0.0, -0.08, 1.0)));
            parent.spawn((Mesh3d(bumper_mesh), MeshMaterial3d(chrome_mat.clone()), Transform::from_xyz(0.0, -0.08, -0.98)));
            for (wx, wy, wz) in [(0.65, -0.08, 0.6), (-0.65, -0.08, 0.6), (0.65, -0.08, -0.6), (-0.65, -0.08, -0.6)] {
                parent.spawn((
                    Mesh3d(wheel_mesh.clone()),
                    MeshMaterial3d(wheel_mat.clone()),
                    Transform::from_xyz(wx, wy, wz).with_rotation(Quat::from_rotation_z(PI / 2.0)),
                ));
            }
            for wz in [0.6, -0.6] {
                parent.spawn((
                    Mesh3d(axle_mesh.clone()),
                    MeshMaterial3d(chrome_mat.clone()),
                    Transform::from_xyz(0.0, -0.08, wz).with_rotation(Quat::from_rotation_z(PI / 2.0)),
                ));
            }
            parent.spawn((Mesh3d(antenna_mesh), MeshMaterial3d(chrome_mat), Transform::from_xyz(0.35, 0.45, -0.7)));
            parent.spawn((Mesh3d(tip_mesh), MeshMaterial3d(tip_mat), Transform::from_xyz(0.35, 0.78, -0.7)));
            parent.spawn((Mesh3d(headlight_mesh.clone()), MeshMaterial3d(headlight_mat.clone()), Transform::from_xyz(0.4, 0.02, 1.01)));
            parent.spawn((Mesh3d(headlight_mesh), MeshMaterial3d(headlight_mat), Transform::from_xyz(-0.4, 0.02, 1.01)));
            parent.spawn((Mesh3d(taillight_mesh.clone()), MeshMaterial3d(taillight_mat.clone()), Transform::from_xyz(0.4, 0.02, -1.0)));
            parent.spawn((Mesh3d(taillight_mesh), MeshMaterial3d(taillight_mat), Transform::from_xyz(-0.4, 0.02, -1.0)));
        });
}

// ─── Player Input ───────────────────────────────────────────────────────────────
fn player_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cars: Query<(&mut Car, &mut Transform, &PlayerIdx)>,
) {
    let dt = time.delta_secs().min(0.03);

    for (mut car, mut transform, pidx) in cars.iter_mut() {
        let (fwd_key, back_key, left, right, brake) = get_controls(pidx.0, &keys);

        // Throttle ramp
        let target_throttle = if fwd_key { 1.0 } else if back_key { -1.0 } else { 0.0 };
        car.throttle += (target_throttle - car.throttle) * (THROTTLE_RAMP * dt).min(1.0);

        let forward_dir = Vec3::new(car.facing.sin(), 0.0, car.facing.cos());
        car.speed = car.velocity.dot(forward_dir);
        let max_speed = CAR_MAX_SPEED;

        // Longitudinal forces
        let mut accel = 0.0;
        if car.throttle > 0.01 {
            let speed_ratio = (car.speed / max_speed).clamp(0.0, 1.0);
            accel = CAR_ACCELERATION * car.throttle * (1.0 - speed_ratio * speed_ratio);
        } else if car.throttle < -0.01 {
            if car.speed > 0.5 {
                accel = -CAR_BRAKE_FORCE;
            } else {
                let rev_ratio = (-car.speed / CAR_REVERSE_MAX).clamp(0.0, 1.0);
                accel = -CAR_ACCELERATION * 0.5 * (1.0 - rev_ratio * rev_ratio);
            }
        } else if car.speed.abs() > 0.1 {
            accel = -car.speed.signum() * ENGINE_BRAKING * car.speed.abs().min(10.0);
        }

        if brake && car.speed.abs() > 0.1 {
            accel -= car.speed.signum() * CAR_BRAKE_FORCE * 0.7;
        }

        car.velocity += forward_dir * accel * dt;

        // Steering
        let steer_input = if left { 1.0_f32 } else if right { -1.0 } else { 0.0 };
        let effective_turn = TURN_SPEED_BASE / (1.0 + car.speed.abs() * STEERING_SPEED_DAMPING);
        let drift_mult = if car.is_drifting { DRIFT_STEER_MULT } else { 1.0 };
        let target_angular = steer_input * effective_turn * drift_mult;
        car.angular_velocity += (target_angular - car.angular_velocity) * (STEER_RESPONSE * dt).min(1.0);

        if car.speed.abs() > 0.3 {
            let steer_sign = if car.speed > 0.0 { 1.0 } else { -1.0 };
            car.facing += car.angular_velocity * steer_sign * dt;
        }

        // Drift model
        let right_dir = Vec3::new(car.facing.cos(), 0.0, -car.facing.sin());
        let lateral_speed = car.velocity.dot(right_dir);
        let speed_xz = Vec2::new(car.velocity.x, car.velocity.z).length();

        let slip_angle = if speed_xz > 0.5 {
            (lateral_speed.abs() / speed_xz).min(1.0)
        } else {
            0.0
        };

        if !car.is_drifting {
            if slip_angle > DRIFT_ENTER_ANGLE || (brake && steer_input.abs() > 0.1 && car.speed > 5.0) {
                car.is_drifting = true;
                car.drift_timer = 0.0;
            }
        } else {
            car.drift_timer += dt;
            if slip_angle < DRIFT_EXIT_ANGLE && !brake {
                car.is_drifting = false;
            }
        }
        car.drift_angle = slip_angle;

        // Lateral grip
        let grip = if car.is_drifting { DRIFT_GRIP } else { NORMAL_GRIP };
        let lateral_correction = -lateral_speed * grip * LATERAL_GRIP_STRENGTH * dt;
        car.velocity += right_dir * lateral_correction;

        // Quadratic drag
        let spd = Vec2::new(car.velocity.x, car.velocity.z).length();
        car.velocity *= 1.0 - (DRAG_COEFFICIENT + DRAG_QUADRATIC * spd) * dt;

        // Integration
        transform.translation += car.velocity * dt;
        transform.translation.y = CAR_Y;

        // Weight transfer
        let target_pitch = -accel * WEIGHT_TRANSFER_PITCH / CAR_ACCELERATION;
        let target_roll = car.angular_velocity * car.speed.abs().min(15.0) * WEIGHT_TRANSFER_ROLL;
        car.body_pitch += (target_pitch.clamp(-0.08, 0.08) - car.body_pitch) * (WEIGHT_TRANSFER_LERP * dt).min(1.0);
        car.body_roll += (target_roll.clamp(-0.06, 0.06) - car.body_roll) * (WEIGHT_TRANSFER_LERP * dt).min(1.0);

        transform.rotation = Quat::from_rotation_y(car.facing)
            * Quat::from_rotation_x(car.body_pitch)
            * Quat::from_rotation_z(car.body_roll);
    }
}

// ─── Traffic Spawner ────────────────────────────────────────────────────────────
fn traffic_spawner(
    mut commands: Commands,
    mut manager: ResMut<TrafficManager>,
    assets: Res<TrafficAssets>,
    road: Res<Road>,
    players: Query<&Transform, With<PlayerIdx>>,
) {
    let mut max_z = f32::MIN;
    for t in players.iter() {
        max_z = max_z.max(t.translation.z);
    }

    let mut rng = rand::thread_rng();

    while manager.next_wave_z < max_z + TRAFFIC_SPAWN_AHEAD && manager.next_wave_z < FINISH_Z - 50.0 {
        let z = manager.next_wave_z;

        // Decide which lanes to fill — always leave at least 1 open
        let num_to_fill = rng.gen_range(1..NUM_LANES);
        let mut lanes: Vec<usize> = (0..NUM_LANES).collect();
        for i in (1..lanes.len()).rev() {
            let j = rng.gen_range(0..=i);
            lanes.swap(i, j);
        }

        for &lane in lanes.iter().take(num_to_fill) {
            let vtype = if rng.gen_bool(0.15) { 2 } else if rng.gen_bool(0.5) { 1 } else { 0 };
            let color_idx = rng.gen_range(0..assets.body_colors.len());
            let speed = match vtype {
                2 => rng.gen_range(TRAFFIC_MIN_SPEED..TRAFFIC_MIN_SPEED + 4.0),
                1 => rng.gen_range(TRAFFIC_MIN_SPEED + 2.0..TRAFFIC_MAX_SPEED),
                _ => rng.gen_range(TRAFFIC_MIN_SPEED + 3.0..TRAFFIC_MAX_SPEED + 2.0),
            };

            let (center, fwd, right) = road_center_at_z(&road, z + rng.gen_range(-3.0..3.0));
            let lane_off = lane_x_offset(lane);
            let he = assets.half_extents[vtype];
            let pos = center + right * lane_off + Vec3::Y * (he.y + 0.01);
            let facing = f32::atan2(fwd.x, fwd.z);

            commands
                .spawn((
                    Mesh3d(assets.body_meshes[vtype].clone()),
                    MeshMaterial3d(assets.body_colors[color_idx].clone()),
                    Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(facing)),
                    TrafficCar {
                        speed,
                        lane,
                        vtype,
                        hit: false,
                        velocity: Vec3::ZERO,
                        spin: 0.0,
                        tumble: Vec3::ZERO,
                        half_extents: he,
                    },
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Mesh3d(assets.cabin_meshes[vtype].clone()),
                        MeshMaterial3d(assets.cabin_mat.clone()),
                        Transform::from_xyz(0.0, assets.cabin_y_offsets[vtype], 0.0),
                    ));
                });
        }

        manager.next_wave_z += TRAFFIC_WAVE_SPACING + rng.gen_range(-5.0..8.0);
    }
}

// ─── Traffic Movement ───────────────────────────────────────────────────────────
fn traffic_movement(
    time: Res<Time>,
    road: Res<Road>,
    mut traffic: Query<(&mut TrafficCar, &mut Transform)>,
) {
    let dt = time.delta_secs().min(0.03);

    for (mut tc, mut t) in traffic.iter_mut() {
        if tc.hit {
            // Full 3D physics for launched cars
            t.translation += tc.velocity * dt;
            tc.velocity.y -= 20.0 * dt; // gravity

            if t.translation.y < tc.half_extents.y {
                t.translation.y = tc.half_extents.y;
                // Bounce off ground
                if tc.velocity.y < -2.0 {
                    tc.velocity.y = tc.velocity.y.abs() * 0.4;
                    // Add random tumble on bounce
                    tc.tumble.x += tc.velocity.z * 0.1;
                    tc.tumble.z += tc.velocity.x * 0.1;
                } else {
                    tc.velocity.y = 0.0;
                }
                tc.velocity.x *= 0.92;
                tc.velocity.z *= 0.92;
            }

            tc.velocity *= 1.0 - TRAFFIC_FRICTION * dt;

            // Apply 3D tumble rotation (flipping!)
            let tumble_len = tc.tumble.length();
            if tumble_len > 0.01 {
                let tumble_axis = tc.tumble / tumble_len;
                t.rotation = Quat::from_axis_angle(tumble_axis, tumble_len * dt) * t.rotation;
            }
            tc.tumble *= 1.0 - 1.5 * dt; // dampen tumble
            tc.spin *= 1.0 - 2.0 * dt;
            continue;
        }

        // Normal AI: drive forward in lane
        t.translation.z += tc.speed * dt;
        let (center, fwd, right) = road_center_at_z(&road, t.translation.z);
        let lane_off = lane_x_offset(tc.lane);
        t.translation.x = center.x + right.x * lane_off;
        t.translation.y = tc.half_extents.y + 0.01;
        let facing = f32::atan2(fwd.x, fwd.z);
        t.rotation = Quat::from_rotation_y(facing);
    }
}

// ─── Traffic Despawn ────────────────────────────────────────────────────────────
fn traffic_despawn(
    mut commands: Commands,
    traffic: Query<(Entity, &Transform, &TrafficCar)>,
    players: Query<&Transform, With<PlayerIdx>>,
) {
    let mut min_z = f32::MAX;
    for t in players.iter() {
        min_z = min_z.min(t.translation.z);
    }

    for (entity, t, tc) in traffic.iter() {
        let behind = t.translation.z < min_z - TRAFFIC_DESPAWN_BEHIND;
        let off_road = tc.hit && (t.translation.y > 30.0 || t.translation.y < -5.0
            || (t.translation.x - 0.0).abs() > 100.0);
        if behind || off_road {
            commands.entity(entity).despawn();
        }
    }
}

// ─── Player vs Traffic Collision ────────────────────────────────────────────────
fn player_traffic_collision(
    mut cam_state: ResMut<CameraState>,
    mut players: Query<(&mut Car, &mut Transform), With<PlayerIdx>>,
    mut traffic: Query<(&mut TrafficCar, &mut Transform), Without<PlayerIdx>>,
) {
    for (mut car, mut car_t) in players.iter_mut() {
        for (mut tc, mut tc_t) in traffic.iter_mut() {
            if tc.hit { continue; }

            // Sphere vs OBB collision
            let local_pos = tc_t.rotation.inverse() * (car_t.translation - tc_t.translation);
            let he = tc.half_extents;
            let closest = Vec3::new(
                local_pos.x.clamp(-he.x, he.x),
                local_pos.y.clamp(-he.y, he.y),
                local_pos.z.clamp(-he.z, he.z),
            );
            let diff = local_pos - closest;
            let dist = diff.length();

            if dist < CAR_RADIUS && dist > 0.001 {
                let local_normal = diff / dist;
                let normal = tc_t.rotation * local_normal;
                let penetration = CAR_RADIUS - dist;

                // Separate
                car_t.translation += normal * penetration * 0.4;
                tc_t.translation -= normal * penetration * 0.6;

                let vel_dot = car.velocity.dot(normal);
                if vel_dot < 0.0 {
                    let impact_speed = vel_dot.abs();

                    // Player bounces back
                    car.velocity -= normal * vel_dot * 0.4;

                    // Traffic gets LAUNCHED!
                    tc.hit = true;
                    tc.velocity = -normal * impact_speed * 2.2
                        + car.velocity * 0.5
                        + Vec3::Y * (impact_speed * 0.7).min(14.0);

                    let right = Vec3::new(car.facing.cos(), 0.0, -car.facing.sin());
                    tc.spin = normal.dot(right) * impact_speed * 0.6;

                    // 3D tumble — cars flip and roll
                    let cross_dir = normal.cross(Vec3::Y);
                    tc.tumble = cross_dir * impact_speed * 1.5
                        + Vec3::new(normal.z, 0.0, -normal.x) * impact_speed * 0.8;

                    // Camera shake
                    let shake = (impact_speed / 10.0).min(1.0) * CAM_SHAKE_COLLISION;
                    cam_state.shake_intensity = cam_state.shake_intensity.max(shake);

                    if impact_speed > 6.0 {
                        car.is_drifting = true;
                        car.drift_timer = 0.0;
                    }
                }
            }
        }
    }
}

// ─── Player vs Player Collision ─────────────────────────────────────────────────
fn player_player_collision(
    mut cam_state: ResMut<CameraState>,
    mut players: Query<(Entity, &mut Car, &mut Transform, &PlayerIdx)>,
) {
    // Collect positions
    let data: Vec<(Entity, Vec3, Vec3, f32)> = players
        .iter()
        .map(|(e, car, t, _)| (e, t.translation, car.velocity, car.facing))
        .collect();

    let mut impulses: Vec<(Entity, Vec3, Vec3, f32)> = Vec::new();

    for i in 0..data.len() {
        for j in (i + 1)..data.len() {
            let diff = data[i].1 - data[j].1;
            let diff_xz = Vec3::new(diff.x, 0.0, diff.z);
            let dist = diff_xz.length();

            if dist < CAR_COLLISION_RADIUS * 2.0 && dist > 0.001 {
                let normal = diff_xz / dist;
                let penetration = CAR_COLLISION_RADIUS * 2.0 - dist;
                let rel_vel = data[i].2 - data[j].2;
                let vel_along = rel_vel.dot(normal);

                if vel_along < 0.0 {
                    let impulse_vec = normal * vel_along * CAR_BOUNCE_FACTOR;
                    let right_i = Vec3::new(data[i].3.cos(), 0.0, -data[i].3.sin());
                    let right_j = Vec3::new(data[j].3.cos(), 0.0, -data[j].3.sin());

                    impulses.push((data[i].0, normal * penetration * 0.5, -impulse_vec * 0.5, normal.dot(right_i) * vel_along.abs() * CAR_SPIN_FACTOR * 0.1));
                    impulses.push((data[j].0, -normal * penetration * 0.5, impulse_vec * 0.5, -normal.dot(right_j) * vel_along.abs() * CAR_SPIN_FACTOR * 0.1));

                    let impact = vel_along.abs() / 15.0;
                    cam_state.shake_intensity = cam_state.shake_intensity.max(CAM_SHAKE_COLLISION * impact.min(1.0));
                }
            }
        }
    }

    for (entity, pos_d, vel_d, ang_d) in impulses {
        if let Ok((_, mut car, mut t, _)) = players.get_mut(entity) {
            t.translation += pos_d;
            car.velocity += vel_d;
            car.angular_velocity += ang_d;
        }
    }
}

// ─── Road Bounds ────────────────────────────────────────────────────────────────
fn car_road_bounds(
    road: Res<Road>,
    mut cars: Query<(&mut Car, &mut Transform), With<PlayerIdx>>,
) {
    let half_w = NUM_LANES as f32 * LANE_WIDTH * 0.5;

    for (mut car, mut t) in cars.iter_mut() {
        let (center, _fwd, right) = road_center_at_z(&road, t.translation.z);
        let to_car = t.translation - center;
        let lateral = to_car.dot(right);

        if lateral.abs() > half_w {
            let push = right * (lateral.signum() * half_w - lateral);
            t.translation.x += push.x;
            t.translation.z += push.z;

            let vel_out = car.velocity.dot(right) * lateral.signum();
            if vel_out > 0.0 {
                car.velocity -= right * lateral.signum() * vel_out * 1.2;
            }
        }
    }
}

// ─── Rope Chain Physics ─────────────────────────────────────────────────────────
fn rope_chain_physics(
    time: Res<Time>,
    mut rope: ResMut<RopeState>,
    mut players: Query<(&mut Car, &mut Transform, &PlayerIdx)>,
) {
    let dt = time.delta_secs().min(0.03);
    let n_ropes = rope.segments.len();
    if n_ropes == 0 { return; }

    // Collect player positions sorted by index
    let mut positions: Vec<(usize, Vec3, Quat)> = players
        .iter()
        .map(|(_, t, p)| (p.0, t.translation, t.rotation))
        .collect();
    positions.sort_by_key(|(idx, _, _)| *idx);

    if positions.len() < 2 { return; }

    // Update each rope segment
    for seg in 0..n_ropes {
        if seg + 1 >= positions.len() { break; }

        let anchor_a = positions[seg].1 + positions[seg].2 * Vec3::new(0.0, 0.0, -0.9);
        let anchor_b = positions[seg + 1].1 + positions[seg + 1].2 * Vec3::new(0.0, 0.0, -0.9);

        let pn = rope.segments[seg].len();
        let segment_len = STRING_MAX / (pn - 1) as f32;

        rope.segments[seg][0] = anchor_a;
        rope.prev_segments[seg][0] = anchor_a;
        rope.segments[seg][pn - 1] = anchor_b;
        rope.prev_segments[seg][pn - 1] = anchor_b;

        // Verlet integration
        for i in 1..pn - 1 {
            let vel = rope.segments[seg][i] - rope.prev_segments[seg][i];
            rope.prev_segments[seg][i] = rope.segments[seg][i];
            rope.segments[seg][i] += vel * 0.99;
            rope.segments[seg][i].y -= ROPE_GRAVITY * dt * dt;
        }

        // Constraints
        for _ in 0..ROPE_ITERATIONS {
            rope.segments[seg][0] = anchor_a;
            rope.segments[seg][pn - 1] = anchor_b;

            for i in 0..pn - 1 {
                let delta = rope.segments[seg][i + 1] - rope.segments[seg][i];
                let dist = delta.length();
                if dist > segment_len && dist > 0.001 {
                    let correction = (dist - segment_len) / dist * 0.5;
                    let offset = delta * correction;
                    if i > 0 { rope.segments[seg][i] += offset; }
                    if i + 1 < pn - 1 { rope.segments[seg][i + 1] -= offset; }
                }
            }

            for i in 1..pn - 1 {
                if rope.segments[seg][i].y < 0.08 {
                    rope.segments[seg][i].y = 0.08;
                }
            }
        }
    }

    // Apply pull forces
    let mut forces: Vec<(usize, Vec3, Vec3)> = Vec::new();

    for seg in 0..n_ropes {
        let pn = rope.segments[seg].len();
        let total_len: f32 = (0..pn - 1)
            .map(|i| rope.segments[seg][i].distance(rope.segments[seg][i + 1]))
            .sum();

        if total_len > STRING_MAX * 0.85 {
            let excess = total_len - STRING_MAX * 0.85;
            let force = excess * STRING_STIFFNESS * dt;

            let pull_a = (rope.segments[seg][1] - rope.segments[seg][0]).normalize_or_zero();
            let pull_b = (rope.segments[seg][pn - 2] - rope.segments[seg][pn - 1]).normalize_or_zero();

            forces.push((seg, pull_a * force * 0.5, pull_a * force * 0.3));
            forces.push((seg + 1, pull_b * force * 0.5, pull_b * force * 0.3));
        }
    }

    for (mut car, mut t, pidx) in players.iter_mut() {
        for &(idx, trans_f, vel_f) in &forces {
            if pidx.0 == idx {
                t.translation += trans_f;
                car.velocity += vel_f;
            }
        }
    }
}

// ─── Rope vs Traffic Collision ───────────────────────────────────────────────────
fn rope_traffic_collision(
    rope: Res<RopeState>,
    mut cam_state: ResMut<CameraState>,
    mut traffic: Query<(&mut TrafficCar, &mut Transform)>,
) {
    for seg in &rope.segments {
        for i in 0..seg.len() {
            let rp = seg[i];

            for (mut tc, tc_t) in traffic.iter_mut() {
                if tc.hit { continue; }

                // Check rope particle vs traffic OBB
                let local_pos = tc_t.rotation.inverse() * (rp - tc_t.translation);
                let he = tc.half_extents;
                let closest = Vec3::new(
                    local_pos.x.clamp(-he.x, he.x),
                    local_pos.y.clamp(-he.y, he.y),
                    local_pos.z.clamp(-he.z, he.z),
                );
                let diff = local_pos - closest;
                let dist = diff.length();

                if dist < 0.3 && dist > 0.001 {
                    let local_normal = diff / dist;
                    let normal = tc_t.rotation * local_normal;

                    // Rope particle is inside/touching traffic car — launch it!
                    tc.hit = true;

                    // Direction: rope pushes sideways/up
                    tc.velocity = normal * ROPE_TRAFFIC_PUSH
                        + Vec3::Y * 6.0;

                    let cross_dir = normal.cross(Vec3::Y);
                    tc.tumble = cross_dir * 8.0 + Vec3::Y * 3.0;
                    tc.spin = normal.x.signum() * 4.0;

                    // Subtle camera shake
                    cam_state.shake_intensity = cam_state.shake_intensity.max(0.2);
                }
            }
        }
    }
}

// ─── Game Timer ─────────────────────────────────────────────────────────────────
fn game_timer_system(
    time: Res<Time>,
    mut timer: ResMut<GameTimer>,
    players: Query<&Transform, With<PlayerIdx>>,
) {
    if timer.finished { return; }

    let any_moved = players.iter().any(|t| t.translation.z > 40.0);
    if any_moved && !timer.started {
        timer.started = true;
    }

    if timer.started {
        timer.elapsed += time.delta_secs();
    }

    let all_finished = players.iter().all(|t| t.translation.z > FINISH_Z);
    if all_finished && timer.started {
        timer.finished = true;
    }
}

// ─── Camera Follow ──────────────────────────────────────────────────────────────
fn camera_follow(
    time: Res<Time>,
    mut cam_state: ResMut<CameraState>,
    road: Res<Road>,
    mut cam: Query<(&mut Transform, &mut Projection), With<MainCamera>>,
    players: Query<&Transform, (With<PlayerIdx>, Without<MainCamera>)>,
) {
    let dt = time.delta_secs().min(0.03);
    let Ok((mut cam_t, mut proj)) = cam.single_mut() else { return };

    let mut sum = Vec3::ZERO;
    let mut count = 0;
    let mut max_z = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_x = f32::MAX;
    for t in players.iter() {
        sum += t.translation;
        count += 1;
        max_z = max_z.max(t.translation.z);
        min_z = min_z.min(t.translation.z);
        max_x = max_x.max(t.translation.x);
        min_x = min_x.min(t.translation.x);
    }
    if count == 0 { return; }

    let mid = sum / count as f32;
    let spread_z = (max_z - min_z).max(0.0);
    let spread_x = (max_x - min_x).max(0.0);
    let spread = spread_z.max(spread_x);

    // Camera zooms out only as much as needed to keep all players visible
    let extra_height = (spread * 0.4).min(8.0);
    let extra_behind = (spread * 0.25).min(5.0);

    let focus_z = min_z + (max_z - min_z) * 0.5;
    let height = CAM_HEIGHT + extra_height;
    let behind = CAM_BEHIND + extra_behind;

    let (focus_center, _, _) = road_center_at_z(&road, focus_z);
    // Camera X follows road center closely so road fills the screen
    let desired = Vec3::new(focus_center.x, height, focus_z - behind);

    // Shake
    cam_state.shake_intensity *= (1.0 - CAM_SHAKE_DECAY * dt).max(0.0);
    cam_state.shake_timer += dt * 30.0;
    let shake_x = cam_state.shake_intensity * (cam_state.shake_timer * 7.3).sin();
    let shake_y = cam_state.shake_intensity * (cam_state.shake_timer * 11.1).cos();

    cam_t.translation = cam_t.translation.lerp(desired, CAM_POS_LERP * dt);
    cam_t.translation.x += shake_x;
    cam_t.translation.y += shake_y;

    let (ahead_center, _, _) = road_center_at_z(&road, focus_z + CAM_LOOKAHEAD);
    let look_target = Vec3::new(ahead_center.x, 0.3, focus_z + CAM_LOOKAHEAD);
    cam_t.look_at(look_target, Vec3::Y);

    // Dynamic FOV: widen when players spread apart, tighter otherwise
    let target_fov = 0.85 + (spread / 20.0).min(0.5);
    if let Projection::Perspective(ref mut persp) = *proj {
        persp.fov += (target_fov - persp.fov) * (4.0 * dt).min(1.0);
    }
}

// ─── Draw Ropes ─────────────────────────────────────────────────────────────────
fn draw_ropes(mut gizmos: Gizmos, rope: Res<RopeState>) {
    for seg in &rope.segments {
        let pn = seg.len();
        let total_len: f32 = (0..pn - 1)
            .map(|i| seg[i].distance(seg[i + 1]))
            .sum();
        let tension = ((total_len - STRING_MAX * 0.7) / (STRING_MAX * 0.3)).clamp(0.0, 1.0);
        let r = 1.0;
        let g = 0.75 * (1.0 - tension);
        let b = 0.08 * (1.0 - tension);
        let color = Color::srgb(r, g, b);
        let dim = Color::srgb(r * 0.6, g * 0.6, b * 0.6);

        let offsets = [
            Vec3::ZERO,
            Vec3::new(0.03, 0.0, 0.0),
            Vec3::new(-0.03, 0.0, 0.0),
            Vec3::new(0.0, 0.03, 0.0),
            Vec3::new(0.0, -0.03, 0.0),
        ];
        for (idx, offset) in offsets.iter().enumerate() {
            let c = if idx == 0 { color } else { dim };
            let pts: Vec<Vec3> = seg.iter().map(|p| *p + *offset).collect();
            gizmos.linestrip(pts, c);
        }
    }
}

// ─── Draw Road Markings ─────────────────────────────────────────────────────────
fn draw_road_markings(
    mut gizmos: Gizmos,
    road: Res<Road>,
    players: Query<&Transform, With<PlayerIdx>>,
) {
    let mut avg_z = 0.0;
    let mut count = 0;
    for t in players.iter() {
        avg_z += t.translation.z;
        count += 1;
    }
    if count == 0 { return; }
    avg_z /= count as f32;

    let cl = &road.centerline;
    let half_w = NUM_LANES as f32 * LANE_WIDTH * 0.5;

    for i in 0..cl.len() - 1 {
        let mid_z = (cl[i].z + cl[i + 1].z) * 0.5;
        if (mid_z - avg_z).abs() > 150.0 { continue; }

        let a = cl[i];
        let b = cl[i + 1];
        let right = road.right_dirs[i];

        // Edge lines
        for sign in [-1.0_f32, 1.0] {
            let ea = a + right * sign * half_w + Vec3::Y * 0.03;
            let eb = b + right * sign * half_w + Vec3::Y * 0.03;
            gizmos.line(ea, eb, Color::srgba(1.0, 1.0, 1.0, 0.5));
        }

        // Lane dividers (dashed)
        if i % 3 == 0 {
            for lane in 1..NUM_LANES {
                let offset = (lane as f32) * LANE_WIDTH - half_w;
                let la = a + right * offset + Vec3::Y * 0.03;
                let lb = b + right * offset + Vec3::Y * 0.03;
                gizmos.line(la, lb, Color::srgba(1.0, 1.0, 1.0, 0.3));
            }
        }
    }
}

// ─── Speed Lines ────────────────────────────────────────────────────────────────
fn draw_speed_lines(
    mut gizmos: Gizmos,
    time: Res<Time>,
    cars: Query<(&Car, &Transform), With<PlayerIdx>>,
) {
    let elapsed = time.elapsed_secs();
    for (car, t) in cars.iter() {
        let speed = Vec2::new(car.velocity.x, car.velocity.z).length();
        if speed < 10.0 { continue; }
        let intensity = ((speed - 10.0) / (CAR_MAX_SPEED - 10.0)).clamp(0.0, 1.0);
        let count = (intensity * 5.0) as i32 + 1;
        let line_len = 0.5 + intensity * 2.5;
        let vel_dir = Vec3::new(car.velocity.x, 0.0, car.velocity.z).normalize_or_zero();
        let right = Vec3::new(vel_dir.z, 0.0, -vel_dir.x);

        for i in 0..count {
            let phase = elapsed * 5.0 + i as f32 * 1.7;
            let lateral = (phase * 0.73).sin() * 1.2;
            let height = 0.2 + (phase * 1.1).cos().abs() * 0.5;
            let start = t.translation - vel_dir * 1.5 + right * lateral + Vec3::Y * height;
            let end = start - vel_dir * line_len;
            gizmos.line(start, end, Color::srgba(1.0, 1.0, 1.0, intensity * 0.4));
        }
    }
}

// ─── UI ─────────────────────────────────────────────────────────────────────────
fn update_ui(
    timer: Res<GameTimer>,
    mut query: Query<&mut Text, With<TimerText>>,
    players: Query<&Transform, With<PlayerIdx>>,
) {
    let mut max_z = 0.0_f32;
    for t in players.iter() {
        max_z = max_z.max(t.translation.z);
    }
    let dist_remaining = (FINISH_Z - max_z).max(0.0) as i32;
    let secs = timer.elapsed;

    for mut text in query.iter_mut() {
        if timer.finished {
            text.0 = format!("FINISHED! Time: {:.1}s", secs);
        } else if timer.started {
            text.0 = format!("Time: {:.1}s | Dist: {}m", secs, dist_remaining);
        } else {
            text.0 = "GO! Drive forward to start the timer".to_string();
        }
    }
}
