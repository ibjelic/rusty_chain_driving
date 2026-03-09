# Rusty Chain Driving

A chaotic multi-player traffic dodging game built in Rust with [Bevy](https://bevyengine.org/).

1-5 players drive connected cars through dense traffic on a winding road. Smash through vehicles, watch them flip and tumble, and race to the finish line as fast as possible.

## Gameplay

- **Connected cars** — players are linked by ropes that interact with traffic
- **Dense traffic** — weave through randomly generated waves (always passable)
- **Ultimate collisions** — traffic cars launch, flip, and tumble in full 3D physics
- **Rope chaos** — ropes sweep through traffic, dragging and launching cars
- **Time attack** — race from start to finish, timer tracks your run

## Controls

| Player | Move | Brake |
|--------|------|-------|
| P1 | WASD | Space |
| P2 | Arrow keys | Right Shift |
| P3 | IJKL | U |
| P4 | Numpad 8456 | Numpad 0 |
| P5 | TFGH | V |

Change `NUM_PLAYERS` in `src/main.rs` to set the number of players (1-5).

## Build & Run

```bash
cargo run            # debug build
cargo run --release  # optimized build
```

## Requirements

- Rust 1.80+
- Bevy 0.18
