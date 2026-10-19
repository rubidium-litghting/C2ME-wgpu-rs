// SPDX-License-Identifier: MIT

//! Priority calculation: chunks closer to any player get dispatched first.
//!
//! This is a low-cost integer Manhattan distance from the batcher's "player
//! interest set" (1–8 player positions, typically).

pub fn priority_of(cx: i32, cz: i32, player_centers: &[(i32, i32)]) -> u32 {
    let mut best = u32::MAX;
    for &(px, pz) in player_centers {
        let d = (cx - px).unsigned_abs() + (cz - pz).unsigned_abs();
        best = best.min(d);
    }
    best
}
