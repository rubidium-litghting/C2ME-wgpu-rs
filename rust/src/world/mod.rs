// SPDX-License-Identifier: MIT

//! World module: owns per-Minecraft-world state (pipeline + bind groups).

pub mod world;
pub mod pipeline_cache;
pub mod bind_group;
pub mod constants;
pub mod aquifer;
pub mod beard;

pub use world::WorldContext;
