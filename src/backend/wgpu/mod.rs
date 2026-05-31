//! wgpu instance, device, and surface wrappers.

mod device;
mod surface;

pub(crate) use device::{WgpuDevice, WgpuInstance};
pub(crate) use surface::{WgpuSurface, WgpuSwapchain};
