//! wgpu bridge entry point and resource lifecycle.

mod init;
mod pipeline_cache;
mod render;
mod resources;
mod surface;
mod types;

use std::collections::HashMap;

use crate::backend::wgpu::{WgpuDevice, WgpuInstance, WgpuSurface, WgpuSwapchain};
use crate::error::{Error, ErrorKind};
use rotex_types::{
    CreatedResources, DeviceDescriptor, Extent2D, FrameDescriptor, InstanceDescriptor,
    ResourceBatchCreate, ResourceBatchUpdate, SceneDescriptor, SurfaceDescriptor,
};

use self::types::{DepthTarget, MaterialPipelineKey, ResourceStorage};

/// wgpu-backed Rotex bridge.
///
/// Owns the wgpu instance and device, optional window surface and swapchain,
/// mesh/material/texture resources, cached render pipelines, and shared texture
/// bind-group layout resources.
pub struct WgpuBridge {
    pub(crate) instance: WgpuInstance,
    pub(crate) device: WgpuDevice,
    pub(crate) texture_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) texture_sampler: wgpu::Sampler,
    pub(crate) fallback_texture_bind_group: wgpu::BindGroup,
    pub(crate) surface: Option<WgpuSurface>,
    pub(crate) swapchain: Option<WgpuSwapchain>,
    pub(crate) resources: ResourceStorage,
    pub(crate) next_mesh_id: u64,
    pub(crate) next_material_id: u64,
    pub(crate) next_texture_id: u64,
    pub(crate) pipeline_cache: HashMap<MaterialPipelineKey, wgpu::RenderPipeline>,
    pub(crate) depth_target: Option<DepthTarget>,
}

impl WgpuBridge {
    /// Creates a bridge from `instance_descriptor` and `device_descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when instance or device creation fails or requested
    /// features are unsupported.
    pub async fn new(
        instance_descriptor: InstanceDescriptor,
        device_descriptor: DeviceDescriptor,
    ) -> Result<Self, Error> {
        init::create_bridge(instance_descriptor, device_descriptor).await
    }

    /// Attaches a window surface and creates an initial swapchain.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when surface or swapchain creation fails.
    pub fn attach_surface(&mut self, surface_descriptor: SurfaceDescriptor) -> Result<(), Error> {
        surface::attach_surface(self, surface_descriptor)
    }

    /// Creates meshes, materials, and textures described by `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when a resource descriptor is invalid or creation fails.
    pub fn create_resources(
        &mut self,
        descriptor: ResourceBatchCreate,
    ) -> Result<CreatedResources, Error> {
        resources::create_resources(self, descriptor)
    }

    /// Applies in-place updates from `descriptor` to existing resources.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when a referenced resource is missing or an update is invalid.
    pub fn update_resources(&mut self, descriptor: ResourceBatchUpdate) -> Result<(), Error> {
        resources::update_resources(self, descriptor)
    }

    /// Renders `scene` according to `frame` into the attached surface.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when no surface is attached, the frame is invalid, or
    /// rendering fails.
    pub fn render(
        &mut self,
        scene: &SceneDescriptor,
        frame: &FrameDescriptor,
    ) -> Result<(), Error> {
        render::render(self, scene, frame)
    }

    /// Resizes the swapchain to `extent`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when no surface is attached or reconfiguration fails.
    pub fn resize(&mut self, extent: Extent2D) -> Result<(), Error> {
        surface::resize(self, extent)
    }

    /// Drops bridge state and GPU resources.
    pub fn destroy(self) {}

    /// Returns static descriptions of wgpu/rotex_types feature gaps.
    pub fn unsupported_feature_reporting() -> &'static [&'static str] {
        &[
            "VertexStepMode is not modeled by rotex_types and is treated as vertex-rate input.",
            "VertexFormat support is limited to the rotex_types subset (Float32/Float32x2/Float32x3/Float32x4/Uint32).",
            "TextureFormat::Rgba8UnormSrgb is unavailable; textures use TextureFormat::Rgba8Unorm.",
            "Advanced Vulkan SPIR-V operations (for example hardware ray tracing or subgroup operations) are unsupported by WGPU WebAPI and may panic/fail during make_spirv translation.",
        ]
    }
}

pub(crate) fn surface_not_attached_error() -> Error {
    Error::fatal(ErrorKind::SurfaceNotAttached)
}
