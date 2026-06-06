mod compute_pipeline_cache;
mod init;
mod pipeline_cache;
mod render;
mod resources;
mod surface;
mod types;

use std::collections::HashMap;

use crate::backend::wgpu::{WgpuDevice, WgpuInstance, WgpuSurface, WgpuSwapchain};
use crate::error::{Error, ErrorKind};
use rotex_core::{
    Error as CoreError, ErrorKind as CoreErrorKind, RenderBackend, Severity as CoreSeverity,
};
use rotex_types::{
    CreatedResources, DeviceDescriptor, Extent2D, InstanceDescriptor, RenderCommand,
    ResourceBatchCreate, ResourceBatchUpdate, SceneDescriptor, SurfaceDescriptor, TextureId,
    TextureReadback,
};

use self::types::{DepthTarget, DepthTargetKey, MaterialPipelineKey, ResourceStorage};

pub struct WgpuBridge {
    pub(crate) instance: WgpuInstance,
    pub(crate) device: WgpuDevice,
    pub(crate) global_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) material_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) object_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) texture_sampler: wgpu::Sampler,
    pub(crate) global_uniform_buffer: wgpu::Buffer,
    pub(crate) object_uniform_buffer: wgpu::Buffer,
    pub(crate) global_bind_group: wgpu::BindGroup,
    pub(crate) object_bind_group: wgpu::BindGroup,
    pub(crate) fallback_texture_bind_group: wgpu::BindGroup,
    pub(crate) object_aligned_stride: u32,
    pub(crate) object_buffer_capacity: u32,
    pub(crate) surface: Option<WgpuSurface>,
    pub(crate) swapchain: Option<WgpuSwapchain>,
    pub(crate) resources: ResourceStorage,
    pub(crate) next_mesh_id: u64,
    pub(crate) next_material_id: u64,
    pub(crate) next_texture_id: u64,
    pub(crate) next_buffer_id: u64,
    pub(crate) next_compute_pipeline_id: u64,
    pub(crate) pipeline_cache: HashMap<MaterialPipelineKey, wgpu::RenderPipeline>,
    pub(crate) depth_targets: HashMap<DepthTargetKey, DepthTarget>,
}

impl WgpuBridge {
    pub async fn new(
        instance_descriptor: InstanceDescriptor,
        device_descriptor: DeviceDescriptor,
    ) -> Result<Self, Error> {
        init::create_bridge(instance_descriptor, device_descriptor).await
    }

    pub fn attach_surface(&mut self, surface_descriptor: SurfaceDescriptor) -> Result<(), Error> {
        surface::attach_surface(self, surface_descriptor)
    }

    pub fn create_resources(
        &mut self,
        descriptor: ResourceBatchCreate,
    ) -> Result<CreatedResources, Error> {
        resources::create_resources(self, descriptor)
    }

    pub fn update_resources(&mut self, descriptor: ResourceBatchUpdate) -> Result<(), Error> {
        resources::update_resources(self, descriptor)
    }

    pub fn execute(
        &mut self,
        scene: &SceneDescriptor,
        commands: &[RenderCommand],
    ) -> Result<(), Error> {
        render::execute(self, scene, commands)
    }

    pub fn resize(&mut self, extent: Extent2D) -> Result<(), Error> {
        surface::resize(self, extent)
    }

    pub fn read_texture(&mut self, _id: TextureId) -> Result<TextureReadback, Error> {
        Err(Error::fatal(ErrorKind::Unsupported(
            "Texture readback is not implemented on the WGPU backend",
        )))
    }

    pub fn destroy(self) {}

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

fn to_core_error(error: Error) -> CoreError {
    let severity = match error.severity {
        crate::error::Severity::Fatal => CoreSeverity::Fatal,
        crate::error::Severity::Info
        | crate::error::Severity::Warning
        | crate::error::Severity::Recoverable => CoreSeverity::Warning,
    };
    let kind = match error.kind {
        crate::error::ErrorKind::NoCompatibleDevice => CoreErrorKind::NoCompatibleDevice,
        crate::error::ErrorKind::Unsupported(message) => CoreErrorKind::Unsupported(message),
        other => CoreErrorKind::Backend(format!("{other:?}")),
    };
    CoreError { kind, severity }
}

impl RenderBackend for WgpuBridge {
    fn attach_surface(
        &mut self,
        surface_descriptor: rotex_types::SurfaceDescriptor,
    ) -> Result<(), CoreError> {
        WgpuBridge::attach_surface(self, surface_descriptor).map_err(to_core_error)
    }

    fn create_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchCreate,
    ) -> Result<rotex_types::CreatedResources, CoreError> {
        WgpuBridge::create_resources(self, descriptor).map_err(to_core_error)
    }

    fn update_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchUpdate,
    ) -> Result<(), CoreError> {
        WgpuBridge::update_resources(self, descriptor).map_err(to_core_error)
    }

    fn execute(
        &mut self,
        scene: &rotex_types::SceneDescriptor,
        commands: &[rotex_types::RenderCommand],
    ) -> Result<(), CoreError> {
        WgpuBridge::execute(self, scene, commands).map_err(to_core_error)
    }

    fn resize(&mut self, extent: rotex_types::Extent2D) -> Result<(), CoreError> {
        WgpuBridge::resize(self, extent).map_err(to_core_error)
    }

    fn read_texture(
        &mut self,
        id: rotex_types::TextureId,
    ) -> Result<rotex_types::TextureReadback, CoreError> {
        WgpuBridge::read_texture(self, id).map_err(to_core_error)
    }

    fn destroy(self: Box<Self>) {
        WgpuBridge::destroy(*self);
    }
}
