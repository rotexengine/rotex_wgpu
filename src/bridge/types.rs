use std::collections::HashMap;

use rotex_types::resource::BufferId;
use rotex_types::{
    ComputePipelineDescriptor, ComputePipelineId, IndexFormat, MaterialDescriptor, MaterialId,
    MeshId, TextureFormat, TextureId, VertexAttribute, VertexFormat,
};

#[derive(Debug, Clone)]
pub struct WgpuVertexLayout {
    pub array_stride: u64,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

impl WgpuVertexLayout {
    pub fn as_wgpu(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.array_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &self.attributes,
        }
    }
}

pub struct WgpuMeshResource {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub index_count: u32,
    pub vertex_layout_id: u64,
    pub vertex_layout: WgpuVertexLayout,
}

impl WgpuMeshResource {
    pub fn to_draw_inputs(
        &self,
    ) -> (
        u64,
        WgpuVertexLayout,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::IndexFormat,
        u32,
    ) {
        (
            self.vertex_layout_id,
            self.vertex_layout.clone(),
            self.vertex_buffer.clone(),
            self.index_buffer.clone(),
            self.index_format,
            self.index_count,
        )
    }
}

pub struct WgpuTextureResource {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub render_view: Option<wgpu::TextureView>,
    pub bind_group: wgpu::BindGroup,
    pub format: wgpu::TextureFormat,
    pub size: (u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthTargetKey {
    Swapchain { width: u32, height: u32 },
    Texture(TextureId),
}

pub struct DepthTarget {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: (u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialPipelineKey {
    pub material_id: MaterialId,
    pub vertex_layout_id: u64,
    pub depth_enabled: bool,
    pub color_format: wgpu::TextureFormat,
}

pub struct WgpuBufferResource {
    pub buffer: wgpu::Buffer,
    pub size: u64,
}

pub struct WgpuComputePipelineResource {
    pub descriptor: ComputePipelineDescriptor,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

#[derive(Default)]
pub struct ResourceStorage {
    pub meshes: HashMap<MeshId, WgpuMeshResource>,
    pub materials: HashMap<MaterialId, MaterialDescriptor>,
    pub textures: HashMap<TextureId, WgpuTextureResource>,
    pub buffers: HashMap<BufferId, WgpuBufferResource>,
    pub compute_pipelines: HashMap<ComputePipelineId, WgpuComputePipelineResource>,
}

pub const GLOBAL_UBO_SIZE: u64 = 128;
pub const OBJECT_MATRIX_SIZE: u64 = 64;
pub const INITIAL_OBJECT_CAPACITY: u32 = 256;

pub fn map_vertex_format(format: VertexFormat) -> wgpu::VertexFormat {
    match format {
        VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
    }
}

pub fn vertex_format_size(format: VertexFormat) -> u64 {
    format.size()
}

pub fn wgpu_vertex_attribute(attribute: VertexAttribute) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format: map_vertex_format(attribute.format),
        offset: attribute.offset,
        shader_location: attribute.location,
    }
}

pub fn map_texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
    }
}

pub fn bytes_per_pixel(format: TextureFormat) -> u32 {
    match format {
        TextureFormat::Rgba8Unorm => 4,
    }
}

pub fn map_index_format(format: IndexFormat) -> wgpu::IndexFormat {
    match format {
        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

pub fn index_format_size(format: IndexFormat) -> usize {
    match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}

pub fn align_uniform_size(size: u64, alignment: u32) -> u32 {
    let alignment = alignment.max(1) as u64;
    ((size + alignment - 1) / alignment * alignment) as u32
}
