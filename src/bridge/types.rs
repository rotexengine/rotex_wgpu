use std::collections::HashMap;

use rotex_types::{
    IndexFormat, MaterialDescriptor, MaterialId, MeshId, TextureFormat, TextureId, VertexAttribute,
    VertexFormat,
};

/// Translated vertex layout used when building wgpu pipelines.
#[derive(Debug, Clone)]
pub(crate) struct WgpuVertexLayout {
    /// Bytes between consecutive vertices.
    pub(crate) array_stride: u64,
    /// wgpu vertex attributes.
    pub(crate) attributes: Vec<wgpu::VertexAttribute>,
}

impl WgpuVertexLayout {
    /// Converts this layout into a wgpu vertex buffer layout.
    pub(crate) fn as_wgpu(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.array_stride,
            // rotex_types does not model step mode; WGPU backend uses vertex-rate input.
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &self.attributes,
        }
    }
}

/// GPU buffers and layout metadata for a mesh resource.
pub(crate) struct WgpuMeshResource {
    /// Vertex buffer uploaded to the GPU.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Index buffer uploaded to the GPU.
    pub(crate) index_buffer: wgpu::Buffer,
    /// Index element format.
    pub(crate) index_format: wgpu::IndexFormat,
    /// Number of indices to draw.
    pub(crate) index_count: u32,
    /// Stable hash of the source vertex layout.
    pub(crate) vertex_layout_id: u64,
    /// Translated vertex layout for pipeline creation.
    pub(crate) vertex_layout: WgpuVertexLayout,
}

impl WgpuMeshResource {
    /// Returns draw inputs needed to issue an indexed draw call.
    pub(crate) fn to_draw_inputs(
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

/// GPU texture and bind group for a texture resource.
pub(crate) struct WgpuTextureResource {
    /// Uploaded texture object.
    pub(crate) texture: wgpu::Texture,
    /// Bind group bound at draw time for texture sampling.
    pub(crate) bind_group: wgpu::BindGroup,
    /// wgpu texture format.
    pub(crate) format: wgpu::TextureFormat,
    /// Texture width and height in pixels.
    pub(crate) size: (u32, u32),
}

/// Depth attachment recreated when the swapchain size changes.
pub(crate) struct DepthTarget {
    _texture: wgpu::Texture,
    /// Depth texture view used as a render pass attachment.
    pub(crate) view: wgpu::TextureView,
    /// `(width, height)` the depth target was allocated for.
    pub(crate) size: (u32, u32),
}

impl DepthTarget {
    /// Creates a depth target from `texture`, `view`, and `size`.
    pub(crate) fn new(texture: wgpu::Texture, view: wgpu::TextureView, size: (u32, u32)) -> Self {
        Self {
            _texture: texture,
            view,
            size,
        }
    }
}

/// Cache key for a material render pipeline variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MaterialPipelineKey {
    /// Source material identifier.
    pub(crate) material_id: MaterialId,
    /// Hashed vertex layout used by the mesh draw.
    pub(crate) vertex_layout_id: u64,
    /// Whether depth testing is enabled for this pipeline.
    pub(crate) depth_enabled: bool,
}

/// In-memory registry of bridge-owned GPU resources.
#[derive(Default)]
pub(crate) struct ResourceStorage {
    /// Mesh resources keyed by mesh ID.
    pub(crate) meshes: HashMap<MeshId, WgpuMeshResource>,
    /// Material descriptors keyed by material ID.
    pub(crate) materials: HashMap<MaterialId, MaterialDescriptor>,
    /// Texture resources keyed by texture ID.
    pub(crate) textures: HashMap<TextureId, WgpuTextureResource>,
}

/// Maps a rotex [`VertexFormat`] to the corresponding wgpu format.
pub(crate) fn map_vertex_format(format: VertexFormat) -> wgpu::VertexFormat {
    match format {
        VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
    }
}

/// Returns the byte size of `format`.
pub(crate) fn vertex_format_size(format: VertexFormat) -> u64 {
    format.size()
}

/// Converts a rotex vertex attribute into a wgpu vertex attribute.
pub(crate) fn wgpu_vertex_attribute(attribute: VertexAttribute) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format: map_vertex_format(attribute.format),
        offset: attribute.offset,
        shader_location: attribute.location,
    }
}

/// Maps a rotex [`TextureFormat`] to the corresponding wgpu format.
pub(crate) fn map_texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Returns the number of bytes per texel for `format`.
pub(crate) fn bytes_per_pixel(format: TextureFormat) -> u32 {
    match format {
        TextureFormat::Rgba8Unorm => 4,
    }
}

/// Maps a rotex [`IndexFormat`] to the corresponding wgpu format.
pub(crate) fn map_index_format(format: IndexFormat) -> wgpu::IndexFormat {
    match format {
        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

/// Returns the byte size of one index element for `format`.
pub(crate) fn index_format_size(format: IndexFormat) -> usize {
    match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}
