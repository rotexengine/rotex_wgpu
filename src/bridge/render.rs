use crate::error::{Error, ErrorKind};
use rotex_types::{
    ColorAttachmentLoad, DepthAttachmentLoad, PassColorTarget, PassDescriptor, RenderCommand,
    SceneDescriptor,
};

use super::WgpuBridge;
use super::compute_pipeline_cache;
use super::pipeline_cache;
use super::surface;
use super::types::{DepthTarget, DepthTargetKey, GLOBAL_UBO_SIZE, OBJECT_MATRIX_SIZE};

struct ColorTargetView {
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

pub(super) fn execute(
    bridge: &mut WgpuBridge,
    scene: &SceneDescriptor,
    commands: &[RenderCommand],
) -> Result<(), Error> {
    if commands.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "frame_has_no_commands",
        )));
    }

    let needs_swapchain = commands.iter().any(|command| {
        matches!(
            command,
            RenderCommand::DrawGraphics(pass)
                if pass.color_target == PassColorTarget::Swapchain
        )
    });

    let surface_texture = if needs_swapchain {
        match surface::acquire_surface_texture(bridge)? {
            Some(texture) => Some(texture),
            None => return Ok(()),
        }
    } else {
        None
    };

    let swapchain_color_view = surface_texture.as_ref().map(|surface_texture| {
        surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    });
    let swapchain_format = bridge
        .swapchain
        .as_ref()
        .map(|swapchain| swapchain.config.format);

    let mut encoder = bridge
        .device
        .raw
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rotex-wgpu-encoder"),
        });

    for command in commands {
        match command {
            RenderCommand::TransitionBuffer { .. } => {}
            RenderCommand::DispatchCompute(compute_pass) => {
                compute_pipeline_cache::record_compute_dispatch(
                    bridge,
                    &mut encoder,
                    compute_pass,
                )?;
            }
            RenderCommand::DrawGraphics(pass) => {
                record_graphics_pass(
                    bridge,
                    scene,
                    pass,
                    swapchain_format,
                    swapchain_color_view.as_ref(),
                    &mut encoder,
                )?;
            }
        }
    }

    bridge.device.queue.submit(Some(encoder.finish()));
    if let Some(surface_texture) = surface_texture {
        surface_texture.present();
    }
    Ok(())
}

fn record_graphics_pass(
    bridge: &mut WgpuBridge,
    scene: &SceneDescriptor,
    pass: &PassDescriptor,
    swapchain_format: Option<wgpu::TextureFormat>,
    swapchain_color_view: Option<&wgpu::TextureView>,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), Error> {
    let draw_list: Vec<usize> = if pass.instance_indices.is_empty() {
        (0..scene.instances.len()).collect()
    } else {
        pass.instance_indices
            .iter()
            .copied()
            .filter(|index| *index < scene.instances.len())
            .collect()
    };

    let pass_uses_depth = pass.uses_depth_attachment()
        || draw_list.iter().any(|index| {
            let instance = scene.instances[*index];
            bridge
                .resources
                .materials
                .get(&instance.material)
                .map(|material| material.enable_depth)
                .unwrap_or(false)
        });

    let color_target = resolve_color_target(bridge, pass, swapchain_format, swapchain_color_view)?;
    write_global_ubo(bridge, scene.camera);
    ensure_object_buffer_capacity(bridge, draw_list.len() as u32)?;
    batch_write_object_ubo(bridge, &draw_list, scene);

    let color_load = match pass.color_load {
        ColorAttachmentLoad::Clear => wgpu::LoadOp::Clear(clear_color_for_surface(
            color_target.format,
            pass.clear_color,
        )),
        ColorAttachmentLoad::Load => wgpu::LoadOp::Load,
    };

    let depth_attachment = if pass_uses_depth {
        let depth_view = ensure_depth_target(bridge, pass)?;
        let depth_load = effective_depth_load(pass, pass_uses_depth);
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: match depth_load {
                    DepthAttachmentLoad::Clear => wgpu::LoadOp::Clear(pass.clear_depth),
                    DepthAttachmentLoad::Load => wgpu::LoadOp::Load,
                    DepthAttachmentLoad::None => wgpu::LoadOp::Clear(1.0),
                },
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        })
    } else {
        None
    };

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("rotex-wgpu-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &color_target.view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: color_load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: depth_attachment,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });

    render_pass.set_bind_group(0, &bridge.global_bind_group, &[]);

    let mut last_material = None;
    for (slot, index) in draw_list.iter().enumerate() {
        let instance = scene.instances[*index];
        let material_bind_group = material_bind_group_for_instance(bridge, instance.material)?;
        if last_material != Some(instance.material) {
            render_pass.set_bind_group(1, material_bind_group, &[]);
            last_material = Some(instance.material);
        }

        let object_offset = slot as u32 * bridge.object_aligned_stride;
        render_pass.set_bind_group(2, &bridge.object_bind_group, &[object_offset]);

        let (
            vertex_layout_id,
            vertex_layout,
            vertex_buffer,
            index_buffer,
            index_format,
            index_count,
        ) = bridge
            .resources
            .meshes
            .get(&instance.mesh)
            .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("mesh")))?
            .to_draw_inputs();

        let pipeline = pipeline_cache::pipeline_for_draw(
            bridge,
            instance.material,
            vertex_layout_id,
            &vertex_layout,
            pass_uses_depth,
            color_target.format,
        )?;
        render_pass.set_pipeline(pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), index_format);
        render_pass.draw_indexed(0..index_count, 0, 0..1);
    }
    Ok(())
}

fn resolve_color_target(
    bridge: &WgpuBridge,
    pass: &rotex_types::PassDescriptor,
    swapchain_format: Option<wgpu::TextureFormat>,
    swapchain_view: Option<&wgpu::TextureView>,
) -> Result<ColorTargetView, Error> {
    match pass.color_target {
        PassColorTarget::Swapchain => {
            let format = swapchain_format.ok_or_else(super::surface_not_attached_error)?;
            let view = swapchain_view.ok_or_else(|| {
                Error::recoverable(ErrorKind::InvalidDescriptor("swapchain_view_missing"))
            })?;
            Ok(ColorTargetView {
                view: view.clone(),
                format,
            })
        }
        PassColorTarget::Texture(texture_id) => {
            let texture = bridge
                .resources
                .textures
                .get(&texture_id)
                .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("texture")))?;
            let render_view = texture.render_view.clone().ok_or_else(|| {
                Error::recoverable(ErrorKind::InvalidDescriptor("texture_missing_render_view"))
            })?;
            Ok(ColorTargetView {
                view: render_view,
                format: texture.format,
            })
        }
    }
}

fn ensure_depth_target<'a>(
    bridge: &'a mut WgpuBridge,
    pass: &rotex_types::PassDescriptor,
) -> Result<&'a wgpu::TextureView, Error> {
    let key = match pass.color_target {
        PassColorTarget::Swapchain => {
            let swapchain = bridge
                .swapchain
                .as_ref()
                .ok_or_else(super::surface_not_attached_error)?;
            DepthTargetKey::Swapchain {
                width: swapchain.config.width.max(1),
                height: swapchain.config.height.max(1),
            }
        }
        PassColorTarget::Texture(texture_id) => DepthTargetKey::Texture(texture_id),
    };

    let needs_recreate = bridge
        .depth_targets
        .get(&key)
        .map(|target| match key {
            DepthTargetKey::Swapchain { width, height } => target.size != (width, height),
            DepthTargetKey::Texture(_) => false,
        })
        .unwrap_or(true);

    if needs_recreate {
        let size = match key {
            DepthTargetKey::Swapchain { width, height } => (width, height),
            DepthTargetKey::Texture(texture_id) => {
                bridge
                    .resources
                    .textures
                    .get(&texture_id)
                    .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("texture")))?
                    .size
            }
        };
        let texture = bridge.device.raw.create_texture(&wgpu::TextureDescriptor {
            label: Some("rotex-wgpu-depth"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        bridge.depth_targets.insert(
            key,
            DepthTarget {
                _texture: texture,
                view,
                size,
            },
        );
    }

    Ok(&bridge.depth_targets.get(&key).expect("depth target").view)
}

fn effective_depth_load(
    pass: &rotex_types::PassDescriptor,
    uses_depth: bool,
) -> DepthAttachmentLoad {
    if !uses_depth {
        return DepthAttachmentLoad::None;
    }
    if pass.uses_depth_attachment() {
        pass.depth_load
    } else {
        DepthAttachmentLoad::Clear
    }
}

fn material_bind_group_for_instance<'a>(
    bridge: &'a WgpuBridge,
    material_id: rotex_types::MaterialId,
) -> Result<&'a wgpu::BindGroup, Error> {
    let texture_id = bridge
        .resources
        .materials
        .get(&material_id)
        .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("material")))?
        .texture;
    match texture_id {
        Some(texture_id) => bridge
            .resources
            .textures
            .get(&texture_id)
            .map(|texture| &texture.bind_group)
            .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("texture"))),
        None => Ok(&bridge.fallback_texture_bind_group),
    }
}

fn write_global_ubo(bridge: &WgpuBridge, camera: rotex_types::CameraDescriptor) {
    let mut bytes = vec![0_u8; GLOBAL_UBO_SIZE as usize];
    bytes[..64].copy_from_slice(unsafe {
        std::slice::from_raw_parts(camera.view.as_ptr() as *const u8, 64)
    });
    bytes[64..128].copy_from_slice(unsafe {
        std::slice::from_raw_parts(camera.projection.as_ptr() as *const u8, 64)
    });
    bridge
        .device
        .queue
        .write_buffer(&bridge.global_uniform_buffer, 0, &bytes);
}

fn ensure_object_buffer_capacity(bridge: &mut WgpuBridge, required: u32) -> Result<(), Error> {
    if required <= bridge.object_buffer_capacity {
        return Ok(());
    }
    let new_capacity = required
        .next_power_of_two()
        .max(bridge.object_buffer_capacity * 2);
    let new_size = bridge.object_aligned_stride as u64 * new_capacity as u64;
    bridge.object_uniform_buffer = bridge.device.raw.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rotex-wgpu-object-ubo"),
        size: new_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    bridge.object_bind_group = bridge
        .device
        .raw
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rotex-wgpu-object-bind-group"),
            layout: &bridge.object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &bridge.object_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(bridge.object_aligned_stride as u64),
                }),
            }],
        });
    bridge.object_buffer_capacity = new_capacity;
    Ok(())
}

fn batch_write_object_ubo(bridge: &WgpuBridge, draw_list: &[usize], scene: &SceneDescriptor) {
    let total_size = draw_list.len() as u64 * bridge.object_aligned_stride as u64;
    let mut bytes = vec![0_u8; total_size as usize];
    for (slot, index) in draw_list.iter().enumerate() {
        let transform = scene.instances[*index].transform;
        let offset = slot * bridge.object_aligned_stride as usize;
        bytes[offset..offset + OBJECT_MATRIX_SIZE as usize].copy_from_slice(unsafe {
            std::slice::from_raw_parts(transform.as_ptr() as *const u8, OBJECT_MATRIX_SIZE as usize)
        });
    }
    bridge
        .device
        .queue
        .write_buffer(&bridge.object_uniform_buffer, 0, &bytes);
}

fn to_wgpu_color(clear: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: clear[0] as f64,
        g: clear[1] as f64,
        b: clear[2] as f64,
        a: clear[3] as f64,
    }
}

fn clear_color_for_surface(format: wgpu::TextureFormat, clear: [f32; 4]) -> wgpu::Color {
    let converted = if format.is_srgb() {
        clear
    } else {
        [
            linear_to_srgb(clear[0]),
            linear_to_srgb(clear[1]),
            linear_to_srgb(clear[2]),
            clear[3],
        ]
    };
    to_wgpu_color(converted)
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
