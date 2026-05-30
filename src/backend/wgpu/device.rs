use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use wgpu::{
    DeviceDescriptor as WgpuDeviceDescriptor, InstanceDescriptor as WgpuInstanceDescriptor,
    PowerPreference, RequestAdapterOptions, SurfaceTargetUnsafe,
};

use super::surface::WgpuSurface;
use crate::error::{Error, ErrorKind};
use rotex_types::{
    DeviceDescriptor as RotexDeviceDescriptor, InstanceDescriptor as RotexInstanceDescriptor,
    QueueCategory,
};

pub struct WgpuInstance {
    raw: wgpu::Instance,
}

pub struct WgpuDevice {
    pub adapter: wgpu::Adapter,
    pub raw: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuInstance {
    pub async fn new(descriptor: &RotexInstanceDescriptor) -> Result<Self, Error> {
        if !descriptor.required_instance_extensions.is_empty() {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "required_instance_extensions_not_supported",
            )));
        }
        Ok(Self {
            raw: wgpu::Instance::new(WgpuInstanceDescriptor::new_without_display_handle()),
        })
    }

    pub async fn request_device(
        &self,
        descriptor: &RotexDeviceDescriptor,
    ) -> Result<WgpuDevice, Error> {
        if !descriptor.enable_swapchain {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "disable_swapchain_not_supported",
            )));
        }
        if descriptor.required_features.sampler_anisotropy {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "sampler_anisotropy_not_supported",
            )));
        }
        if descriptor.required_features.wide_lines {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "wide_lines_not_supported",
            )));
        }
        if !descriptor
            .queues
            .iter()
            .any(|request| request.category == QueueCategory::Graphics && request.count > 0)
        {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "graphics_queue_required",
            )));
        }

        let adapter = self
            .raw
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| Error::fatal(ErrorKind::NoCompatibleDevice))?;

        let mut requested_features = wgpu::Features::empty();
        if descriptor.required_features.fill_mode_non_solid {
            requested_features.insert(wgpu::Features::POLYGON_MODE_LINE);
        }
        if !adapter.features().contains(requested_features) {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "required_device_features_unavailable",
            )));
        }

        let mut wgpu_descriptor = WgpuDeviceDescriptor::default();
        wgpu_descriptor.required_features = requested_features;
        let (raw, queue) = adapter
            .request_device(&wgpu_descriptor)
            .await
            .map_err(|_| Error::fatal(ErrorKind::NoCompatibleDevice))?;

        Ok(WgpuDevice {
            adapter,
            raw,
            queue,
        })
    }

    pub fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<WgpuSurface, Error> {
        let surface = unsafe {
            self.raw
                .create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: Some(display_handle),
                    raw_window_handle: window_handle,
                })
        }
        .map_err(|_| Error::fatal(ErrorKind::Backend("surface_create_failed")))?;

        Ok(WgpuSurface::new(surface))
    }
}
