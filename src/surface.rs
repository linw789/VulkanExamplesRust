use ash::prelude::*;
use ash::{extensions::khr, vk, Entry, Instance};
use winit::window::Window;

pub struct Surface {
    surface_handle: vk::SurfaceKHR,
    surface: khr::Surface,
}

impl Surface {
    pub fn new(entry: &Entry, instance: &Instance, window: &Window) -> VkResult<Self> {
        let surface_handle = unsafe { ash_window::create_surface(entry, instance, window, None)? };

        Ok(Surface {
            surface_handle,
            surface: khr::Surface::new(entry, instance),
        })
    }

    pub fn destroy(&self) {
        unsafe { self.surface.destroy_surface(self.surface_handle, None) }
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface_handle
    }

    /// Check whether a queue family of a _physical device_ supports presentation
    /// to this surface.
    pub fn support_present(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_index: u32,
    ) -> VkResult<bool> {
        unsafe {
            self.surface.get_physical_device_surface_support(
                physical_device,
                queue_index,
                self.surface_handle,
            )
        }
    }

    pub fn formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            self.surface
                .get_physical_device_surface_formats(physical_device, self.surface_handle)
        }
    }

    pub fn capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            self.surface
                .get_physical_device_surface_capabilities(physical_device, self.surface_handle)
        }
    }

    pub fn present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            self.surface
                .get_physical_device_surface_present_modes(physical_device, self.surface_handle)
        }
    }
}
