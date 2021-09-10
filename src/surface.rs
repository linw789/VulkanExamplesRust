use ash::{extensions::khr, prelude::*, vk, Entry, Instance};

use winit::window::Window;

pub struct Surface {
    surface: vk::SurfaceKHR,
    surface_pfns: khr::Surface,
    physical_device: vk::PhysicalDevice,
}

impl Surface {
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        window: &Window,
    ) -> Self {
        let surface = Surface {
            surface: unsafe { ash_window::create_surface(entry, instance, window, None).unwrap() },
            surface_pfns: khr::Surface::new(entry, instance),
            physical_device,
        };
    }

    pub fn vk_surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn support_present(&self, queue_index: u32) -> VkResult<bool> {
        unsafe {
            self.surface_pfns.get_physical_device_surface_support(
                self.physical_device,
                queue_index,
                self.surface,
            )
        }
    }

    pub fn formats(&self) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            self.surface_pfns
                .get_physical_device_surface_formats(self.physical_device, self.surface)
        }
    }

    pub fn capabilities(&self) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            self.surface_pfns
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
    }

    pub fn present_modes(&self) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            self.surface_pfns
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
        }
    }
}
