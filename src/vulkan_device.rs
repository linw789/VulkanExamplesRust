use std::os::raw::c_char;
use ash::vk;

pub struct VulkanDevice {
    phys_device: vk::PhysicalDevice,
    phys_device_prop: vk::PhysicalDeviceProperties,
    phys_device_mem_prop: vk::PhysicalDeviceMemoryProperties,
    phys_device_features: vk::PhysicalDeviceFeatures,
    phys_device_enabled_features: vk::PhysicalDeviceFeatures,

    logical_device: vk::Device,

    queue_family_prop: Vec<vk::QueueFamilyProperties>,
    supported_extensions: Vec<c_char>,
    cmd_pool: vk::CommandPool,

    queue_family_indices: QueueFamilyIndices,

    enable_debug_marker: bool,
}

struct QueueFamilyIndices {
    graphics: u32,
    compute: u32,
    transfer: u32,
}

impl VulkanDevice {
    pub fn new(phyical_device: vk::PhysicalDevice, instance: vk::Instance) -> Self {
        let phys_device_prop = 


    }
}

