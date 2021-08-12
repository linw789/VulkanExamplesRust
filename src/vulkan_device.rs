use std::os::raw::c_char;
use std::ffi::CStr;

use ash::{
    extensions::khr::Swapchain, 
    vk,
};

pub struct VulkanDevice {
    phys_device: vk::PhysicalDevice,
    // phys_device_props: vk::PhysicalDeviceProperties,
    // phys_device_features: vk::PhysicalDeviceFeatures,
    // phys_device_mem_props: vk::PhysicalDeviceMemoryProperties,
    phys_device_enabled_features: vk::PhysicalDeviceFeatures,

    // supported_extensions: Vec<*const c_char>,

    queue_family_indices: QueueFamilyIndices,
    // queue_family_props: Vec<vk::QueueFamilyProperties>,

    cmd_pool: vk::CommandPool,

    logical_device: vk::Device,
}

struct QueueFamilyIndices {
    graphics: u32,
    compute: u32,
    transfer: u32,
}

struct VulkanDeviceBuilder<'a, T: vk::ExtendsDeviceCreateInfo + 'a> {
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,

    enabled_features: vk::PhysicalDeviceFeatures,
    requested_extensions: Vec<[c_char; 256]>,
    requested_queue_types: vk::QueueFlags,

    next: Option<&'a T>,
}

impl VulkanDevice {
    pub fn builder(instance: vk::Instance, physical_device: vk::PhysicalDevice) -> VulkanDeviceBuilder {
        VulkanDeviceBuilder {
            instance: instance,
            physical_device: physical_device,

            enabled_features: vk::PhysicalDeviceFeatures::default(),
            requested_extensions: new Vec(),
            requested_queue_types: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,

            next: None,
        }
    }
}

impl Default for QueueFamilyIndices {
    pub fn default() -> Self {
        QueueFamilyIndices {
            0,
            0,
            0,
        }
    }
}

impl<'a, T: vk::ExtendsDeviceCreateInfo + 'a> VulkanDeviceBuilder<T> {
    pub fn with_features(mut self, enabled_features: vk::PhysicalDeviceFeatures) -> Self {
        self.enabled_features = enabled_features;
        return self;
    }

    pub fn with_extensions(mut self, extensions: Vec<*const c_char>) -> Self {
        self.requested_extensions = extensions;
        return self;
    }

    pub fn with_queue_flags(mut self, queue_flags: vk::QueueFlags) -> Self {
        self.requested_queue_types = queue_flags;
        return self;
    }

    pub fn push_next<T: vk::ExtendsDeviceCreateInfo>(mut self, next: &'a mut T) -> Self {
        self.next = Some(next);
        return self;
    }

    pub fn build(self) -> VulkanDevice {

        // Find command queues.

        let queue_family_indices = QueueFamilyIndices::default(); 
        let queue_create_infos: Vec::<vk::DeviceQueueCreateInfo> = Vec::new();

        let default_queue_priority = [0f32];
        let queue_family_props = unsafe { instance.get_physical_device_queue_family_properties(phys_device) };

        if requested_queue_types.contains(vk::QueueFlags::GRAPHICS) {
            queue_family_indices.graphics = get_queue_family_index(vk::QueueFlags::GRAPHICS, queue_family_props).unwrap();
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family_indices.graphics)
                    .queue_priorities(&default_queue_priority)
                    .build()
            );
        }

        if requested_queue_types.contains(vk::QueueFlags::COMPUTE) {
            // Try to create dedicated compute queue.
            queue_family_indices.compute = get_queue_family_index(vk::QueueFlags::COMPUTE, queue_family_props).unwrap();
            if queue_family_indices.compute != queue_family_indices.graphics {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(queue_family_indices.compute)
                        .queue_priorities(&default_queue_priority)
                        .build()
                );
            }
        } else {
            queue_family_indices.compute = queue_family_indices.graphics;
        }

        if requested_queue_types.contains(vk::QueueFlags::TRANSFER) {
            // Try to create dedicated transfer queue.
            queue_family_indices.transfer = get_queue_family_index(vk::QueueFlags::TRANSFER, queue_family_props).unwrap();
            if (queue_family_indices.transfer != queue_family_indices.graphics) && 
               (queue_family_indices.transfer != queue_family_indices.compute) {

                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(queue_family_indices.transfer)
                        .queue_priorities(&default_queue_priority)
                        .build()
                );
            }
        } else {
            queue_family_indices.transfer = queue_family_indices.graphics;
        }


        // Check is requested extensions are supported on this device.
        let supported_extensions = unsafe { self.instance.enumerate_device_extension_properties(self.physical_device).unwrap() };
        for requested_ext in self.requested_extensions.iter() {
            let mut supported = false;
            let requested_ext_cstr = CStr::from_bytes_with_nul(requested_ext).unwrap();

            for ext_prop in supported_extensions.iter() {
                let supported_ext_cstr = CStr::from_bytes_with_nul(ext_prop.extension_name).unwrap();
                if requested_ext_cstr == supported_ext_cstr {
                    supported = true;
                    break;
                }
            }

            if supported == false {
                println!("Requested extension ({?}) is not supported.", requested_ext);
            }
        }

        let logical_device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&self.requested_extensions)
            .enabled_features(&self.enabled_features);

        if let Some(next) = self.next {
            logical_device_create_info.push_next(next);
        }

        let logical_device = unsafe { instance.create_device(physical_device, &logical_device_create_info, None).unwrap() };

        let cmd_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics);
        
        let cmd_pool = unsafe { logical_device.create_command_pool(&cmd_pool_create_info, None).unwrap() };

        VulkanDevice {
            phys_device: physical_device,
            phys_device_enabled_features: self.enabled_features,
            queue_family_indices: queue_family_indices,
            cmd_pool: cmd_pool,
            logical_device: logical_device,
        }
    }
}

fn get_queue_family_index(
    request_queue_flags: vk::QueueFlags, 
    queue_family_props: &Vec<vk::QueueFamilyProperties>) -> Option<u32>
{
    // If requesting compute queue, try to find a queue family that supports compute but not
    // graphics.
    if request_queue_flags.contains(vk::QueueFlags::COMPUTE) {
        for (i, prop) in queue_family_props.enumerate() {
            if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                return Some(i as u32);
            }
        }
    }

    // If requesting transfer queue, try to find a queue family that supports transfer but not
    // graphics nor compute.
    if request_queue_flags.contains(vk::QueueFlags::TRANSFER) {
        for (i, prop) in queue_family_props.enumerate() {
            if prop.queue_flags.contains(vk::QueueFlags::TRANSFER) 
                && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) 
                && !prop.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Some(i as u32);
            }
        }
    }

    // For other queue types or if no dedicated queue is present, return the first qualified one.
    for (i, prop) in queue_family_props.enumerate() {
        if prop.queue_flags.contains(request_queue_flags) { 
            return Some(i as u32);
        }
    }

    return None;
}

