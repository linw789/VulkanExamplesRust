use std::ffi::CStr;
use std::ffi::CString;

use ash::{vk, Instance};

pub struct Device {
    physical_device: vk::PhysicalDevice,
    // physical_device_props: vk::PhysicalDeviceProperties,
    // physical_device_features: vk::PhysicalDeviceFeatures,
    // physical_device_mem_props: vk::PhysicalDeviceMemoryProperties,
    // physical_device_enabled_features: vk::PhysicalDeviceFeatures,

    // supported_extensions: Vec<*const c_char>,
    device: ash::Device,

    queue_family_props: Vec<vk::QueueFamilyProperties>,
    queue_family_indices: QueueFamilyIndices,

    cmd_pool: vk::CommandPool,
}

struct QueueFamilyIndices {
    graphics: u32,
    compute: u32,
    transfer: u32,
}

pub struct DeviceBuilder<'a> {
    instance: &'a Instance,
    physical_device: vk::PhysicalDevice,

    enabled_features: vk::PhysicalDeviceFeatures,
    requested_extensions: Vec<CString>,
    requested_queue_types: vk::QueueFlags,
    // next: Option<&'a T>,
}

impl Device {
    pub fn builder<'a>(
        instance: &'a Instance,
        physical_device: vk::PhysicalDevice,
    ) -> DeviceBuilder<'a> {
        DeviceBuilder {
            instance: instance,
            physical_device: physical_device,

            enabled_features: vk::PhysicalDeviceFeatures::default(),
            requested_extensions: Vec::new(),
            requested_queue_types: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            // next: None,
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn queue_family_properties(&self) -> &[vk::QueueFamilyProperties] {
        &self.queue_family_props
    }
}

impl Default for QueueFamilyIndices {
    fn default() -> Self {
        QueueFamilyIndices {
            graphics: 0,
            compute: 0,
            transfer: 0,
        }
    }
}

impl<'a> DeviceBuilder<'a> {
    pub fn with_features(mut self, enabled_features: vk::PhysicalDeviceFeatures) -> Self {
        self.enabled_features = enabled_features;
        return self;
    }

    pub fn with_extensions(mut self, mut extensions: Vec<CString>) -> Self {
        self.requested_extensions.append(&mut extensions);
        return self;
    }

    pub fn with_queue_flags(mut self, queue_flags: vk::QueueFlags) -> Self {
        self.requested_queue_types = queue_flags;
        return self;
    }

    /*
    pub fn push_next<T: vk::ExtendsDeviceCreateInfo>(mut self, next: &'a mut T) -> Self {
        self.next = Some(next);
        return self;
    }
    */

    pub fn build(self) -> Device {
        // Find command queues.

        let mut queue_family_indices = QueueFamilyIndices::default();
        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();

        let default_queue_priority = [0f32];
        let queue_family_props = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(self.physical_device)
        };

        if self
            .requested_queue_types
            .contains(vk::QueueFlags::GRAPHICS)
        {
            queue_family_indices.graphics =
                get_queue_family_index(vk::QueueFlags::GRAPHICS, &queue_family_props).unwrap();
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family_indices.graphics)
                    .queue_priorities(&default_queue_priority)
                    .build(),
            );
        }

        if self.requested_queue_types.contains(vk::QueueFlags::COMPUTE) {
            // Try to create dedicated compute queue.
            queue_family_indices.compute =
                get_queue_family_index(vk::QueueFlags::COMPUTE, &queue_family_props).unwrap();
            if queue_family_indices.compute != queue_family_indices.graphics {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(queue_family_indices.compute)
                        .queue_priorities(&default_queue_priority)
                        .build(),
                );
            }
        } else {
            queue_family_indices.compute = queue_family_indices.graphics;
        }

        if self
            .requested_queue_types
            .contains(vk::QueueFlags::TRANSFER)
        {
            // Try to create dedicated transfer queue.
            queue_family_indices.transfer =
                get_queue_family_index(vk::QueueFlags::TRANSFER, &queue_family_props).unwrap();
            if (queue_family_indices.transfer != queue_family_indices.graphics)
                && (queue_family_indices.transfer != queue_family_indices.compute)
            {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(queue_family_indices.transfer)
                        .queue_priorities(&default_queue_priority)
                        .build(),
                );
            }
        } else {
            queue_family_indices.transfer = queue_family_indices.graphics;
        }

        // Check if requested extensions are supported on this device.

        let mut extension_names_ptr = Vec::new();
        let supported_extensions = unsafe {
            self.instance
                .enumerate_device_extension_properties(self.physical_device)
                .unwrap()
        };
        for requested_ext in self.requested_extensions.iter() {
            let mut supported = false;

            for ext_prop in supported_extensions.iter() {
                let supported_ext_cstr = CStr::from_bytes_with_nul(unsafe {
                    &*(&ext_prop.extension_name as *const [i8] as *const [u8])
                })
                .unwrap();
                if &**requested_ext == supported_ext_cstr {
                    extension_names_ptr.push(requested_ext.as_ptr());
                    supported = true;
                    break;
                }
            }

            if !supported {
                println!(
                    "Requested extension ({:?}) is not supported.",
                    requested_ext
                );
            }
        }

        let logical_device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names_ptr)
            .enabled_features(&self.enabled_features);

        let logical_device = unsafe {
            self.instance
                .create_device(self.physical_device, &logical_device_create_info, None)
                .unwrap()
        };

        let cmd_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics);

        let cmd_pool = unsafe {
            logical_device
                .create_command_pool(&cmd_pool_create_info, None)
                .unwrap()
        };

        Device {
            physical_device: self.physical_device,
            device: logical_device,
            queue_family_props,
            queue_family_indices,
            cmd_pool: cmd_pool,
        }
    }
}

fn get_queue_family_index(
    request_queue_flags: vk::QueueFlags,
    queue_family_props: &Vec<vk::QueueFamilyProperties>,
) -> Option<u32> {
    // If requesting compute queue, try to find a queue family that supports compute but not
    // graphics.
    if request_queue_flags.contains(vk::QueueFlags::COMPUTE) {
        for (i, prop) in queue_family_props.iter().enumerate() {
            if prop.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                return Some(i as u32);
            }
        }
    }

    // If requesting transfer queue, try to find a queue family that supports transfer but not
    // graphics nor compute.
    if request_queue_flags.contains(vk::QueueFlags::TRANSFER) {
        for (i, prop) in queue_family_props.iter().enumerate() {
            if prop.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && !prop.queue_flags.contains(vk::QueueFlags::COMPUTE)
            {
                return Some(i as u32);
            }
        }
    }

    // For other queue types or if no dedicated queue is present, return the first qualified one.
    for (i, prop) in queue_family_props.iter().enumerate() {
        if prop.queue_flags.contains(request_queue_flags) {
            return Some(i as u32);
        }
    }

    return None;
}
