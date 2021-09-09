use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::fmt;

use ash::{extensions::{ext::DebugUtils, khr::Swapchain, }, vk, Instance};

use crate::device::VulkanDevice;

pub struct VulkanBase {
    app_name: String,
    entry: ash::Entry,
    instance: Instance,
    device: VulkanDevice,
    debug_utils: Option<DebugUtils>,
    debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

pub struct VulkanBaseBuilder {
    app_name: Option<String>,
    layer_names: Vec<CString>,
    extension_names: Vec<CString>,
    enable_debug_utils: bool,
}

pub enum VulkanBaseBuilderError {
    AppNameAbsent,
}

impl VulkanBase {
    pub fn builder() -> VulkanBaseBuilder {
        VulkanBaseBuilder {
            app_name: None,
            layer_names: Vec::new(),
            extension_names: Vec::new(),
            enable_debug_utils: false,
        }
    }
}

impl VulkanBaseBuilder {
    pub fn name(mut self, name: String) -> Self {
        self.app_name = Some(name);
        return self;
    }

    pub fn with_layers(mut self, mut layer_names: Vec<CString>) -> Self {
        self.layer_names.append(&mut layer_names);
        return self;
    }

    pub fn with_extensions(mut self, mut ext_names: Vec<CString>) -> Self {
        self.extension_names.append(&mut ext_names);
        return self;
    }

    pub fn enable_debug_utils(mut self) -> Self {
        self.enable_debug_utils = true;
        self.layer_names
            .push(CString::new(DebugUtils::name().to_bytes_with_nul()).unwrap());
        return self;
    }

    pub fn build(mut self) -> Result<VulkanBase, VulkanBaseBuilderError> {
        let app_name = match self.app_name {
            Some(name) => {
                if name.is_empty() {
                    return Err(VulkanBaseBuilderError::AppNameAbsent);
                } else {
                    name
                }
            }
            None => return Err(VulkanBaseBuilderError::AppNameAbsent),
        };

        let app_name_cstr = CString::new(app_name.clone()).unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name_cstr)
            .application_version(0)
            .engine_name(&app_name_cstr)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let entry = unsafe { ash::Entry::new().unwrap() };

        // Check if requested layers are supported.
        let mut layer_names_ptr = Vec::new();
        let layer_props = entry.enumerate_instance_layer_properties().unwrap();
        for requested_layer_name in self.layer_names.iter() {
            let mut supported = false;

            for layer_prop in layer_props.iter() {
                let supported_layer_name = CStr::from_bytes_with_nul(unsafe {
                    &*(&layer_prop.layer_name as *const [i8] as *const [u8])
                })
                .unwrap();
                if supported_layer_name == &**requested_layer_name {
                    layer_names_ptr.push(requested_layer_name.as_ptr());
                    supported = true;
                    break;
                }
            }
            if !supported {
                println!(
                    "Requested layer ({:?}) is not supported.",
                    requested_layer_name
                );

                if &**requested_layer_name == DebugUtils::name() {
                    self.enable_debug_utils = false;
                }
            }
        }

        // Check if requested instance extensions are supported.
        let mut extension_names_ptr = Vec::new();
        let extension_props = entry.enumerate_instance_extension_properties().unwrap();
        for requested_ext_name in self.extension_names.iter() {
            let mut supported = false;
            for ext_prop in extension_props.iter() {
                let supported_ext_name = CStr::from_bytes_with_nul(unsafe {
                    &*(&ext_prop.extension_name as *const [i8] as *const [u8])
                })
                .unwrap();
                if supported_ext_name == &**requested_ext_name {
                    extension_names_ptr.push(requested_ext_name.as_ptr());
                    supported = true;
                    break;
                }
            }
            if !supported {
                println!(
                    "Requested extension ({:?}) is not supported.",
                    requested_ext_name
                );
            }
        }

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_ptr)
            .enabled_extension_names(&extension_names_ptr);

        let instance = unsafe { entry.create_instance(&instance_create_info, None).unwrap() };

        let (debug_utils, debug_utils_messenger) = if self.enable_debug_utils {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils = DebugUtils::new(&entry, &instance);
            let debug_utils_messenger = unsafe {
                debug_utils
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap()
            };
            (Some(debug_utils), Some(debug_utils_messenger))
        } else {
            (None, None)
        };

        // Create logical device.

        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };

        let device_extension_names = vec!(CString::from(Swapchain::name()));
        let device = VulkanDevice::builder(&instance, physical_devices[0])
            .with_extensions(device_extension_names)
            .with_queue_flags(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            .build();

        let swapchain = VulkanSwapchain::new(entry, instance, physical_devices, logical_device);

        return Ok(VulkanBase {
            app_name,
            entry,
            instance,
            device,
            debug_utils,
            debug_utils_messenger,
        });
    }
}

impl Drop for VulkanBase {
    fn drop(&mut self) {
        if let Some(debug_utils) = self.debug_utils.as_ref() {
            unsafe {
                debug_utils
                    .destroy_debug_utils_messenger(self.debug_utils_messenger.unwrap(), None);
            }
        }

        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

impl fmt::Debug for VulkanBaseBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            &Self::AppNameAbsent => f.write_str("No app name provided."),
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("null id")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("no message")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "[{:?}] [{:?}] ({}, {}) {}",
        message_severity, message_type, message_id_name, message_id_number, message
    );

    return vk::FALSE;
}
