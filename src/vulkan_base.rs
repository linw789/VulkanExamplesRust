use std::ffi::CString;
use std::os::raw::c_char;

use ash::{
    vk,
    Entry,
    Device, 
    Instance,
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
};

struct VulkanBase {
    app_name: String,
    instance: Instance,
    device: VulkanDevice,
}

struct VulkanBaseBuilder {
    app_name: String,
    layer_names: Option<Vec<CString>>,
    extension_names: Options<Vec<CString>>,
    enable_debug_utils: bool,
}

impl VulkanBase {
    pub fn build(name: String) -> VulkanBaseBuilder {
        VulkanBaseBuilder {
            app_name: String::new(),
            layer_names: None,
            extension_names: None,
            enable_debug_utils: false,
        }
    }
}

impl VulkanBaseBuilder {
    pub fn name(mut self, name: String) -> Self {
        self.inner.app_name = name;
        return self;
    }

    pub fn with_layers(mut self, layer_names: Vec<CString>) -> Self {
        self.layer_names = Some(layer_names);
        return self;
    }

    pub fn with_extensions(mut self, ext_names: Vec<CString>) -> Self {
        self.extension_names = Some(ext_names);
        return self;
    }

    pub fn enable_debug_utils(mut self) -> Self {
        self.enable_debug_utils = true;
        self.extension_names.push(DebugUtils::name());
        return self;
    }

    pub fn build(self) -> VulkanBase {
        let app_name_cstr = CString::new(self.app_name).unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name_cstr)
            .application_version(0)
            .engine_name(&app_name_cstr)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let layer_names_ptr: Vec<*const c_char> = self.layer_names.iter().map(|name| name.as_ptr()).collect();
        let ext_names_ptr: Vec<*const c_char> = self.extension_names.iter().map(|name| name.as_ptr()).collect();

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_ptr)
            .enabled_extension_names(&ext_names_ptr);

        
    }
}

