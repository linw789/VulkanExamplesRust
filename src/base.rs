use std::ffi::CString;
use std::os::raw::c_char;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

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
    window: Window,
    event_loop: EventLoop,

    instance: Instance,
}

impl VulkanBase {
    pub fn new(&mut self, name: String, width: u32, height: u32) {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(name)
            .with_inner_size(winit::dpi::LogicalSize::new(
                    width as f64,
                    height as f64
            ))
            .build(&event_loop)
            .unwrap();

        // Create Vulkan instance.

        let app_name = CString::new(name).unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let layer_names = [CString::new("vk_layer_khronos_validation").unwrap()];
        let layer_names_ptr: Vec<*const c_char> = layer_names.iter().map(|name| name.as_ptr()).collect();

        let surface_extensions = ash_window::enumerate_required_extensions(&window).unwrap();
        let surface_extensions_ptr: Vec<*const c_char> = surface_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let surface_extensions_ptr.push(DebugUtils::name().as_ptr());
        println!("surface extensions: {:?}, {:?}", DebugUtil::name(), surface_extensions);

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_ptr)
            .enabled_extension_names(&surface_extensions_ptr);

        let vulkan_entry = unsafe { Entry::new().unwrap() };
        let instance: Instance = unsafe {
            entry
                .create_instance(&instance_create_info, None).unwrap();
        };

        let debug_util_createinfo = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR |
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            )
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(None); // TODO

        let debug_utils_loader = DebugUtils::new(&entry, &instance);
        let debug_callback = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_util_createinfo, None)
                .unwrap()
        };

        // Find a physical device.

        let physical_devices = unsafe {
            instance.enumerate_physical_devices().unwrap()
        };

        println!("Available Vulkan devices:");
        for (i, phys_device) in physical_devices.enumerate() {
            let phys_device_prop = unsafe {
                instance.get_physical_device_properties(*phys_device)
            };
            println!("Device [{}]: {}\n  Type: {}\n  API:{}.{}.{}", 
                 i, 
                 phys_device_prop.device_name, 
                 phys_device_prop.device_type,
                 phys_device_prop.api_version >> 22,
                 (phys_device_prop.api_version >> 12) & 0x3ff,
                 phys_device_prop.api_version & 0xfff
            );
        }

        let phys_device = physical_devices.remove(0);
        let phys_device_prop = instance.get_physical_device_properties(phys_device);
        let phys_device_features = instance.get_physical_device_features(phys_device);
        let phys_device_mem_prop = instance.get_physical_device_memory_properties(phys_device);

        // Create a surface.

        let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None).unwrap() };
        let surface_loader = Surface::new(&entry, &instance);

    }
}
