use std::ffi::CString;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

extern crate vulkan_base;
use vulkan_base::vulkan_base::VulkanBase;

fn main() {
    const WINDOW_WIDTH: u32 = 800;
    const WINDOW_HEIGHT: u32 = 800;

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Vulkan Test")
        .with_inner_size(winit::dpi::LogicalSize::new(
            WINDOW_WIDTH as f64,
            WINDOW_HEIGHT as f64,
        ))
        .build(&event_loop)
        .unwrap();

    let layer_names = vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()];

    let surface_extensions = ash_window::enumerate_required_extensions(&window).unwrap();
    let extension_names = surface_extensions
        .iter()
        .map(|&name| CString::from(name))
        .collect::<Vec<_>>();

    let vulkan_base = VulkanBase::builder()
        .name(String::from("cube"))
        .with_layers(layer_names)
        .with_extensions(extension_names)
        .enable_debug_utils()
        .build()
        .unwrap();

    std::mem::drop(vulkan_base);
}
