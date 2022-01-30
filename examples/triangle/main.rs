use ash::extensions::{ext::DebugUtils, khr::Swapchain};
use ash::util::*;
use ash::{vk, Entry};
pub use ash::{Device, Instance};
use std::borrow::Cow;
use std::default::Default;
use std::ffi::{CStr, CString};
use std::io::Cursor;
use std::vec::Vec;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

extern crate vulkan_examples;
use vulkan_examples::surface::Surface;

macro_rules! offset_of {
    ($base:path, $field:ident) => {
        unsafe {
            let b: $base = std::mem::zeroed();
            (&b.$field as *const _ as isize) - (&b as *const _ as isize)
        }
    };
}

#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}

fn main() {
    const WINDOW_WIDTH: u32 = 800;
    const WINDOW_HEIGHT: u32 = 800;

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Hello Triangle")
        .with_inner_size(winit::dpi::LogicalSize::new(
            WINDOW_WIDTH as f64,
            WINDOW_HEIGHT as f64,
        ))
        .build(&event_loop)
        .unwrap();

    let app_name = CString::new("Triangle").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
    let layer_names_raw: Vec<*const i8> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let mut surface_extensions = ash_window::enumerate_required_extensions(&window).unwrap();
    surface_extensions.push(DebugUtils::name());
    let extension_names_raw = surface_extensions
        .iter()
        .map(|ext_name| ext_name.as_ptr())
        .collect::<Vec<_>>();
    // println!("surface extensions: {:?}", surface_extensions);

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names_raw)
        .enabled_extension_names(&extension_names_raw);

    let entry = Entry::linked();

    let instance: Instance = unsafe {
        entry
            .create_instance(&instance_create_info, None)
            .expect("Vulkan instance creation failed.")
    };

    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_loader = DebugUtils::new(&entry, &instance);
    let debug_callback = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap()
    };

    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Physical device error.")
    };

    /*
    for physical_device in &physical_devices {
        let physical_device_prop =
            unsafe { instance.get_physical_device_properties(*physical_device) };
        println!("physical device properties: {:?}\n", physical_device_prop);
    }
    */

    let surface = Surface::new(&entry, instance.clone(), &window).unwrap();

    // Find the first physical device that contains a queue family that supports graphics
    // queue as well as presentation to a given surface, also return the index of the
    // qualified queue family.
    let (physical_device, queue_family_index) = physical_devices
        .iter()
        .map(|&device| unsafe {
            instance
                .get_physical_device_queue_family_properties(device)
                .iter()
                .enumerate()
                .find_map(|(index, ref queue_info)| {
                    let supports_graphics_and_surface =
                        queue_info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && surface.support_present(device, index as u32).unwrap();
                    if supports_graphics_and_surface {
                        Some((device, index as u32))
                    } else {
                        None
                    }
                })
        })
        .find_map(|v| v)
        .expect("Couldn't find suitable physical device.");

    let priorities = [1.0];
    let queue_info = [vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)
        .build()];

    let device_extension_names_raw = [Swapchain::name().as_ptr()];

    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        ..Default::default()
    };

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&device_extension_names_raw)
        .enabled_features(&features);

    let device: Device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap()
    };

    let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let supported_surface_formats = surface.formats(physical_device).unwrap();
    let surface_format = supported_surface_formats[0];

    let surface_capabilities = surface.capabilities(physical_device).unwrap();
    println!("Surface capabilities: {:?}\n", surface_capabilities);

    let mut swapchain_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0
        && swapchain_image_count > surface_capabilities.max_image_count
    {
        swapchain_image_count = surface_capabilities.max_image_count;
    }

    // If the width (and height) equals the special value 0xffffffff, the size of the
    // surface can be set by the swapchain. Otherwise the size of the swapchain is
    // determined by that of the surface.
    let surface_resolution = match surface_capabilities.current_extent.width {
        std::u32::MAX => vk::Extent2D {
            width: WINDOW_WIDTH,
            height: WINDOW_HEIGHT,
        },
        _ => surface_capabilities.current_extent,
    };

    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let present_modes = surface.present_modes(physical_device).unwrap();

    let present_mode = present_modes
        .iter()
        .copied()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        // VK_PRESENT_MODE_FIFO_KHR must be supported per the specification.
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let swapchain = Swapchain::new(&instance, &device);
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface.handle())
        .min_image_count(swapchain_image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);

    let swapchain_handle = unsafe {
        swapchain
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    };

    let cmd_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);
    let cmd_pool = unsafe {
        device
            .create_command_pool(&cmd_pool_create_info, None)
            .unwrap()
    };

    let cmd_buf_alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(2)
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY);
    let cmd_buffers = unsafe {
        device
            .allocate_command_buffers(&cmd_buf_alloc_info)
            .unwrap()
    };

    let setup_cmd_buf = cmd_buffers[0];
    let draw_cmd_buf = cmd_buffers[1];

    let present_images = unsafe { swapchain.get_swapchain_images(swapchain_handle).unwrap() };
    let present_image_views: Vec<vk::ImageView> = present_images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);
            unsafe { device.create_image_view(&create_view_info, None).unwrap() }
        })
        .collect();

    let depth_image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::D16_UNORM)
        .extent(vk::Extent3D {
            width: surface_resolution.width,
            height: surface_resolution.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // What is depth image for?
    let depth_image = unsafe { device.create_image(&depth_image_create_info, None).unwrap() };
    let depth_image_memory_req = unsafe { device.get_image_memory_requirements(depth_image) };

    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let depth_image_memeory_index = find_memory_type_index(
        &depth_image_memory_req,
        &device_memory_properties,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .expect("Unable to find suitable memory index for depth image.");

    let depth_image_alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(depth_image_memory_req.size)
        .memory_type_index(depth_image_memeory_index);
    let depth_image_memory = unsafe {
        device
            .allocate_memory(&depth_image_alloc_info, None)
            .unwrap()
    };

    unsafe {
        device
            .bind_image_memory(depth_image, depth_image_memory, 0)
            .expect("Unable to bind image memory");
    };

    let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    let draw_cmds_reuse_fence = unsafe {
        device
            .create_fence(&fence_create_info, None)
            .expect("Fence creation failed.")
    };
    let setup_cmds_reuse_fence = unsafe {
        device
            .create_fence(&fence_create_info, None)
            .expect("Fence creation failed.")
    };

    record_submit_commandbuffer(
        &device,
        setup_cmd_buf,
        setup_cmds_reuse_fence,
        present_queue,
        &[],
        &[],
        &[],
        |device, setup_command_buffer| {
            let layout_transition_barrier = vk::ImageMemoryBarrier::builder()
                .image(depth_image)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .layer_count(1)
                        .level_count(1)
                        .build(),
                );

            unsafe {
                device.cmd_pipeline_barrier(
                    setup_command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[layout_transition_barrier.build()],
                );
            }
        },
    );

    let depth_image_view_info = vk::ImageViewCreateInfo::builder()
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .layer_count(1)
                .level_count(1)
                .build(),
        )
        .image(depth_image)
        .format(depth_image_create_info.format)
        .view_type(vk::ImageViewType::TYPE_2D);

    let depth_image_view = unsafe {
        device
            .create_image_view(&depth_image_view_info, None)
            .unwrap()
    };

    let semaphore_create_info = vk::SemaphoreCreateInfo::default();

    let present_complete_semaphore = unsafe {
        device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap()
    };
    let rendering_complete_semaphore = unsafe {
        device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap()
    };

    let renderpass_attachments = [
        vk::AttachmentDescription {
            format: surface_format.format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        },
        vk::AttachmentDescription {
            format: vk::Format::D16_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        },
    ];

    let color_attachment_ref = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];

    let subpasses = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_ref)
        .depth_stencil_attachment(&depth_attachment_ref)
        .build()];

    let renderpass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&renderpass_attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let renderpass = unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    };

    let framebuffers: Vec<vk::Framebuffer> = present_image_views
        .iter()
        .map(|&present_image_view| {
            let framebuffer_attachments = [present_image_view, depth_image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(&framebuffer_attachments)
                .width(surface_resolution.width)
                .height(surface_resolution.height)
                .layers(1);
            unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .unwrap()
            }
        })
        .collect();

    let index_data: [u32; 3] = [0, 1, 2];

    let index_buf_create_info = vk::BufferCreateInfo::builder()
        .size(std::mem::size_of_val(&index_data) as u64)
        .usage(vk::BufferUsageFlags::INDEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let index_buf = unsafe { device.create_buffer(&index_buf_create_info, None).unwrap() };
    let index_buf_memory_req = unsafe { device.get_buffer_memory_requirements(index_buf) };
    let index_buf_memory_index = find_memory_type_index(
        &index_buf_memory_req,
        &device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .expect("Failed to find suitable memory for index buffer.");

    let index_buf_alloc_info = vk::MemoryAllocateInfo {
        allocation_size: index_buf_memory_req.size,
        memory_type_index: index_buf_memory_index,
        ..Default::default()
    };
    let index_buf_memory = unsafe { device.allocate_memory(&index_buf_alloc_info, None).unwrap() };
    let index_buf_ptr = unsafe {
        device
            .map_memory(
                index_buf_memory,
                0,
                index_buf_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap()
    };

    let mut index_buf_slice = unsafe {
        Align::new(
            index_buf_ptr,
            std::mem::align_of::<u32>() as u64,
            index_buf_memory_req.size,
        )
    };
    index_buf_slice.copy_from_slice(&index_data);
    unsafe { device.unmap_memory(index_buf_memory) };
    unsafe {
        device
            .bind_buffer_memory(index_buf, index_buf_memory, 0)
            .unwrap()
    };

    let vertices = [
        Vertex {
            pos: [0.0, -0.5, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            pos: [0.5, 0.5, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            pos: [-0.5, 0.5, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    let vert_input_buf_create_info = vk::BufferCreateInfo {
        size: std::mem::size_of_val(&vertices) as u64,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let vert_input_buf = unsafe {
        device
            .create_buffer(&vert_input_buf_create_info, None)
            .unwrap()
    };

    let vert_input_buf_memory_req =
        unsafe { device.get_buffer_memory_requirements(vert_input_buf) };
    let vert_input_buf_memory_index = find_memory_type_index(
        &vert_input_buf_memory_req,
        &device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .expect("Unable to find suitable memory index for vertex buffer.");

    let vert_buf_alloc_info = vk::MemoryAllocateInfo {
        allocation_size: vert_input_buf_memory_req.size,
        memory_type_index: vert_input_buf_memory_index,
        ..Default::default()
    };
    let vert_input_buf_memory =
        unsafe { device.allocate_memory(&vert_buf_alloc_info, None).unwrap() };

    let vert_buf_ptr = unsafe {
        device
            .map_memory(
                vert_input_buf_memory,
                0,
                vert_input_buf_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap()
    };

    let mut vert_buf_slice = unsafe {
        Align::new(
            vert_buf_ptr,
            std::mem::align_of::<Vertex>() as u64,
            vert_input_buf_memory_req.size,
        )
    };
    vert_buf_slice.copy_from_slice(&vertices);
    unsafe { device.unmap_memory(vert_input_buf_memory) };
    unsafe {
        device
            .bind_buffer_memory(vert_input_buf, vert_input_buf_memory, 0)
            .unwrap()
    };

    let mut vert_spv_file = Cursor::new(include_bytes!("vert.spv"));
    let mut frag_spv_file = Cursor::new(include_bytes!("frag.spv"));

    let vert_code = read_spv(&mut vert_spv_file).expect("Failed to read vertex shader spv file.");
    let vert_shader_info = vk::ShaderModuleCreateInfo::builder().code(&vert_code);

    let frag_code = read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file.");
    let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);

    let vert_shader_module = unsafe {
        device
            .create_shader_module(&vert_shader_info, None)
            .expect("Vertex shader module error.")
    };

    let frag_shader_module = unsafe {
        device
            .create_shader_module(&frag_shader_info, None)
            .expect("Fragment shader module error.")
    };

    let layout_create_info = vk::PipelineLayoutCreateInfo::default();
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap()
    };

    let shader_entry_name = CString::new("main").unwrap();
    let shader_stage_create_infos = [
        vk::PipelineShaderStageCreateInfo {
            module: vert_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            module: frag_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let vert_input_binding_descs = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let vert_input_attribute_descs = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Vertex, pos) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Vertex, color) as u32,
        },
    ];

    let vert_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&vert_input_attribute_descs)
        .vertex_binding_descriptions(&vert_input_binding_descs)
        .build();

    let vertex_input_assembly_state_create_info =
        vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .build();

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: surface_resolution.width as f32,
        height: surface_resolution.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: surface_resolution,
    }];

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
        .scissors(&scissors)
        .viewports(&viewports)
        .build();

    let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo {
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        ..Default::default()
    };

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };

    let noop_stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        ..Default::default()
    };
    let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable: 1,
        depth_write_enable: 1,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        front: noop_stencil_state,
        back: noop_stencil_state,
        max_depth_bounds: 1.0,
        ..Default::default()
    };
    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: 0,
        src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ZERO,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    }];
    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op(vk::LogicOp::CLEAR)
        .attachments(&color_blend_attachment_states);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_create_info =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vert_input_state_create_info)
        .input_assembly_state(&vertex_input_assembly_state_create_info)
        .viewport_state(&viewport_state_create_info)
        .rasterization_state(&rasterization_create_info)
        .multisample_state(&multisample_state_create_info)
        .depth_stencil_state(&depth_state_create_info)
        .color_blend_state(&color_blend_state_create_info)
        .dynamic_state(&dynamic_state_create_info)
        .layout(pipeline_layout)
        .render_pass(renderpass);

    let graphics_pipelines = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &[graphics_pipeline_create_info.build()],
            None,
        )
    }
    .expect("Unable to create graphics pipeline");

    let graphics_pipeline = graphics_pipelines[0];

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                unsafe {
                    device.device_wait_idle().unwrap();
                    for &pipeline in graphics_pipelines.iter() {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_shader_module(vert_shader_module, None);
                    device.destroy_shader_module(frag_shader_module, None);
                    device.free_memory(index_buf_memory, None);
                    device.destroy_buffer(index_buf, None);
                    device.free_memory(vert_input_buf_memory, None);
                    device.destroy_buffer(vert_input_buf, None);
                    for &framebuffer in framebuffers.iter() {
                        device.destroy_framebuffer(framebuffer, None);
                    }
                    device.destroy_render_pass(renderpass, None);
                }

                unsafe {
                    device.device_wait_idle().unwrap();
                    device.destroy_semaphore(present_complete_semaphore, None);
                    device.destroy_semaphore(rendering_complete_semaphore, None);
                    device.destroy_fence(draw_cmds_reuse_fence, None);
                    device.destroy_fence(setup_cmds_reuse_fence, None);
                    device.free_memory(depth_image_memory, None);
                    device.destroy_image_view(depth_image_view, None);
                    device.destroy_image(depth_image, None);
                    for &image_view in present_image_views.iter() {
                        device.destroy_image_view(image_view, None)
                    }
                    device.destroy_command_pool(cmd_pool, None);
                    swapchain.destroy_swapchain(swapchain_handle, None);
                    device.destroy_device(None);
                    surface.destroy();
                    debug_utils_loader.destroy_debug_utils_messenger(debug_callback, None);
                    instance.destroy_instance(None);
                }

                *control_flow = ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                // render loop
                unsafe {
                    let (present_index, _) = swapchain
                        .acquire_next_image(
                            swapchain_handle,
                            std::u64::MAX,
                            present_complete_semaphore,
                            vk::Fence::null(),
                        )
                        .unwrap();
                    let clear_values = [
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(renderpass)
                        .framebuffer(framebuffers[present_index as usize])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: surface_resolution,
                        })
                        .clear_values(&clear_values);

                    record_submit_commandbuffer(
                        &device,
                        draw_cmd_buf,
                        draw_cmds_reuse_fence,
                        present_queue,
                        &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                        &[present_complete_semaphore],
                        &[rendering_complete_semaphore],
                        |device, draw_command_buffer| {
                            device.cmd_begin_render_pass(
                                draw_command_buffer,
                                &render_pass_begin_info,
                                vk::SubpassContents::INLINE,
                            );
                            device.cmd_bind_pipeline(
                                draw_command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                graphics_pipeline,
                            );
                            device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                            device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                            device.cmd_bind_vertex_buffers(
                                draw_command_buffer,
                                0,
                                &[vert_input_buf],
                                &[0],
                            );
                            device.cmd_bind_index_buffer(
                                draw_command_buffer,
                                index_buf,
                                0,
                                vk::IndexType::UINT32,
                            );
                            device.cmd_draw_indexed(
                                draw_command_buffer,
                                index_data.len() as u32,
                                1,
                                0,
                                0,
                                1,
                            );
                            device.cmd_end_render_pass(draw_command_buffer);
                        },
                    );

                    let wait_semaphores = [rendering_complete_semaphore];
                    let swapchains = [swapchain_handle];
                    let image_indices = [present_index];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&wait_semaphores)
                        .swapchains(&swapchains)
                        .image_indices(&image_indices);

                    swapchain
                        .queue_present(present_queue, &present_info)
                        .unwrap();
                }
            }
            _ => (),
        }
    });
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
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "[{:?}] [{:?}] ({}, {}) {}",
        message_severity, message_type, message_id_name, message_id_number, message
    );

    return vk::FALSE;
}

/// Find the index of the physical device's memory type that has the properties
/// indicated by `flags`.
fn find_memory_type_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            ((1 << index) & memory_req.memory_type_bits != 0)
                && ((memory_type.property_flags & flags) == flags)
        })
        .map(|(index, _memory_type)| index as _)
}

fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    cmd_buf: vk::CommandBuffer,
    cmd_buf_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .wait_for_fences(&[cmd_buf_reuse_fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");

        device
            .reset_fences(&[cmd_buf_reuse_fence])
            .expect("Reset fence failed.");

        device
            .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
            .expect("Reset command buffer failed.");

        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
            .expect("Begin command buffer failed.");

        f(device, cmd_buf);

        device
            .end_command_buffer(cmd_buf)
            .expect("End command buffer failed.");

        let command_buffers = vec![cmd_buf];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info.build()], cmd_buf_reuse_fence)
            .expect("Submit queue failed.");
    }
}
