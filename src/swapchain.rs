use crate::surface::Surface;
use crate::device::Device;

use ash::{
    extensions::khr,
    Instance,
    vk,
};

pub struct Swapchain {
    surface: Surface,
    swapchain: vk::SwapchainKHR,
    swapchain_pfns: khr::Swapchain,
    present_queue_index: u32,
    color_format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    image_views: Vec<(vk::Image, vk::ImageView)>,
}

impl Swapchain {
    pub fn new(surface: Surface, instance: &Instance, device: &Device, extent: &mut vk::Extent2D, vsync: bool) -> Self {
        let queue_family_props = device.queue_family_properties();

        // Find a queue that supports both graphics and presenting. The queue is used to 
        // present swap chain images to the windowing system.
        let mut graphics_queue_index = 0;
        let mut graphics_queue_found = false;
        let mut present_queue_index = 0;
        let mut present_queue_found = false;
        for (i, queue_prop) in queue_family_props.iter().enumerate() {
            if !graphics_queue_found && queue_prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_queue_index = i as u32;
                graphics_queue_found = true;
            }
            
            if !present_queue_found && surface.support_present(i as u32).unwrap() {
                present_queue_index = i as u32;
                present_queue_found = true;
            }
        }
        assert!(graphics_queue_found && present_queue_found);
        // Separate graphics and present queues are not supported.
        assert!(graphics_queue_index == present_queue_index);

        let surface_formats = surface.formats().unwrap();

        // If the only format present is VK_FORMAT_UNDEFINED, there is no preferred format, 
        // so we assume VK_FORMAT_B8G8R8A8_UNORM.
        if surface_formats.size() == 1 && surface_formats[0].format == vk::Format::VK_FORMAT_UNDEFINED {
            color_format = vk::Format::VK_FORMAT_B8G8R8A8_UNORM;
            color_space = surface_formats[0].color_space;
        } else {
            // Find format VK_FORMAT_B8G8R8A8_UNORM.
            let mut found = false;
            for format in surface_formats.iter() {
                if format.format == VK_FORMAT_B8G8R8A8_UNORM {
                    color_format = format.format;
                    color_space = format.color_space;
                    found = true;
                    break;
                }
            }
            // If not found, use the first one.
            if !found {
                color_format = surface_formats[0].format;
                color_space = surface_formats[0].color_space;
            }
        }

        let surface_caps = unsafe { surface_pfns.get_physical_device_surface_capabilities(physical_device, surface) };
        let present_modes = unsafe { surface_pfns.get_physical_device_surface_present_modes(physical_device, surface) };

        // If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
        let swapchain_extent = match surface_caps.current_extent.width {
            std::u32::MAX => *extent,
            _ => {
                // If surface size is defined, swapchain must match.
                *extent = surface_caps.current_extent;
                surface_caps.current_extent
            }
        };

        // Select a present mode for the swapchain.

        // VK_PRESENT_MODE_FIFO_KHR must always be present as per spec.
        // This mode waits for the vertical blank (v-sync).
        let swapchain_present_mode = if vsync {
            vk::PresentModeKHR::FIFO
        } else {
            // If vsync is not requested, try to find mailbox mode. It's the lowest latency non-tearing mode available.
            present_modes.iter().find(vk::PresentModeKHR::MAILBOX).unwrap()
        };

        // Determine the number of images.
        let mut desired_num_swapchain_images = surface_caps.min_image_count;
        if surface_caps.max_image_count > 0 && desired_num_swapchain_images > surface_caps.max_image_count {
            desired_num_swapchain_images = surface_caps.max_image_count;
        }

        // Find the transformation of the surface.
        let pre_transform = if surface_caps.supported_transforms.contains(vk::SurfaceTransformFlagsKHR::IDENTITY) {
            // Prefer non-rotated transform.
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_caps.current_transform
        };

        // Find a supported composite alpha format (not all devices support alpha opaque).
        let composite_alpha_flags = [vk::CompositeAlphaFlagsKHR::OPAQUE, 
                       vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED, 
                       vk::CompositeAlphaFlagsKHR::POS_MULTIPLIED,
                       vk::CompositeAlphaFlagsKHR::INHERIT];
        let mut composite_alpha = vk::CompositeAlphaFlagsKHR::OPAQUE;
        for alpha in composite_alpha_flags.into_iter() {
            // Simply select the first composite alpha available.
            if surface_caps.supported_composite_alpha.contains(alpha) {
                composite_alpha = alpha;
                break;
            }
        }

        let mut swapchain_usage_flags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if surface_caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
            swapchain_usage_flags |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if surface_caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DSC) {
            swapchain_usage_flags |= vk::ImageUsageFlags::TRANSFER_DSC;
        }

        let swapchain_createinfo = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(desired_num_swapchain_images)
            .image_format(color_format)
            .image_color_space(color_space)
            .image_extent(swapchain_extent)
            .image_usage(swapchain_usage_flags)
            .pre_transform(pre_transform)
            .image_array_layers(1)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .present_mode(swapchain_present_mode)
            .clipped(true)
            .composite_alpha(composite_alpha);

        let swapchain_pfns = khr::Swapchain(instance, logical_device);
        let swapchain = unsafe { swapchain_pfns.create_swapchain(&swapchain_createinfo, None).unwrap() };

        let mut image_views = Vec::new();

        let swapchain_images = unsafe { swapchain_pfns.get_swapchain_images(swapchain).unwrap() };
        for image in swapchain_images {
            let color_attachment_image_view_createinfo = vk::ImageViewCreateInfo::builder()
                .format(color_format)
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
                .view_type(vk::ImageViewType::TYPE_2D)
                .flags(vk::ImageViewCreateFlags::empty());

                let view = unsafe { logical_device.create_image_view(&color_attachment_image_view_createinfo, None).unwrap() }
                image_views.push((image, view));
        }

        khr::Swapchain {
            physical_device,
            swapchain,
            swapchain_pfns, 
            surface,
            surface_pfns,
            present_queue_index,
            color_format,
            color_space,
            image_views,
        }
    }
}