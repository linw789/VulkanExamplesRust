use crate::surface::Surface;

use ash::{extensions::khr, vk, Instance};

use std::vec::Vec;

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
    pub fn new(
        surface: Surface,
        instance: &Instance,
        device: &Device,
        extent: &mut vk::Extent2D,
        vsync: bool,
    ) -> Self {
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

        let color_format;
        let color_space;

        let surface_formats = surface.formats().unwrap();

        // If the only format present is VK_FORMAT_UNDEFINED, there is no preferred format,
        // so we assume VK_FORMAT_B8G8R8A8_UNORM.
        if surface_formats.len() == 1 && surface_formats[0].format == vk::Format::UNDEFINED {
            color_format = vk::Format::B8G8R8A8_UNORM;
            color_space = surface_formats[0].color_space;
        } else {
            // Find format VK_FORMAT_B8G8R8A8_UNORM.
            let mut found = false;
            for format in surface_formats.iter() {
                if format.format == vk::Format::B8G8R8A8_UNORM {
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

        let surface_caps = surface.capabilities().unwrap();
        let present_modes = surface.present_modes().unwrap();

        // If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
        if surface_caps.current_extent.width != std::u32::MAX {
            // If surface size is defined, swapchain must match.
            *extent = surface_caps.current_extent;
        }
        let swapchain_extent = *extent;

        // Select a present mode for the swapchain.

        // VK_PRESENT_MODE_FIFO_KHR must always be present as per spec.
        // This mode waits for the vertical blank (v-sync).
        let swapchain_present_mode = if vsync {
            vk::PresentModeKHR::FIFO
        } else {
            // If vsync is not requested, try to find mailbox mode. It's the lowest latency non-tearing mode available.
            present_modes
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap()
        };

        // Determine the number of images.
        let mut desired_num_swapchain_images = surface_caps.min_image_count;
        if surface_caps.max_image_count > 0
            && desired_num_swapchain_images > surface_caps.max_image_count
        {
            desired_num_swapchain_images = surface_caps.max_image_count;
        }

        // Find the transformation of the surface.
        let pre_transform = if surface_caps
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            // Prefer non-rotated transform.
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_caps.current_transform
        };

        // Find a supported composite alpha format (not all devices support alpha opaque).
        let composite_alpha_flags = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ];
        let mut composite_alpha = vk::CompositeAlphaFlagsKHR::OPAQUE;
        for &alpha in composite_alpha_flags.into_iter() {
            // Simply select the first composite alpha available.
            if surface_caps.supported_composite_alpha.contains(alpha) {
                composite_alpha = alpha;
                break;
            }
        }

        let mut swapchain_usage_flags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if surface_caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            swapchain_usage_flags |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if surface_caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            swapchain_usage_flags |= vk::ImageUsageFlags::TRANSFER_DST;
        }

        let swapchain_createinfo = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.vk_surface())
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

        let swapchain_pfns = khr::Swapchain::new(instance, device.device());
        let swapchain = unsafe {
            swapchain_pfns
                .create_swapchain(&swapchain_createinfo, None)
                .unwrap()
        };

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

            let view = unsafe {
                device
                    .device()
                    .create_image_view(&color_attachment_image_view_createinfo, None)
                    .unwrap()
            };
            image_views.push((image, view));
        }

        Swapchain {
            surface,
            swapchain,
            swapchain_pfns,
            present_queue_index,
            color_format,
            color_space,
            image_views,
        }
    }
}
