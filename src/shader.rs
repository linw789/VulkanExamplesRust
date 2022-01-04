use ash::util::read_spv;
use ash::{vk, Device};
use std::ffi::CString;
use std::fs;
use std::io::Cursor;
use std::path::Path;

pub struct ShaderModule {
    handle: vk::ShaderModule,
    stage: vk::ShaderStageFlags,
    entry_name: CString,
    spv: Vec<u8>,
}

impl ShaderModule {
    /// `spv_path` - An absolute path to a SPIR-V file.
    pub fn new<P: AsRef<Path>>(
        device: &Device,
        spv_path: P,
        stage: vk::ShaderStageFlags,
    ) -> ShaderModule {
        let spv_code = fs::read(spv_path).unwrap();
        let mut cursor = Cursor::new(spv_code);
        let code = read_spv(&mut cursor).expect("Failed to read shader.");
        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&code);

        let shader_module = unsafe {
            device
                .create_shader_module(&shader_info, None)
                .expect("Failed to create shader module.")
        };

        ShaderModule {
            handle: shader_module,
            stage,
            entry_name: CString::new("main").unwrap(),
            spv: cursor.into_inner(),
        }
    }

    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }

    pub fn shader_stage_create_info(&self) -> vk::PipelineShaderStageCreateInfo {
        vk::PipelineShaderStageCreateInfo {
            module: self.handle,
            p_name: self.entry_name.as_ptr(),
            stage: self.stage,
            ..Default::default()
        }
    }
}
