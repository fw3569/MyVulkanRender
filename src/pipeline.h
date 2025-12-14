#pragma once

#include "third_part/vulkan_headers.h"

vk::raii::ShaderModule CreateShaderModule(const char* shader_file_path);
void CreatePipelines();
