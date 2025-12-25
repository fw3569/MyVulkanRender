#pragma once

#include <cstdint>

#include "third_part/vulkan_headers.h"

namespace DeferLightingPass {
void UpdateResources();
void CreatePipeline(const vk::raii::ShaderModule& shader_module);
void Draw(uint32_t image_index, uint32_t frame_index, vk::Viewport viewport,
          vk::Rect2D scissor);
void UpdateDescriptorSetInfo();
}  // namespace DeferLightingPass
