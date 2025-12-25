#pragma once

#include <cstdint>

#include "third_part/vulkan_headers.h"

namespace ParticlePass {
void UpdateResources();
void CreatePipeline(const vk::raii::ShaderModule& shader_module);
void Compute(uint32_t compute_cb_index, uint32_t frame_index);
// draw particle is in draw pass, ect. defer_lighting_pass
void UpdateDescriptorSetInfo();
}  // namespace ParticlePass
