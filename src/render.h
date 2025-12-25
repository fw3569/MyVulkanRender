#pragma once

#include <cstdint>

namespace RenderManager {
void Init();
void UpdateDescriptorSetInfo();
bool PrepareData(uint32_t frame_index);
bool DrawFrame(uint32_t frame_index);
}  // namespace RenderManager
