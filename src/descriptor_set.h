#pragma once

#include <cstdint>
#include <vector>

#include "third_part/vulkan_headers.h"

namespace DescriptorSetManager {
struct DescriptorSetInfo {
  uint32_t binding;
  vk::DescriptorType type;
  vk::ShaderStageFlags stage;
  std::vector<vk::DescriptorImageInfo> image_info;
  std::vector<vk::DescriptorBufferInfo> buffer_info;
};
void CreateDescriptorSetLayout();
void CreateDescriptorPool();
void UpdateDescriptorSets();
void CreateDescriptorSets();
void ClearDescriptorSetInfo();
void RegisterDescriptorSetInfo(
    uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stage,
    std::vector<vk::DescriptorImageInfo> image_info,
    std::vector<vk::DescriptorBufferInfo> buffer_info);

}  // namespace DescriptorSetManager
