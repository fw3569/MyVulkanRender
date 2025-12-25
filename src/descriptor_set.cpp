#include "descriptor_set.h"

#include "context.h"
#include "third_part/vulkan_headers.h"

using DescriptorSetInfo = DescriptorSetManager::DescriptorSetInfo;

namespace {
std::vector<DescriptorSetInfo> descriptor_set_infos;
}  // namespace

void DescriptorSetManager::CreateDescriptorSetLayout() {
  std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
  for (const DescriptorSetInfo& descriptor_set_info : descriptor_set_infos) {
    layout_bindings.emplace_back(descriptor_set_info.binding,
                                 descriptor_set_info.type, 1,
                                 descriptor_set_info.stage, nullptr);
  }
  vk::DescriptorSetLayoutCreateInfo set_layout_info{
      .flags = {},
      .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
      .pBindings = layout_bindings.data()};
  Context::Instance()->g_descriptor_set_layout = vk::raii::DescriptorSetLayout(
      Context::Instance()->g_device, set_layout_info);
}

void DescriptorSetManager::CreateDescriptorPool() {
  std::vector<vk::DescriptorPoolSize> pool_sizes;
  std::unordered_map<vk::DescriptorType, uint32_t> type_counts;
  for (const DescriptorSetInfo& descriptor_set_info : descriptor_set_infos) {
    ++type_counts[descriptor_set_info.type];
  }
  for (const auto& type_count : type_counts) {
    pool_sizes.emplace_back(vk::DescriptorPoolSize{
        .type = type_count.first,
        .descriptorCount =
            Context::Instance()->g_frame_in_flight * type_count.second,
    });
  }
  vk::DescriptorPoolCreateInfo descriptor_pool_info{
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = static_cast<uint32_t>(Context::Instance()->g_frame_in_flight *
                                       pool_sizes.size()),
      .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
  };
  Context::Instance()->g_descriptor_pool = vk::raii::DescriptorPool(
      Context::Instance()->g_device, descriptor_pool_info);
}

void DescriptorSetManager::UpdateDescriptorSets() {
  for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
    std::vector<vk::WriteDescriptorSet> descriptor_write;
    for (const DescriptorSetInfo& descriptor_set_info : descriptor_set_infos) {
      switch (descriptor_set_info.type) {
        case vk::DescriptorType::eSampler:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eInputAttachment:
          descriptor_write.emplace_back(vk::WriteDescriptorSet{
              .dstSet = Context::Instance()->g_descriptor_sets[i],
              .dstBinding = descriptor_set_info.binding,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = descriptor_set_info.type,
              .pImageInfo = &descriptor_set_info.image_info[i]});
          break;
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eUniformBufferDynamic:
        case vk::DescriptorType::eStorageBufferDynamic:
          descriptor_write.emplace_back(vk::WriteDescriptorSet{
              .dstSet = Context::Instance()->g_descriptor_sets[i],
              .dstBinding = descriptor_set_info.binding,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = descriptor_set_info.type,
              .pBufferInfo = &descriptor_set_info.buffer_info[i]});
          break;
        default:
          break;
      }
    }
    Context::Instance()->g_device.updateDescriptorSets(descriptor_write, {});
  }
}

void DescriptorSetManager::CreateDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(
      Context::Instance()->g_frame_in_flight,
      *Context::Instance()->g_descriptor_set_layout);
  vk::DescriptorSetAllocateInfo alloc_info{
      .descriptorPool = *Context::Instance()->g_descriptor_pool,
      .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
      .pSetLayouts = layouts.data()};
  Context::Instance()->g_descriptor_sets =
      Context::Instance()->g_device.allocateDescriptorSets(alloc_info);
  UpdateDescriptorSets();
}

void DescriptorSetManager::ClearDescriptorSetInfo() {
  descriptor_set_infos.clear();
}

void DescriptorSetManager::RegisterDescriptorSetInfo(
    uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stage,
    std::vector<vk::DescriptorImageInfo> image_info,
    std::vector<vk::DescriptorBufferInfo> buffer_info) {
  descriptor_set_infos.emplace_back(binding, type, stage, image_info,
                                    buffer_info);
}
