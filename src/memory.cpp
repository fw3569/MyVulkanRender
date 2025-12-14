#include "memory.h"

uint32_t FindMemoryType(uint32_t type_filter,
                        vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties memory_properties =
      Context::Instance()->g_physical_device.getMemoryProperties();
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((type_filter & (1 << i)) &&
        (memory_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

void CreateBuffer(uint32_t size, vk::BufferUsageFlags usage,
                  vk::SharingMode sharing_mode,
                  vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer,
                  vk::raii::DeviceMemory& memory) {
  vk::BufferCreateInfo vertex_buffer_info{
      .flags = {}, .size = size, .usage = usage, .sharingMode = sharing_mode};
  buffer = vk::raii::Buffer(Context::Instance()->g_device, vertex_buffer_info);
  vk::MemoryRequirements memory_requirements = buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo memory_alloc_info{
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex =
          FindMemoryType(memory_requirements.memoryTypeBits, properties)};
  memory =
      vk::raii::DeviceMemory(Context::Instance()->g_device, memory_alloc_info);
  buffer.bindMemory(*memory, 0);
}

void CopyBuffer(vk::raii::Buffer& src_buffer, vk::raii::Buffer& dst_buffer,
                uint32_t size) {
  vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
  command_buffer.copyBuffer(src_buffer, dst_buffer, vk::BufferCopy{0, 0, size});
  EndOneTimeCommandBuffer(command_buffer);
}

void CreateImage(uint32_t width, uint32_t height, uint32_t mip_levels,
                 vk::SampleCountFlagBits sample_count, vk::Format format,
                 vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                 vk::MemoryPropertyFlags properties, vk::raii::Image& image,
                 vk::raii::DeviceMemory& memory) {
  vk::ImageCreateInfo image_info{
      .flags = {},
      .imageType = vk::ImageType::e2D,
      .format = format,
      .extent = {width, height, 1},
      .mipLevels = mip_levels,
      .arrayLayers = 1,
      .samples = sample_count,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &Context::Instance()->g_queue_index,
      .initialLayout = vk::ImageLayout::eUndefined,
  };
  image = vk::raii::Image(Context::Instance()->g_device, image_info);
  vk::MemoryRequirements memory_requirements = image.getMemoryRequirements();
  vk::MemoryAllocateInfo memory_alloc_info{
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex =
          FindMemoryType(memory_requirements.memoryTypeBits, properties)};
  memory =
      vk::raii::DeviceMemory(Context::Instance()->g_device, memory_alloc_info);
  image.bindMemory(*memory, 0);
}

vk::raii::ImageView CreateImageView(const vk::Image& image,
                                    uint32_t base_mip_level,
                                    uint32_t mip_levels, vk::Format format,
                                    vk::ImageAspectFlagBits aspect) {
  vk::ImageViewCreateInfo image_view_create_info{
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .components =
          {
              .r = static_cast<vk::ComponentSwizzle>(
                  VK_COMPONENT_SWIZZLE_IDENTITY),
              .g = static_cast<vk::ComponentSwizzle>(
                  VK_COMPONENT_SWIZZLE_IDENTITY),
              .b = static_cast<vk::ComponentSwizzle>(
                  VK_COMPONENT_SWIZZLE_IDENTITY),
              .a = static_cast<vk::ComponentSwizzle>(
                  VK_COMPONENT_SWIZZLE_IDENTITY),
          },
      .subresourceRange = {.aspectMask = aspect,
                           .baseMipLevel = base_mip_level,
                           .levelCount = mip_levels,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
  return vk::raii::ImageView(Context::Instance()->g_device,
                             image_view_create_info);
}

void CopyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image,
                       uint32_t width, uint32_t height) {
  vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
  vk::BufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {width, height, 1}};
  command_buffer.copyBufferToImage(
      buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
  EndOneTimeCommandBuffer(command_buffer);
}

void TransformImageLayout(const vk::Image& image, uint32_t command_buffer_index,
                          vk::ImageLayout old_layout,
                          vk::ImageLayout new_layout,
                          vk::AccessFlags2 src_access_mask,
                          vk::AccessFlags2 dst_access_mask,
                          vk::PipelineStageFlags2 src_stage_mask,
                          vk::PipelineStageFlags2 dst_stage_mask,
                          vk::ImageAspectFlags aspect_flags,
                          uint32_t base_mip_level, uint32_t level_count) {
  vk::ImageMemoryBarrier2 barrier = {
      .srcStageMask = src_stage_mask,
      .srcAccessMask = src_access_mask,
      .dstStageMask = dst_stage_mask,
      .dstAccessMask = dst_access_mask,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {.aspectMask = aspect_flags,
                           .baseMipLevel = base_mip_level,
                           .levelCount = level_count,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
  vk::DependencyInfo dependency_info{
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  };
  Context::Instance()->g_command_buffer[command_buffer_index].pipelineBarrier2(
      dependency_info);
}

void TransformImageLayoutImmediately(
    const vk::raii::Image& image, vk::ImageLayout old_layout,
    vk::ImageLayout new_layout, vk::AccessFlags src_access_mask,
    vk::AccessFlags dst_access_mask, vk::PipelineStageFlags src_stage_mask,
    vk::PipelineStageFlags dst_stage_mask, vk::ImageAspectFlags aspect_flags,
    uint32_t base_mip_level, uint32_t level_count) {
  vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
  vk::ImageMemoryBarrier barrier{
      .srcAccessMask = src_access_mask,
      .dstAccessMask = dst_access_mask,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {.aspectMask = aspect_flags,
                           .baseMipLevel = base_mip_level,
                           .levelCount = level_count,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
  command_buffer.pipelineBarrier(src_stage_mask, dst_stage_mask, {}, {},
                                 nullptr, barrier);
  EndOneTimeCommandBuffer(command_buffer);
}
