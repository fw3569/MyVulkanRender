#include "command_buffer.h"

vk::raii::CommandBuffer BeginOneTimeCommandBuffer() {
  vk::raii::CommandBuffer command_buffer = nullptr;
  vk::CommandBufferAllocateInfo alloc_info{
      .commandPool = Context::Instance()->g_command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1};
  command_buffer = std::move(
      vk::raii::CommandBuffers(Context::Instance()->g_device, alloc_info)
          .front());
  command_buffer.begin(
      {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  return command_buffer;
}

void EndOneTimeCommandBuffer(vk::raii::CommandBuffer& command_buffer) {
  command_buffer.end();
  Context::Instance()->g_queue.submit(
      vk::SubmitInfo{.commandBufferCount = 1,
                     .pCommandBuffers = &*command_buffer},
      nullptr);
  Context::Instance()->g_queue.waitIdle();
}

void CreateCommandPool() {
  vk::CommandPoolCreateInfo pool_info{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = Context::Instance()->g_queue_index};
  Context::Instance()->g_command_pool =
      vk::raii::CommandPool(Context::Instance()->g_device, pool_info);
}

void CreateCommandBuffer() {
  vk::CommandBufferAllocateInfo alloc_info{
      .commandPool = Context::Instance()->g_command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount =
          static_cast<uint32_t>(Context::Instance()->g_frame_in_flight) * 3};
  Context::Instance()->g_command_buffer =
      vk::raii::CommandBuffers(Context::Instance()->g_device, alloc_info);
}
