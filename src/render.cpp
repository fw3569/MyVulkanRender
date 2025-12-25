#include "render.h"

#include <vector>

#include "context.h"
#include "descriptor_set.h"
#include "memory.h"
#include "model.h"
#include "render_pass/bloom_pass.h"
#include "render_pass/defer_lighting_pass.h"
#include "render_pass/particle_pass.h"
#include "render_pass/shadowmap_pass.h"
#include "swapchain.h"
#include "third_part/vulkan_headers.h"
#include "utils.h"

namespace {
vk::raii::ShaderModule CreateShaderModule(const char* shader_file_path) {
  std::vector<char> shader_code = ReadFile(shader_file_path);
  vk::ShaderModuleCreateInfo create_info{
      .codeSize = shader_code.size() * sizeof(char),
      .pCode = reinterpret_cast<uint32_t*>(shader_code.data()),
  };
  return vk::raii::ShaderModule{Context::Instance()->g_device, create_info};
}

void CreatePipelines() {
  LOG(std::string("SHADER_FILE_PATH: ") + SHADER_FILE_PATH);
  vk::raii::ShaderModule shader_module = CreateShaderModule(SHADER_FILE_PATH);
  DeferLightingPass::CreatePipeline(shader_module);
  BloomPass::CreatePipeline(shader_module);
  ShadowmapPass::CreatePipeline(shader_module);
  ParticlePass::CreatePipeline(shader_module);
}

void RecordCommandBuffer(
    uint32_t image_index, uint32_t frame_index,
    vk::Viewport viewport = vk::Viewport(
        0.0f, 0.0f,
        static_cast<float>(Context::Instance()->g_swapchain_extent.width),
        static_cast<float>(Context::Instance()->g_swapchain_extent.height),
        0.0f, 1.0f),
    vk::Rect2D scissor = vk::Rect2D({0, 0},
                                    Context::Instance()->g_swapchain_extent)) {
  Context::Instance()->g_command_buffer[frame_index].begin({});
  TransformImageLayout(Context::Instance()->g_swapchain_images[image_index],
                       frame_index, vk::ImageLayout::eUndefined,
                       vk::ImageLayout::eTransferDstOptimal, {},
                       vk::AccessFlagBits2::eTransferWrite,
                       vk::PipelineStageFlagBits2::eTopOfPipe,
                       vk::PipelineStageFlagBits2::eTransfer);

  ShadowmapPass::Draw(image_index, frame_index, viewport, scissor);
  DeferLightingPass::Draw(image_index, frame_index, viewport, scissor);

  if (Context::Instance()->g_enable_bloom) {
    BloomPass::Draw(image_index, frame_index, viewport, scissor);
  }

  vk::ImageMemoryBarrier bloom_barrier{
      .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
      .dstAccessMask = vk::AccessFlagBits::eTransferRead,
      .oldLayout = vk::ImageLayout::eGeneral,
      .newLayout = vk::ImageLayout::eTransferSrcOptimal,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = Context::Instance()->g_bloom_image,
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
  };
  Context::Instance()->g_command_buffer[frame_index].pipelineBarrier(
      vk::PipelineStageFlagBits::eFragmentShader,
      vk::PipelineStageFlagBits::eTransfer, {}, {}, nullptr, bloom_barrier);
  vk::ImageBlit image_blit{
      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .srcOffsets =
          std::array{
              vk::Offset3D{0, 0, 0},
              vk::Offset3D{static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.width),
                           static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.height),
                           1},
          },
      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .dstOffsets =
          std::array{
              vk::Offset3D{0, 0, 0},
              vk::Offset3D{static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.width),
                           static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.height),
                           1},
          },
  };
  Context::Instance()->g_command_buffer[frame_index].blitImage(
      Context::Instance()->g_bloom_image, vk::ImageLayout::eTransferSrcOptimal,
      Context::Instance()->g_swapchain_images[image_index],
      vk::ImageLayout::eTransferDstOptimal, image_blit, vk::Filter::eLinear);

  // imgui
  TransformImageLayout(Context::Instance()->g_swapchain_images[image_index],
                       frame_index, vk::ImageLayout::eTransferDstOptimal,
                       vk::ImageLayout::eColorAttachmentOptimal,
                       vk::AccessFlagBits2::eTransferWrite,
                       vk::AccessFlagBits2::eColorAttachmentWrite,
                       vk::PipelineStageFlagBits2::eTransfer,
                       vk::PipelineStageFlagBits2::eColorAttachmentOutput);
  std::vector<vk::RenderingAttachmentInfo> imgui_attachment_infos{{
      .imageView = Context::Instance()->g_swapchain_image_views[image_index],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eLoad,
      .storeOp = vk::AttachmentStoreOp::eStore,
  }};
  vk::RenderingInfo imgui_rendering_info{
      .renderArea = {.offset = {0, 0},
                     .extent = Context::Instance()->g_swapchain_extent},
      .layerCount = 1,
      .colorAttachmentCount =
          static_cast<uint32_t>(imgui_attachment_infos.size()),
      .pColorAttachments = imgui_attachment_infos.data(),
      .pDepthAttachment = nullptr,
  };
  Context::Instance()->g_command_buffer[frame_index].beginRendering(
      imgui_rendering_info);
  ImGui_ImplVulkan_RenderDrawData(
      ImGui::GetDrawData(),
      *Context::Instance()->g_command_buffer[frame_index]);
  Context::Instance()->g_command_buffer[frame_index].endRendering();

  // present
  TransformImageLayout(Context::Instance()->g_swapchain_images[image_index],
                       frame_index, vk::ImageLayout::eColorAttachmentOptimal,
                       vk::ImageLayout::ePresentSrcKHR,
                       vk::AccessFlagBits2::eColorAttachmentWrite, {},
                       vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                       vk::PipelineStageFlagBits2::eBottomOfPipe);
  Context::Instance()->g_command_buffer[frame_index].end();
}

void CreateSyncObjects() {
  for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
    Context::Instance()->g_present_complete_semaphore.emplace_back(
        vk::raii::Semaphore(Context::Instance()->g_device,
                            vk::SemaphoreCreateInfo()));
    Context::Instance()->g_render_finished_semaphore.emplace_back(
        vk::raii::Semaphore(Context::Instance()->g_device,
                            vk::SemaphoreCreateInfo()));
    Context::Instance()->g_draw_fence.emplace_back(
        vk::raii::Fence(Context::Instance()->g_device,
                        {.flags = vk::FenceCreateFlagBits::eSignaled}));
  }
  vk::SemaphoreTypeCreateInfo timeline_type_info{
      .semaphoreType = vk::SemaphoreType::eTimeline,
      .initialValue = 0,
  };
  Context::Instance()->g_particle_compute_semaphore = vk::raii::Semaphore(
      Context::Instance()->g_device, {.pNext = &timeline_type_info});
}

bool CpuPrepareData(uint32_t frame_index) {
  UniformBufferObject ubo;
  glm::vec3 camera_pos{1.0f, 1.0f, 1.0f};
  ubo.modu = glm::rotate<float>(
      glm::mat4(1.0f), Context::Instance()->g_time * glm::radians(10.0f),
      glm::vec3(0.0f, 0.0f, 1.0f));
  // https://learnopengl.com/Getting-started/Coordinate-Systems
  // https://learnopengl.com/Getting-started/Camera
  ubo.view = glm::lookAt(camera_pos, glm::vec3(0.0f, 0.0f, 0.0f),
                         glm::vec3(0.0f, 0.0f, -1.0f));
  ubo.proj = glm::perspective<float>(
      glm::radians(90.0f),
      static_cast<float>(Context::Instance()->g_swapchain_extent.width) /
          static_cast<float>(Context::Instance()->g_swapchain_extent.height),
      0.1f, 3.0f);
  ubo.light.pos = glm::vec3(1.0f, 0.0f, 2.0f);
  ubo.light.intensities = glm::vec3(Context::Instance()->g_light_intensity);
  ubo.camera_pos = camera_pos;
  ubo.light_view = glm::lookAt(ubo.light.pos, glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, -1.0f));
  ubo.light_proj = glm::perspective<float>(
      glm::radians(90.0f),
      static_cast<float>(Context::Instance()->g_shadowmap_width) /
          static_cast<float>(Context::Instance()->g_shadowmap_height),
      0.8f, 3.0f);
  ubo.shadowmap_resolution = glm::vec2(Context::Instance()->g_shadowmap_width,
                                       Context::Instance()->g_shadowmap_height);
  ubo.shadowmap_scale =
      glm::vec2(Context::Instance()->g_shadowmap_width /
                    (float)Context::Instance()->g_swapchain_extent.width,
                Context::Instance()->g_shadowmap_height /
                    (float)Context::Instance()->g_swapchain_extent.height);
  memcpy(Context::Instance()->g_ubo_buffer_maped[frame_index], &ubo,
         sizeof(ubo));
  UpdateMesh();
  return true;
}

void UpdateParticle(uint32_t frame_index) {
  ParticleUbo ubo;
  static double last_particle_update_time = 0.0f;
  if (last_particle_update_time == 0.0f) {
    last_particle_update_time = Context::Instance()->g_time;
  }
  ubo.delta_time = Context::Instance()->g_time - last_particle_update_time;
  last_particle_update_time = Context::Instance()->g_time;
  memcpy(Context::Instance()->g_particle_ubo_buffer_maped[frame_index], &ubo,
         sizeof(ParticleUbo));

  uint32_t compute_cb_index =
      frame_index + Context::Instance()->g_frame_in_flight * 2;
  Context::Instance()->g_command_buffer[compute_cb_index].reset();
  Context::Instance()->g_command_buffer[compute_cb_index].begin({});
  ParticlePass::Compute(compute_cb_index, frame_index);
  Context::Instance()->g_command_buffer[compute_cb_index].end();
  vk::PipelineStageFlags compute_wait_dst_stage_mask =
      vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
  uint64_t wait_semaphore_value = Context::Instance()->g_particle_compute_count;
  uint64_t signal_semaphore_value =
      ++Context::Instance()->g_particle_compute_count;
  vk::TimelineSemaphoreSubmitInfo compute_semaphore_submit_info{
      .waitSemaphoreValueCount = 1,
      .pWaitSemaphoreValues = &wait_semaphore_value,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &signal_semaphore_value,
  };
  vk::SubmitInfo compute_submit_info{
      .pNext = &compute_semaphore_submit_info,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*Context::Instance()->g_particle_compute_semaphore,
      .pWaitDstStageMask = &compute_wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers =
          &*Context::Instance()->g_command_buffer[compute_cb_index],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &*Context::Instance()->g_particle_compute_semaphore,
  };
  Context::Instance()->g_queue.submit(compute_submit_info, nullptr);
}

bool GpuPrepareData(uint32_t frame_index) {
  UpdateParticle(frame_index);
  return true;
}

void CreateUboBuffer() {
  Context::Instance()->g_ubo_buffer.clear();
  Context::Instance()->g_ubo_buffer_memory.clear();
  Context::Instance()->g_ubo_buffer_maped.clear();

  for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
    uint32_t size = sizeof(UniformBufferObject);
    vk::raii::Buffer buffer = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    CreateBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::SharingMode::eExclusive,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 buffer, memory);
    void* data = memory.mapMemory(0, size);
    Context::Instance()->g_ubo_buffer.emplace_back(std::move(buffer));
    Context::Instance()->g_ubo_buffer_memory.emplace_back(std::move(memory));
    Context::Instance()->g_ubo_buffer_maped.emplace_back(data);
  }
}
}  // namespace

void RenderManager::Init() {
  CreateUboBuffer();
  ShadowmapPass::UpdateResources();
  DeferLightingPass::UpdateResources();
  BloomPass::UpdateResources();
  ParticlePass::UpdateResources();

  UpdateDescriptorSetInfo();

  DescriptorSetManager::CreateDescriptorSetLayout();
  CreatePipelines();
  DescriptorSetManager::CreateDescriptorPool();
  DescriptorSetManager::CreateDescriptorSets();
  CreateCommandBuffer();
  CreateSyncObjects();
  SwapChainManager::RegisterRecreateFunction(
      RenderManager::UpdateDescriptorSetInfo);
  SwapChainManager::RegisterRecreateFunction(
      DescriptorSetManager::UpdateDescriptorSets);
}

void RenderManager::UpdateDescriptorSetInfo() {
  DescriptorSetManager::ClearDescriptorSetInfo();
  std::vector<vk::DescriptorBufferInfo> buffer_info;
  {
    std::vector<vk::DescriptorBufferInfo> buffer_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      buffer_info.emplace_back(Context::Instance()->g_ubo_buffer[i], 0,
                               sizeof(UniformBufferObject));
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        0, vk::DescriptorType::eUniformBuffer,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        {}, buffer_info);
  }
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(*Context::Instance()->g_texture_image_sampler,
                              *Context::Instance()->g_texture_image_view,
                              vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        1, vk::DescriptorType::eCombinedImageSampler,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
  ShadowmapPass::UpdateDescriptorSetInfo();
  DeferLightingPass::UpdateDescriptorSetInfo();
  BloomPass::UpdateDescriptorSetInfo();
  ParticlePass::UpdateDescriptorSetInfo();
}

bool RenderManager::PrepareData(uint32_t frame_index) {
  return CpuPrepareData(frame_index) && GpuPrepareData(frame_index);
}

bool RenderManager::DrawFrame(uint32_t frame_index) {
  if (Context::Instance()->g_window_resized) {
    SwapChainManager::RecreateSwapchain();
  }
  auto [result, image_index] =
      Context::Instance()->g_swapchain.acquireNextImage(
          UINT64_MAX,
          *Context::Instance()->g_present_complete_semaphore[frame_index],
          nullptr);
  bool window_resized = false;
  if (result != vk::Result::eSuccess) {
    LOG("acquireNextImage: " + to_string(result));
    if (result == vk::Result::eErrorOutOfDateKHR) {
      SwapChainManager::RecreateSwapchain();
      return false;
    } else if (result == vk::Result::eSuboptimalKHR) {
      window_resized = true;
    } else {
      return false;
    }
  }
  RecordCommandBuffer(image_index, frame_index);
  vk::PipelineStageFlags wait_dst_stage_mask =
      vk::PipelineStageFlagBits::eColorAttachmentOutput |
      vk::PipelineStageFlagBits::eVertexInput;
  std::vector<uint64_t> wait_values{
      Context::Instance()->g_particle_compute_count, 1};
  vk::TimelineSemaphoreSubmitInfo graphics_semaphore_submit_info{
      .waitSemaphoreValueCount = 2,
      .pWaitSemaphoreValues = wait_values.data(),
      .signalSemaphoreValueCount = 0,
  };
  std::vector<vk::Semaphore> graphics_wait_semaphores{
      *Context::Instance()->g_particle_compute_semaphore,
      *Context::Instance()->g_present_complete_semaphore[frame_index]};
  vk::SubmitInfo submit_info{
      .pNext = &graphics_semaphore_submit_info,
      .waitSemaphoreCount =
          static_cast<uint32_t>(graphics_wait_semaphores.size()),
      .pWaitSemaphores = graphics_wait_semaphores.data(),
      .pWaitDstStageMask = &wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &*Context::Instance()->g_command_buffer[frame_index],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores =
          &*Context::Instance()->g_render_finished_semaphore[frame_index],
  };
  Context::Instance()->g_device.resetFences(
      *Context::Instance()->g_draw_fence[frame_index]);
  Context::Instance()->g_queue.submit(
      submit_info, *Context::Instance()->g_draw_fence[frame_index]);
  while (vk::Result::eTimeout ==
         Context::Instance()->g_device.waitForFences(
             *Context::Instance()->g_draw_fence[frame_index], vk::True,
             UINT64_MAX));
  const vk::PresentInfoKHR present_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores =
          &*Context::Instance()->g_render_finished_semaphore[frame_index],
      .swapchainCount = 1,
      .pSwapchains = &*Context::Instance()->g_swapchain,
      .pImageIndices = &image_index,
  };
  try {
    result = Context::Instance()->g_queue.presentKHR(present_info);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  if (result != vk::Result::eSuccess) {
    LOG("presentKHR: " + to_string(result));
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
      window_resized = true;
    }
  }
  if (window_resized || Context::Instance()->g_window_resized) {
    SwapChainManager::RecreateSwapchain();
  }
  return true;
}
