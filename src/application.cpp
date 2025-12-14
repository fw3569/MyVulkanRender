#include "application.h"

#include <chrono>

#include "command_buffer.h"
#include "context.h"
#include "descriptor_set.h"
#include "device.h"
#include "model.h"
#include "pipeline.h"
#include "render.h"
#include "render_system.h"
#include "swapchain.h"
#include "utils.h"
#include "vulkan_configure.h"

void Application::Run() {
  Init();
  Work();
  Cleanup();
}

void Application::Init() {
  gui.InitWindow();
  InitVulkan();
  gui.InitImGui();
  m_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count() /
                 1000.0;
}
void Application::InitVulkan() {
  LOG(std::string("DATA_FILE_PATH: ") + DATA_FILE_PATH);
  VulkanConfigure::CreateInstance();
  gui.CreateaSurface();
  PickPhysicalDevice();
  CreateLogicalDevice();
  SwapChainManager::CreateSwapChain();
  SwapChainManager::RegisterRecreateFunction(CreateColorResources);
  SwapChainManager::RegisterRecreateFunction(CreateDepthResources);
  SwapChainManager::RegisterRecreateFunction(CreateShadowmapResources);
  SwapChainManager::RegisterRecreateFunction(CreateGbufferResources);
  SwapChainManager::RegisterRecreateFunction(CreateBloomResources);
  SwapChainManager::RegisterRecreateFunction(UpdateDescriptorSets);
  CreateColorResources();
  CreateDepthResources();
  CreateShadowmapResources();
  CreateGbufferResources();
  CreateBloomResources();
  CreateDescriptorSetLayout();
  CreatePipelines();
  CreateCommandPool();
  LoadMesh();
  CreateTextureImage();
  CreateTextureSampler();
  CreateVertexBuffer();
  CreateIndexBuffer();
  CreateUboBuffer();
  CreateParticleResources();
  CreateDescriptorPool();
  CreateDescriptorSets();
  CreateCommandBuffer();
  CreateSyncObjects();
}
void Application::Work() {
  while (!gui.Closed()) {
    constexpr uint32_t fps = 30;
    constexpr double draw_internal = 1.0 / fps;
    static double next_draw_time = 0.0;
    m_current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count() /
                     1000.0;
    gui.Update();
    if (next_draw_time < m_current_time && Tick()) {
      next_draw_time = m_current_time + draw_internal;
    }
  }

  Context::Instance()->g_device.waitIdle();
}
bool Application::Tick() {
  return CpuPrepareData() && GpuPrepareData() && DrawFrame();
}
bool Application::CpuPrepareData() {
  UniformBufferObject ubo;
  glm::vec3 camera_pos{1.0f, 1.0f, 1.0f};
  ubo.modu = glm::rotate<float>(
      glm::mat4(1.0f), (m_current_time - m_start_time) * glm::radians(10.0f),
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
  memcpy(Context::Instance()->g_ubo_buffer_maped[m_frame_index], &ubo,
         sizeof(ubo));
  UpdateMesh();
  return true;
}
void Application::UpdateParticle() {
  ParticleUbo ubo;
  static double last_particle_update_time = 0.0f;
  if (last_particle_update_time == 0.0f) {
    last_particle_update_time = m_current_time;
  }
  ubo.delta_time = m_current_time - last_particle_update_time;
  last_particle_update_time = m_current_time;
  memcpy(Context::Instance()->g_particle_ubo_buffer_maped[m_frame_index], &ubo,
         sizeof(ParticleUbo));

  uint32_t compute_cb_index =
      m_frame_index + Context::Instance()->g_frame_in_flight * 2;
  Context::Instance()->g_command_buffer[compute_cb_index].reset();
  Context::Instance()->g_command_buffer[compute_cb_index].begin({});
  Context::Instance()->g_command_buffer[compute_cb_index].bindPipeline(
      vk::PipelineBindPoint::eCompute, Context::Instance()->g_compute_pipeline);
  Context::Instance()->g_command_buffer[compute_cb_index].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      Context::Instance()->g_compute_pipeline_layout, 0,
      *Context::Instance()->g_descriptor_sets[m_frame_index], nullptr);
  Context::Instance()->g_command_buffer[compute_cb_index].dispatch(
      Context::Instance()->kParticleCount / 256, 1, 1);
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
bool Application::GpuPrepareData() {
  UpdateParticle();
  return true;
}
bool Application::DrawFrame() {
  if (Context::Instance()->g_window_resized) {
    SwapChainManager::RecreateSwapchain();
  }
  auto [result, image_index] =
      Context::Instance()->g_swapchain.acquireNextImage(
          UINT64_MAX,
          *Context::Instance()->g_present_complete_semaphore[m_frame_index],
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
  RecordCommandBuffer(image_index, m_frame_index);
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
      *Context::Instance()->g_present_complete_semaphore[m_frame_index]};
  vk::SubmitInfo submit_info{
      .pNext = &graphics_semaphore_submit_info,
      .waitSemaphoreCount =
          static_cast<uint32_t>(graphics_wait_semaphores.size()),
      .pWaitSemaphores = graphics_wait_semaphores.data(),
      .pWaitDstStageMask = &wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &*Context::Instance()->g_command_buffer[m_frame_index],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores =
          &*Context::Instance()->g_render_finished_semaphore[m_frame_index],
  };
  Context::Instance()->g_device.resetFences(
      *Context::Instance()->g_draw_fence[m_frame_index]);
  Context::Instance()->g_queue.submit(
      submit_info, *Context::Instance()->g_draw_fence[m_frame_index]);
  while (vk::Result::eTimeout ==
         Context::Instance()->g_device.waitForFences(
             *Context::Instance()->g_draw_fence[m_frame_index], vk::True,
             UINT64_MAX));
  const vk::PresentInfoKHR present_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores =
          &*Context::Instance()->g_render_finished_semaphore[m_frame_index],
      .swapchainCount = 1,
      .pSwapchains = &*Context::Instance()->g_swapchain,
      .pImageIndices = &image_index,
  };
  try {
    result = Context::Instance()->g_queue.presentKHR(present_info);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  m_frame_index = (m_frame_index + 1) % Context::Instance()->g_frame_in_flight;
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
  return result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR;
}
void Application::Cleanup() { gui.Cleanup(); }
