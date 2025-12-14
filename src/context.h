#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "data.h"
#include "third_part/glfw_headers.h"
#include "third_part/glm_headers.h"
#include "third_part/imgui_headers.h"
#include "third_part/stb_headers.h"
#include "third_part/tiny_obj_loader_headers.h"
#include "third_part/vulkan_headers.h"

class Context {
 public:
  static constexpr uint32_t kWindowWeight = 800;
  static constexpr uint32_t kWindowHeight = 600;
  GLFWwindow* g_window;
  std::mutex g_window_resized_mtx;
  std::atomic<bool> g_window_resized = false;
  vk::raii::Context g_vk_context;
  vk::raii::Instance g_vk_instance = nullptr;
  const std::vector<const char*> kRequiredDeviceExtensions = {
      vk::KHRSwapchainExtensionName, vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
      vk::KHRDynamicRenderingLocalReadExtensionName};
  vk::raii::PhysicalDevice g_physical_device = nullptr;
  vk::raii::Device g_device = nullptr;
  vk::raii::Queue g_queue = nullptr;
  uint32_t g_queue_index = 0;
  vk::raii::SurfaceKHR g_surface = nullptr;
  vk::raii::SwapchainKHR g_swapchain = nullptr;
  vk::Format g_swapchain_image_format = vk::Format::eUndefined;
  vk::Format g_gbuffer_format = vk::Format::eR32G32B32A32Sfloat;
  vk::Extent2D g_swapchain_extent;
  vk::raii::PipelineLayout g_pipeline_layout = nullptr;
  vk::raii::Pipeline g_graphics_pipeline = nullptr;
  vk::raii::PipelineLayout g_lighting_pipeline_layout = nullptr;
  vk::raii::Pipeline g_lighting_pipeline = nullptr;
  uint32_t g_frame_in_flight = 2;
  std::vector<vk::Image> g_swapchain_images;
  std::vector<vk::raii::ImageView> g_swapchain_image_views;
  vk::raii::Image g_color_image = nullptr;
  vk::raii::DeviceMemory g_color_image_memory = nullptr;
  vk::raii::ImageView g_color_image_view = nullptr;
  vk::SampleCountFlagBits g_msaa_samples = vk::SampleCountFlagBits::e1;
  vk::raii::Buffer g_vertex_buffer = nullptr;
  vk::raii::DeviceMemory g_vertex_buffer_memory = nullptr;
  vk::raii::Buffer g_index_buffer = nullptr;
  vk::raii::DeviceMemory g_index_buffer_memory = nullptr;
  std::vector<vk::raii::Buffer> g_ubo_buffer;
  std::vector<vk::raii::DeviceMemory> g_ubo_buffer_memory;
  std::vector<void*> g_ubo_buffer_maped;
  vk::raii::Buffer g_transfer_buffer = nullptr;
  vk::raii::DeviceMemory g_transfer_buffer_memory = nullptr;
  void* g_transfer_buffer_maped = nullptr;
  uint32_t g_mip_levels = 1;
  vk::raii::Image g_texture_image = nullptr;
  vk::raii::DeviceMemory g_texture_image_memory = nullptr;
  vk::raii::ImageView g_texture_image_view = nullptr;
  vk::raii::Sampler g_texture_image_sampler = nullptr;
  vk::raii::Image g_depth_image = nullptr;
  vk::raii::DeviceMemory g_depth_image_memory = nullptr;
  vk::raii::ImageView g_depth_image_view = nullptr;
  vk::raii::Sampler g_depth_image_sampler = nullptr;
  vk::raii::Image g_gbuffer_color_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_color_image_memory = nullptr;
  vk::raii::ImageView g_gbuffer_color_image_view = nullptr;
  vk::raii::Image g_gbuffer_position_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_position_image_memory = nullptr;
  vk::raii::ImageView g_gbuffer_position_image_view = nullptr;
  vk::raii::Image g_gbuffer_normal_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_normal_image_memory = nullptr;
  vk::raii::ImageView g_gbuffer_normal_image_view = nullptr;
  vk::raii::Image g_gbuffer_roughness_f0_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_roughness_f0_image_memory = nullptr;
  vk::raii::ImageView g_gbuffer_roughness_f0_image_view = nullptr;
  vk::Format g_depth_image_format = vk::Format::eUndefined;
  vk::raii::CommandPool g_command_pool = nullptr;
  std::vector<vk::raii::CommandBuffer> g_command_buffer;
  std::vector<vk::raii::Semaphore> g_present_complete_semaphore;
  std::vector<vk::raii::Semaphore> g_render_finished_semaphore;
  std::vector<vk::raii::Fence> g_draw_fence;
  vk::raii::DescriptorPool g_descriptor_pool = nullptr;
  vk::raii::DescriptorSetLayout g_descriptor_set_layout = nullptr;
  std::vector<vk::raii::DescriptorSet> g_descriptor_sets;
  std::vector<Vertex> g_vertex_in;
  std::vector<uint32_t> g_index_in;
  static constexpr uint32_t kParticleCount = 256;
  vk::raii::PipelineLayout g_particle_pipeline_layout = nullptr;
  vk::raii::Pipeline g_particle_pipeline = nullptr;
  vk::raii::PipelineLayout g_compute_pipeline_layout = nullptr;
  vk::raii::Pipeline g_compute_pipeline = nullptr;
  std::vector<vk::raii::Buffer> g_particle_ubo_buffer;
  std::vector<vk::raii::DeviceMemory> g_particle_ubo_buffer_memory;
  std::vector<void*> g_particle_ubo_buffer_maped;
  std::vector<vk::raii::Buffer> g_particle_buffer;
  std::vector<vk::raii::DeviceMemory> g_particle_buffer_memory;
  uint64_t g_particle_compute_count = 0;
  vk::raii::Semaphore g_particle_compute_semaphore = nullptr;
  vk::raii::PipelineLayout g_shadowmap_pipeline_layout = nullptr;
  vk::raii::Pipeline g_shadowmap_pipeline = nullptr;
  vk::Format g_shadowmap_image_format = vk::Format::eD32Sfloat;
  vk::raii::Image g_shadowmap_image = nullptr;
  vk::raii::DeviceMemory g_shadowmap_image_memory = nullptr;
  vk::raii::ImageView g_shadowmap_image_view = nullptr;
  uint32_t g_shadowmap_width = 1600;
  uint32_t g_shadowmap_height = 1200;
  vk::raii::Image g_bloom_image = nullptr;
  vk::raii::DeviceMemory g_bloom_image_memory = nullptr;
  vk::raii::ImageView g_bloom_image_view = nullptr;
  std::vector<vk::raii::ImageView> g_bloom_image_views;
  static constexpr uint32_t g_bloom_mip_levels = 6;
  vk::raii::PipelineLayout g_bloom_downsample_pipeline_layout = nullptr;
  vk::raii::Pipeline g_bloom_downsample_pipeline = nullptr;
  vk::raii::PipelineLayout g_bloom_upsample_pipeline_layout = nullptr;
  vk::raii::Pipeline g_bloom_upsample_pipeline = nullptr;
  static constexpr float kBloomRate = 1.5f;
  ImGuiContext* g_imgui_context;
  vk::raii::DescriptorPool g_imgui_pool = nullptr;
  float g_pbr_roughness = 0.5f;
  float g_pbr_f0 = 0.04f;
  float g_pbr_metallic = 0.0f;
  float g_light_intensity = 10.0f;
  bool g_enable_ssao = true;
  bool g_enable_bloom = true;

  const std::vector<const char*> kValidationLayers = {
      "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
  static constexpr bool kEnableValidationLayers = false;
#else
  static constexpr bool kEnableValidationLayers = true;
#endif

  static Context* Instance() {
    if (instance == nullptr) {
      instance.reset(new Context());
    }
    return instance.get();
  }
  static void Cleanup() { instance.reset(); }
  static std::unique_ptr<Context> instance;
};
