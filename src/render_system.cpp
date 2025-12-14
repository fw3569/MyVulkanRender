#include "render_system.h"

#include "context.h"
#include "memory.h"
#include "swapchain.h"

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

void CreateParticleResources() {
  Context::Instance()->g_particle_ubo_buffer.clear();
  Context::Instance()->g_particle_ubo_buffer_memory.clear();
  Context::Instance()->g_particle_ubo_buffer_maped.clear();
  Context::Instance()->g_particle_buffer.clear();
  Context::Instance()->g_particle_buffer_memory.clear();

  uint32_t size = sizeof(ParticleUbo);
  for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
    vk::raii::Buffer buffer = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    CreateBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::SharingMode::eExclusive,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 buffer, memory);
    void* data = memory.mapMemory(0, size);
    Context::Instance()->g_particle_ubo_buffer.emplace_back(std::move(buffer));
    Context::Instance()->g_particle_ubo_buffer_memory.emplace_back(
        std::move(memory));
    Context::Instance()->g_particle_ubo_buffer_maped.emplace_back(data);
    size = sizeof(Particle) * Context::Instance()->kParticleCount;
    buffer = nullptr;
    memory = nullptr;
    CreateBuffer(size,
                 vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eVertexBuffer |
                     vk::BufferUsageFlagBits::eTransferDst,
                 vk::SharingMode::eExclusive,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, buffer, memory);
    Context::Instance()->g_particle_buffer.emplace_back(std::move(buffer));
    Context::Instance()->g_particle_buffer_memory.emplace_back(
        std::move(memory));
  }
  std::vector<Particle> particles{Context::Instance()->kParticleCount};
  constexpr float kParticleRange = 1.3f;
  for (uint32_t i = 0; i < Context::Instance()->kParticleCount; ++i) {
    particles[i].pos.x =
        std::rand() / float(RAND_MAX) * kParticleRange - kParticleRange / 2.0f;
    particles[i].pos.y =
        std::rand() / float(RAND_MAX) * kParticleRange - kParticleRange / 2.0f;
    particles[i].pos.z = 0.3f;
    particles[i].v = glm::vec3{0.0f, 0.0f, 0.2f};
    particles[i].color = glm::vec3(1.0f);
  }
  vk::raii::Buffer temp_buffer = nullptr;
  vk::raii::DeviceMemory temp_memory = nullptr;
  size = sizeof(Particle) * Context::Instance()->kParticleCount;
  CreateBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
               vk::SharingMode::eExclusive,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               temp_buffer, temp_memory);
  void* data = temp_memory.mapMemory(0, size);
  memcpy(data, particles.data(), size);
  temp_memory.unmapMemory();
  CopyBuffer(
      temp_buffer,
      Context::Instance()
          ->g_particle_buffer[Context::Instance()->g_frame_in_flight - 1],
      size);
}
vk::Format FindSupportFormat(const std::vector<vk::Format>& candidates,
                             vk::ImageTiling tiling,
                             vk::FormatFeatureFlags flags) {
  for (const vk::Format& format : candidates) {
    vk::FormatProperties properties =
        Context::Instance()->g_physical_device.getFormatProperties(format);
    if (tiling == vk::ImageTiling::eLinear &&
        (properties.linearTilingFeatures & flags) == flags) {
      return format;
    }
    if (tiling == vk::ImageTiling::eOptimal &&
        (properties.optimalTilingFeatures & flags) == flags) {
      return format;
    }
  }
  throw std::runtime_error("failed to find supported format!");
}
vk::Format FindSupportDepthFormat() {
  return FindSupportFormat(
      {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
       vk::Format::eD24UnormS8Uint},
      vk::ImageTiling::eOptimal,
      vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}
void CreateColorResources() {
  vk::Format format = Context::Instance()->g_swapchain_image_format;
  CreateImage(Context::Instance()->g_swapchain_extent.width,
              Context::Instance()->g_swapchain_extent.height, 1,
              Context::Instance()->g_msaa_samples, format,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransientAttachment |
                  vk::ImageUsageFlagBits::eColorAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_color_image,
              Context::Instance()->g_color_image_memory);
  Context::Instance()->g_color_image_view =
      CreateImageView(*Context::Instance()->g_color_image, 0, 1, format,
                      vk::ImageAspectFlagBits::eColor);
}

void CreateDepthResources() {
  Context::Instance()->g_depth_image_format = FindSupportDepthFormat();
  CreateImage(Context::Instance()->g_swapchain_extent.width,
              Context::Instance()->g_swapchain_extent.height, 1,
              Context::Instance()->g_msaa_samples,
              Context::Instance()->g_depth_image_format,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eDepthStencilAttachment |
                  vk::ImageUsageFlagBits::eSampled,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_depth_image,
              Context::Instance()->g_depth_image_memory);
  Context::Instance()->g_depth_image_view =
      CreateImageView(*Context::Instance()->g_depth_image, 0, 1,
                      Context::Instance()->g_depth_image_format,
                      vk::ImageAspectFlagBits::eDepth);
  vk::PhysicalDeviceProperties properties =
      Context::Instance()->g_physical_device.getProperties();
  vk::SamplerCreateInfo sampler_info{
      .flags = {},
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eNearest,
      .addressModeU = vk::SamplerAddressMode::eClampToEdge,
      .addressModeV = vk::SamplerAddressMode::eClampToEdge,
      .addressModeW = vk::SamplerAddressMode::eClampToEdge,
      .mipLodBias = 0.0f,
      .anisotropyEnable = vk::False,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0,
      .maxLod = 0,
      .borderColor = vk::BorderColor::eIntTransparentBlack,
      .unnormalizedCoordinates = vk::False,
  };
  Context::Instance()->g_depth_image_sampler =
      vk::raii::Sampler(Context::Instance()->g_device, sampler_info);
}
void CreateShadowmapResources() {
  CreateImage(
      Context::Instance()->g_shadowmap_width,
      Context::Instance()->g_shadowmap_height, 1, vk::SampleCountFlagBits::e1,
      Context::Instance()->g_shadowmap_image_format, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment |
          vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      Context::Instance()->g_shadowmap_image,
      Context::Instance()->g_shadowmap_image_memory);
  Context::Instance()->g_shadowmap_image_view =
      CreateImageView(*Context::Instance()->g_shadowmap_image, 0, 1,
                      Context::Instance()->g_shadowmap_image_format,
                      vk::ImageAspectFlagBits::eDepth);
}

void CreateGbufferResources() {
  vk::Format format = Context::Instance()->g_gbuffer_format;
  // dont use msaa if using deferred lighting
  CreateImage(Context::Instance()->g_swapchain_extent.width,
              Context::Instance()->g_swapchain_extent.height, 1,
              Context::Instance()->g_msaa_samples, format,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransientAttachment |
                  vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eInputAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_gbuffer_color_image,
              Context::Instance()->g_gbuffer_color_image_memory);
  Context::Instance()->g_gbuffer_color_image_view =
      CreateImageView(*Context::Instance()->g_gbuffer_color_image, 0, 1, format,
                      vk::ImageAspectFlagBits::eColor);
  CreateImage(Context::Instance()->g_swapchain_extent.width,
              Context::Instance()->g_swapchain_extent.height, 1,
              Context::Instance()->g_msaa_samples, format,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransientAttachment |
                  vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eInputAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_gbuffer_position_image,
              Context::Instance()->g_gbuffer_position_image_memory);
  Context::Instance()->g_gbuffer_position_image_view =
      CreateImageView(*Context::Instance()->g_gbuffer_position_image, 0, 1,
                      format, vk::ImageAspectFlagBits::eColor);
  CreateImage(Context::Instance()->g_swapchain_extent.width,
              Context::Instance()->g_swapchain_extent.height, 1,
              Context::Instance()->g_msaa_samples, format,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransientAttachment |
                  vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eInputAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_gbuffer_normal_image,
              Context::Instance()->g_gbuffer_normal_image_memory);
  Context::Instance()->g_gbuffer_normal_image_view =
      CreateImageView(*Context::Instance()->g_gbuffer_normal_image, 0, 1,
                      format, vk::ImageAspectFlagBits::eColor);
  CreateImage(Context::Instance()->g_swapchain_extent.width,
              Context::Instance()->g_swapchain_extent.height, 1,
              Context::Instance()->g_msaa_samples, format,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransientAttachment |
                  vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eInputAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_gbuffer_roughness_f0_image,
              Context::Instance()->g_gbuffer_roughness_f0_image_memory);
  Context::Instance()->g_gbuffer_roughness_f0_image_view =
      CreateImageView(*Context::Instance()->g_gbuffer_roughness_f0_image, 0, 1,
                      format, vk::ImageAspectFlagBits::eColor);
}

void CreateBloomResources() {
  CreateImage(Context::Instance()->g_swapchain_extent.width,
              Context::Instance()->g_swapchain_extent.height,
              Context::Instance()->g_bloom_mip_levels,
              vk::SampleCountFlagBits::e1,
              Context::Instance()->g_gbuffer_format, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eTransferSrc |
                  vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_bloom_image,
              Context::Instance()->g_bloom_image_memory);
  Context::Instance()->g_bloom_image_view = CreateImageView(
      *Context::Instance()->g_bloom_image, 0,
      Context::Instance()->g_bloom_mip_levels,
      Context::Instance()->g_gbuffer_format, vk::ImageAspectFlagBits::eColor);
  Context::Instance()->g_bloom_image_views.clear();
  for (int i = 0; i < Context::Instance()->g_bloom_mip_levels; ++i) {
    Context::Instance()->g_bloom_image_views.emplace_back(
        CreateImageView(*Context::Instance()->g_bloom_image, i, 1,
                        Context::Instance()->g_gbuffer_format,
                        vk::ImageAspectFlagBits::eColor));
  }
}
