#include "model.h"

#define NOMINMAX
#include "command_buffer.h"
#include "context.h"
#include "memory.h"

void GenerateMipmaps(const vk::raii::Image& image, vk::Format format,
                     int32_t width, int32_t height, uint32_t mip_levels) {
  vk::FormatProperties properties =
      Context::Instance()->g_physical_device.getFormatProperties(format);
  if (!(properties.optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
    throw std::runtime_error(
        "texture image format does not support linear blitting!");
  }
  vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
  vk::ImageMemoryBarrier barrier{
      .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
      .dstAccessMask = vk::AccessFlagBits::eTransferRead,
      .oldLayout = vk::ImageLayout::eTransferDstOptimal,
      .newLayout = vk::ImageLayout::eTransferSrcOptimal,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
  int32_t mip_width = width, mip_height = height;
  for (uint32_t i = 1; i < mip_levels; ++i) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eTransfer, {}, {},
                                   nullptr, barrier);
    vk::ImageBlit image_blit{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{mip_width, mip_height, 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, i, 0, 1},
        .dstOffsets =
            std::array{vk::Offset3D{0, 0, 0},
                       vk::Offset3D{mip_width > 1 ? mip_width / 2 : 1,
                                    mip_height > 1 ? mip_height / 2 : 1, 1}},
    };
    command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image,
                             vk::ImageLayout::eTransferDstOptimal, image_blit,
                             vk::Filter::eLinear);
    barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eFragmentShader,
                                   {}, {}, nullptr, barrier);
    mip_width = mip_width > 1 ? mip_width / 2 : 1;
    mip_height = mip_height > 1 ? mip_height / 2 : 1;
  }
  barrier.subresourceRange.baseMipLevel = mip_levels - 1;
  barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eFragmentShader, {},
                                 {}, nullptr, barrier);
  EndOneTimeCommandBuffer(command_buffer);
}

void CreateTexture() {
  int tex_width, tex_height, tex_channels;
  stbi_uc* pixels = stbi_load(DATA_FILE_PATH "/viking_room.png", &tex_width,
                              &tex_height, &tex_channels, STBI_rgb_alpha);
  vk::DeviceSize image_size = tex_width * tex_height * 4;
  uint32_t mip_levels =
      std::floor(std::log2(std::max(tex_width, tex_height))) + 1;
  if (!pixels) {
    throw std::runtime_error("failed to load texture image!");
  }
  vk::raii::Buffer buffer = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
  CreateBuffer(image_size, vk::BufferUsageFlagBits::eTransferSrc,
               vk::SharingMode::eExclusive,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               buffer, memory);
  void* data = memory.mapMemory(0, image_size);
  memcpy(data, pixels, image_size);
  memory.unmapMemory();
  // stbi need free
  stbi_image_free(pixels);
  CreateImage(tex_width, tex_height, mip_levels, vk::SampleCountFlagBits::e1,
              vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eTransferSrc |
                  vk::ImageUsageFlagBits::eSampled,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              Context::Instance()->g_texture_image,
              Context::Instance()->g_texture_image_memory);
  TransformImageLayoutImmediately(
      Context::Instance()->g_texture_image, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eTransferDstOptimal, {},
      vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTopOfPipe,
      vk::PipelineStageFlagBits::eTransfer, vk::ImageAspectFlagBits::eColor, 0,
      mip_levels);
  CopyBufferToImage(buffer, Context::Instance()->g_texture_image, tex_width,
                    tex_height);
  GenerateMipmaps(Context::Instance()->g_texture_image,
                  vk::Format::eR8G8B8A8Srgb, tex_width, tex_height, mip_levels);
  Context::Instance()->g_texture_image_view = CreateImageView(
      *Context::Instance()->g_texture_image, 0, mip_levels,
      vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
  vk::PhysicalDeviceProperties properties =
      Context::Instance()->g_physical_device.getProperties();
  vk::SamplerCreateInfo sampler_info{
      .flags = {},
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = vk::SamplerAddressMode::eRepeat,
      .addressModeV = vk::SamplerAddressMode::eRepeat,
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .mipLodBias = 0.0f,
      .anisotropyEnable = vk::True,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0,
      .maxLod = vk::LodClampNone,
      .borderColor = vk::BorderColor::eIntTransparentBlack,
      .unnormalizedCoordinates = vk::False,
  };
  Context::Instance()->g_texture_image_sampler =
      vk::raii::Sampler(Context::Instance()->g_device, sampler_info);
}

void LoadMesh() {
  tinyobj::attrib_t attr;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err, warn;
  if (!tinyobj::LoadObj(&attr, &shapes, &materials, &warn, &err,
                        DATA_FILE_PATH "/viking_room.obj")) {
    throw std::runtime_error(warn + err);
  }
  std::unordered_map<Vertex, uint32_t> index_map;
  for (const auto& shape : shapes) {
    for (const auto& index : shape.mesh.indices) {
      Vertex vertex{
          .position = {attr.vertices[3 * index.vertex_index + 0],
                       attr.vertices[3 * index.vertex_index + 1],
                       attr.vertices[3 * index.vertex_index + 2]},
          .roughness_f0 = {Context::Instance()->g_pbr_roughness,
                           Context::Instance()->g_pbr_f0,
                           Context::Instance()->g_pbr_f0,
                           Context::Instance()->g_pbr_f0},
          .normal =
              {
                  attr.normals[3 * index.normal_index + 0],
                  attr.normals[3 * index.normal_index + 1],
                  attr.normals[3 * index.normal_index + 2],
              },
          .metallic = Context::Instance()->g_pbr_metallic,
          .tex_coord = {attr.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attr.texcoords[2 * index.texcoord_index + 1]}};
      if (!index_map.count(vertex)) {
        index_map[vertex] = Context::Instance()->g_vertex_in.size();
        Context::Instance()->g_vertex_in.emplace_back(vertex);
      }
      Context::Instance()->g_index_in.emplace_back(index_map[vertex]);
    }
  }
}

void UpdateMesh() {
  for (Vertex& vertex : Context::Instance()->g_vertex_in) {
    vertex.roughness_f0 = {
        Context::Instance()->g_pbr_roughness, Context::Instance()->g_pbr_f0,
        Context::Instance()->g_pbr_f0, Context::Instance()->g_pbr_f0};
    vertex.metallic = Context::Instance()->g_pbr_metallic;
  }
  uint32_t size = sizeof(Context::Instance()->g_vertex_in[0]) *
                  Context::Instance()->g_vertex_in.size();
  memcpy(Context::Instance()->g_transfer_buffer_maped,
         Context::Instance()->g_vertex_in.data(), size);
  CopyBuffer(Context::Instance()->g_transfer_buffer,
             Context::Instance()->g_vertex_buffer, size);
}

void CreateVertexBuffer() {
  uint32_t size = sizeof(Context::Instance()->g_vertex_in[0]) *
                  Context::Instance()->g_vertex_in.size();
  CreateBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
               vk::SharingMode::eExclusive,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               Context::Instance()->g_transfer_buffer,
               Context::Instance()->g_transfer_buffer_memory);
  Context::Instance()->g_transfer_buffer_maped =
      Context::Instance()->g_transfer_buffer_memory.mapMemory(0, size);
  memcpy(Context::Instance()->g_transfer_buffer_maped,
         Context::Instance()->g_vertex_in.data(), size);
  CreateBuffer(size,
               vk::BufferUsageFlagBits::eVertexBuffer |
                   vk::BufferUsageFlagBits::eTransferDst,
               vk::SharingMode::eExclusive,
               vk::MemoryPropertyFlagBits::eDeviceLocal,
               Context::Instance()->g_vertex_buffer,
               Context::Instance()->g_vertex_buffer_memory);
  CopyBuffer(Context::Instance()->g_transfer_buffer,
             Context::Instance()->g_vertex_buffer, size);
}

void CreateIndexBuffer() {
  uint32_t size = sizeof(Context::Instance()->g_index_in[0]) *
                  Context::Instance()->g_index_in.size();
  memcpy(Context::Instance()->g_transfer_buffer_maped,
         Context::Instance()->g_index_in.data(), size);
  CreateBuffer(size,
               vk::BufferUsageFlagBits::eIndexBuffer |
                   vk::BufferUsageFlagBits::eTransferDst,
               vk::SharingMode::eExclusive,
               vk::MemoryPropertyFlagBits::eDeviceLocal,
               Context::Instance()->g_index_buffer,
               Context::Instance()->g_index_buffer_memory);
  CopyBuffer(Context::Instance()->g_transfer_buffer,
             Context::Instance()->g_index_buffer, size);
}

void LoadModel() {
  LoadMesh();
  CreateTexture();
  CreateVertexBuffer();
  CreateIndexBuffer();
}
