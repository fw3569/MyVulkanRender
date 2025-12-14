#include "descriptor_set.h"

#include "context.h"
#include "third_part/vulkan_headers.h"

void CreateDescriptorSetLayout() {
  std::vector<vk::DescriptorSetLayoutBinding> layout_bindings{
      {.binding = 0,
       .descriptorType = vk::DescriptorType::eUniformBuffer,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eVertex |
                     vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 1,
       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 2,
       .descriptorType = vk::DescriptorType::eUniformBuffer,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eCompute,
       .pImmutableSamplers = nullptr},
      {.binding = 3,
       .descriptorType = vk::DescriptorType::eStorageBuffer,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eCompute,
       .pImmutableSamplers = nullptr},
      {.binding = 4,
       .descriptorType = vk::DescriptorType::eStorageBuffer,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eCompute,
       .pImmutableSamplers = nullptr},
      {.binding = 5,
       .descriptorType = vk::DescriptorType::eInputAttachment,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 6,
       .descriptorType = vk::DescriptorType::eInputAttachment,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 7,
       .descriptorType = vk::DescriptorType::eInputAttachment,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 8,
       .descriptorType = vk::DescriptorType::eInputAttachment,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 9,
       .descriptorType = vk::DescriptorType::eSampledImage,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 10,
       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr},
      {.binding = 11,
       .descriptorType = vk::DescriptorType::eSampledImage,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eFragment,
       .pImmutableSamplers = nullptr}};
  vk::DescriptorSetLayoutCreateInfo set_layout_info{
      .flags = {},
      .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
      .pBindings = layout_bindings.data()};
  Context::Instance()->g_descriptor_set_layout = vk::raii::DescriptorSetLayout(
      Context::Instance()->g_device, set_layout_info);
}
void CreateDescriptorPool() {
  std::vector<vk::DescriptorPoolSize> pool_sizes{
      {.type = vk::DescriptorType::eUniformBuffer,
       .descriptorCount = Context::Instance()->g_frame_in_flight * 2},
      {.type = vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount = Context::Instance()->g_frame_in_flight * 2},
      {.type = vk::DescriptorType::eStorageBuffer,
       .descriptorCount = Context::Instance()->g_frame_in_flight * 2},
      {.type = vk::DescriptorType::eInputAttachment,
       .descriptorCount = Context::Instance()->g_frame_in_flight * 4},
      {.type = vk::DescriptorType::eSampledImage,
       .descriptorCount = Context::Instance()->g_frame_in_flight * 2}};
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
void UpdateDescriptorSets() {
  for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
    vk::DescriptorBufferInfo buffer_info{
        .buffer = Context::Instance()->g_ubo_buffer[i],
        .offset = 0,
        .range = sizeof(UniformBufferObject)};
    vk::DescriptorBufferInfo particle_ubo_buffer_info{
        .buffer = Context::Instance()->g_particle_ubo_buffer[i],
        .offset = 0,
        .range = sizeof(ParticleUbo)};
    vk::DescriptorBufferInfo particle_last_frame_buffer_info{
        .buffer =
            Context::Instance()
                ->g_particle_buffer[(i - 1 +
                                     Context::Instance()->g_frame_in_flight) %
                                    Context::Instance()->g_frame_in_flight],
        .offset = 0,
        .range = sizeof(Particle) * Context::Instance()->kParticleCount};
    vk::DescriptorBufferInfo particle_this_frame_buffer_info{
        .buffer = Context::Instance()->g_particle_buffer[i],
        .offset = 0,
        .range = sizeof(Particle) * Context::Instance()->kParticleCount};
    vk::DescriptorImageInfo image_info{
        .sampler = *Context::Instance()->g_texture_image_sampler,
        .imageView = *Context::Instance()->g_texture_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo gbuffer_color_info{
        .imageView = *Context::Instance()->g_gbuffer_color_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo gbuffer_position_info{
        .imageView = *Context::Instance()->g_gbuffer_position_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo gbuffer_normal_info{
        .imageView = *Context::Instance()->g_gbuffer_normal_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo gbuffer_roughness_f0_info{
        .imageView = *Context::Instance()->g_gbuffer_roughness_f0_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo shadowmap_info{
        .imageView = *Context::Instance()->g_shadowmap_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo depth_info{
        .sampler = *Context::Instance()->g_depth_image_sampler,
        .imageView = *Context::Instance()->g_depth_image_view,
        .imageLayout = vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal};
    vk::DescriptorImageInfo bloom_info{
        .imageView = *Context::Instance()->g_bloom_image_view,
        .imageLayout = vk::ImageLayout::eGeneral};
    std::vector<vk::WriteDescriptorSet> descriptor_write{
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 0,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .pBufferInfo = &buffer_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &image_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 2,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .pBufferInfo = &particle_ubo_buffer_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 3,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &particle_last_frame_buffer_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 4,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &particle_this_frame_buffer_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 5,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eInputAttachment,
         .pImageInfo = &gbuffer_color_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 6,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eInputAttachment,
         .pImageInfo = &gbuffer_position_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 7,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eInputAttachment,
         .pImageInfo = &gbuffer_normal_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 8,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eInputAttachment,
         .pImageInfo = &gbuffer_roughness_f0_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 9,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &shadowmap_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 10,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &depth_info},
        {.dstSet = Context::Instance()->g_descriptor_sets[i],
         .dstBinding = 11,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &bloom_info}};
    Context::Instance()->g_device.updateDescriptorSets(descriptor_write, {});
  }
}
void CreateDescriptorSets() {
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
