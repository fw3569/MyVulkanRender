#include "particle_pass.h"

#include "context.h"
#include "descriptor_set.h"
#include "memory.h"
#include "swapchain.h"
#include "utils.h"

namespace {
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
}  // namespace

void ParticlePass::UpdateResources() { CreateParticleResources(); }

void ParticlePass::CreatePipeline(const vk::raii::ShaderModule& shader_module) {
  vk::PipelineShaderStageCreateInfo
      particle_pipeline_shader_stage_create_info[2] = {
          {
              .stage = vk::ShaderStageFlagBits::eVertex,
              .module = shader_module,
              .pName = "vertParticle",
              .pSpecializationInfo = nullptr,
          },
          {
              .stage = vk::ShaderStageFlagBits::eFragment,
              .module = shader_module,
              .pName = "fragParticle",
              .pSpecializationInfo = nullptr,
          },
      };
  std::vector dynamic_states = {vk::DynamicState::eViewport,
                                vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dyanmic_state_create_info = {
      .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
      .pDynamicStates = dynamic_states.data(),
  };
  auto binding_desc = Particle::GetBindingDescription();
  auto particle_attribute_desc = Particle::GetAttributeDescription();
  vk::PipelineVertexInputStateCreateInfo particle_vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding_desc,
      .vertexAttributeDescriptionCount = particle_attribute_desc.size(),
      .pVertexAttributeDescriptions = particle_attribute_desc.data(),
  };
  vk::PipelineInputAssemblyStateCreateInfo input_assembly_info{
      .topology = vk::PrimitiveTopology::ePointList};
  vk::PipelineViewportStateCreateInfo viewport_state_info{
      .viewportCount = 1,
      .pViewports = nullptr,
      .scissorCount = 1,
      .pScissors = nullptr,
  };
  vk::PipelineRasterizationStateCreateInfo rasterization_create_info{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasConstantFactor = 1.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth = 1.0f,
  };
  vk::PipelineMultisampleStateCreateInfo multisample_create_info{
      .rasterizationSamples = Context::Instance()->g_msaa_samples,
      .sampleShadingEnable = vk::False,
  };
  vk::PipelineDepthStencilStateCreateInfo particle_depth_stencil_info{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::False,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False,
  };
  vk::PipelineColorBlendAttachmentState transparent_blend_attachment{
      .blendEnable = vk::True,
      .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
      .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = vk::BlendFactor::eOne,
      .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  std::vector<vk::PipelineColorBlendAttachmentState> particle_color_blend_infos(
      5, transparent_blend_attachment);
  vk::PipelineColorBlendStateCreateInfo particle_color_blend_info{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount =
          static_cast<uint32_t>(particle_color_blend_infos.size()),
      .pAttachments = particle_color_blend_infos.data(),
  };
  vk::PipelineLayoutCreateInfo pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*Context::Instance()->g_descriptor_set_layout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr,
  };
  std::vector<vk::Format> graphsic_formats{
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
  };
  vk::PipelineRenderingCreateInfo particle_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(graphsic_formats.size()),
      .pColorAttachmentFormats = graphsic_formats.data(),
      .depthAttachmentFormat = Context::Instance()->g_depth_image_format,
  };
  Context::Instance()->g_particle_pipeline_layout = vk::raii::PipelineLayout(
      Context::Instance()->g_device, pipeline_layout_info);
  vk::GraphicsPipelineCreateInfo particle_pipeline_info{
      .pNext = &particle_pipeline_rending_info,
      .stageCount = 2,
      .pStages = particle_pipeline_shader_stage_create_info,
      .pVertexInputState = &particle_vertex_input_info,
      .pInputAssemblyState = &input_assembly_info,
      .pTessellationState = {},
      .pViewportState = &viewport_state_info,
      .pRasterizationState = &rasterization_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = &particle_depth_stencil_info,
      .pColorBlendState = &particle_color_blend_info,
      .pDynamicState = &dyanmic_state_create_info,
      .layout = Context::Instance()->g_particle_pipeline_layout,
      .renderPass = nullptr,
      .subpass = {},
      .basePipelineHandle = {},
      .basePipelineIndex = {},
  };

  Context::Instance()->g_particle_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, particle_pipeline_info);

  vk::PipelineShaderStageCreateInfo compute_pipeline_shader_stage_create_info{
      .stage = vk::ShaderStageFlagBits::eCompute,
      .module = shader_module,
      .pName = "compParticle",
      .pSpecializationInfo = nullptr,
  };
  vk::PipelineLayoutCreateInfo compute_pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*Context::Instance()->g_descriptor_set_layout,
  };
  Context::Instance()->g_compute_pipeline_layout = vk::raii::PipelineLayout(
      Context::Instance()->g_device, compute_pipeline_layout_info);
  vk::ComputePipelineCreateInfo compute_pipeline_info{
      .stage = compute_pipeline_shader_stage_create_info,
      .layout = Context::Instance()->g_compute_pipeline_layout,
  };
  Context::Instance()->g_compute_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, compute_pipeline_info);
}

void ParticlePass::Compute(uint32_t compute_cb_index, uint32_t frame_index) {
  Context::Instance()->g_command_buffer[compute_cb_index].bindPipeline(
      vk::PipelineBindPoint::eCompute, Context::Instance()->g_compute_pipeline);
  Context::Instance()->g_command_buffer[compute_cb_index].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      Context::Instance()->g_compute_pipeline_layout, 0,
      *Context::Instance()->g_descriptor_sets[frame_index], nullptr);
  Context::Instance()->g_command_buffer[compute_cb_index].dispatch(
      Context::Instance()->kParticleCount / 256, 1, 1);
}

void ParticlePass::UpdateDescriptorSetInfo() {
  {
    std::vector<vk::DescriptorBufferInfo> buffer_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      buffer_info.emplace_back(Context::Instance()->g_particle_ubo_buffer[i], 0,
                               sizeof(ParticleUbo));
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        2, vk::DescriptorType::eUniformBuffer,
        vk::ShaderStageFlagBits::eCompute, {}, buffer_info);
  }
  {
    std::vector<vk::DescriptorBufferInfo> buffer_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      buffer_info.emplace_back(
          Context::Instance()
              ->g_particle_buffer[(i - 1 +
                                   Context::Instance()->g_frame_in_flight) %
                                  Context::Instance()->g_frame_in_flight],
          0, sizeof(Particle) * Context::Instance()->kParticleCount);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        3, vk::DescriptorType::eStorageBuffer,
        vk::ShaderStageFlagBits::eCompute, {}, buffer_info);
  }
  {
    std::vector<vk::DescriptorBufferInfo> buffer_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      buffer_info.emplace_back(
          Context::Instance()->g_particle_buffer[i], 0,
          sizeof(Particle) * Context::Instance()->kParticleCount);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        4, vk::DescriptorType::eStorageBuffer,
        vk::ShaderStageFlagBits::eCompute, {}, buffer_info);
  }
}
