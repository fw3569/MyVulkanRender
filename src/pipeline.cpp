#include "pipeline.h"

#include "context.h"
#include "utils.h"

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
  vk::PipelineShaderStageCreateInfo pipeline_shader_stage_create_info[2] = {
      {
          .stage = vk::ShaderStageFlagBits::eVertex,
          .module = shader_module,
          .pName = "vertMain",
          .pSpecializationInfo = nullptr,
      },
      {
          .stage = vk::ShaderStageFlagBits::eFragment,
          .module = shader_module,
          .pName = "fragMain",
          .pSpecializationInfo = nullptr,
      },
  };
  std::vector dynamic_states = {vk::DynamicState::eViewport,
                                vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dyanmic_state_create_info = {
      .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
      .pDynamicStates = dynamic_states.data(),
  };
  auto binding_desc = Vertex::GetBindingDescription();
  auto attribute_desc = Vertex::GetAttributeDescription();
  vk::PipelineVertexInputStateCreateInfo vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding_desc,
      .vertexAttributeDescriptionCount = attribute_desc.size(),
      .pVertexAttributeDescriptions = attribute_desc.data(),
  };
  vk::PipelineInputAssemblyStateCreateInfo input_assembly_info{
      .topology = vk::PrimitiveTopology::eTriangleList};
  // vk::Viewport viewport{
  //     .x = 0.0f,
  //     .y = 0.0f,
  //     .width =
  //     static_cast<float>(Context::Instance()->g_swapchain_extent.width),
  //     .height =
  //     static_cast<float>(Context::Instance()->g_swapchain_extent.height),
  //     .minDepth = 0.0f,
  //     .maxDepth = 1.0f};
  // vk::Rect2D scissor{vk::Offset2D{0, 0},
  // Context::Instance()->g_swapchain_extent};
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
  vk::PipelineDepthStencilStateCreateInfo depth_stencil_info{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False,
  };
  vk::PipelineColorBlendAttachmentState opaque_blend_attachment{
      .blendEnable = vk::False,
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments(
      5, opaque_blend_attachment);
  vk::PipelineColorBlendStateCreateInfo color_blend_info{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = static_cast<uint32_t>(color_blend_attachments.size()),
      .pAttachments = color_blend_attachments.data(),
      .blendConstants = {},
  };
  vk::PipelineLayoutCreateInfo pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*Context::Instance()->g_descriptor_set_layout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr,
  };
  Context::Instance()->g_pipeline_layout = vk::raii::PipelineLayout(
      Context::Instance()->g_device, pipeline_layout_info);
  std::vector<vk::Format> graphsic_formats{
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
  };
  vk::PipelineRenderingCreateInfo pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(graphsic_formats.size()),
      .pColorAttachmentFormats = graphsic_formats.data(),
      .depthAttachmentFormat = Context::Instance()->g_depth_image_format,
      .stencilAttachmentFormat = vk::Format::eUndefined,
  };
  vk::GraphicsPipelineCreateInfo pipeline_info{
      .pNext = &pipeline_rending_info,
      .stageCount = 2,
      .pStages = pipeline_shader_stage_create_info,
      .pVertexInputState = &vertex_input_info,
      .pInputAssemblyState = &input_assembly_info,
      .pTessellationState = {},
      .pViewportState = &viewport_state_info,
      .pRasterizationState = &rasterization_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = &depth_stencil_info,
      .pColorBlendState = &color_blend_info,
      .pDynamicState = &dyanmic_state_create_info,
      .layout = Context::Instance()->g_pipeline_layout,
      .renderPass = nullptr,
      .subpass = {},
      .basePipelineHandle = {},
      .basePipelineIndex = {},
  };
  Context::Instance()->g_graphics_pipeline =
      vk::raii::Pipeline(Context::Instance()->g_device, nullptr, pipeline_info);

  vk::GraphicsPipelineCreateInfo lighting_pipeline_info = pipeline_info;
  vk::PipelineRenderingCreateInfo lighting_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(graphsic_formats.size()),
      .pColorAttachmentFormats = graphsic_formats.data(),
      .depthAttachmentFormat = Context::Instance()->g_depth_image_format,
  };
  vk::PipelineShaderStageCreateInfo
      lighting_pipeline_shader_stage_create_info[2] = {
          {
              .stage = vk::ShaderStageFlagBits::eVertex,
              .module = shader_module,
              .pName = "vertLighting",
              .pSpecializationInfo = nullptr,
          },
          {
              .stage = vk::ShaderStageFlagBits::eFragment,
              .module = shader_module,
              .pName = "fragLighting",
              .pSpecializationInfo = nullptr,
          },
      };
  vk::PipelineVertexInputStateCreateInfo lighting_vertex_input_info{};
  vk::PipelineInputAssemblyStateCreateInfo lighting_input_assembly_info{
      .topology = vk::PrimitiveTopology::eTriangleStrip};
  vk::PipelineRasterizationStateCreateInfo lighting_rasterization_create_info{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasConstantFactor = 1.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth = 1.0f,
  };
  vk::PipelineDepthStencilStateCreateInfo lighting_depth_stencil_info{
      .depthTestEnable = vk::False,
      .depthWriteEnable = vk::False,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False,
  };
  std::vector<vk::PushConstantRange> lighting_push_constant_range{{
      .stageFlags = vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size = sizeof(LightingPushConstants),
  }};
  vk::PipelineLayoutCreateInfo lighting_pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*Context::Instance()->g_descriptor_set_layout,
      .pushConstantRangeCount =
          static_cast<uint32_t>(lighting_push_constant_range.size()),
      .pPushConstantRanges = lighting_push_constant_range.data(),
  };
  Context::Instance()->g_lighting_pipeline_layout = vk::raii::PipelineLayout(
      Context::Instance()->g_device, lighting_pipeline_layout_info);
  lighting_pipeline_info.pNext = &lighting_pipeline_rending_info;
  lighting_pipeline_info.pStages = lighting_pipeline_shader_stage_create_info;
  lighting_pipeline_info.pVertexInputState = &lighting_vertex_input_info;
  lighting_pipeline_info.pInputAssemblyState = &lighting_input_assembly_info;
  lighting_pipeline_info.pRasterizationState =
      &lighting_rasterization_create_info;
  lighting_pipeline_info.pDepthStencilState = &lighting_depth_stencil_info;
  lighting_pipeline_info.layout =
      Context::Instance()->g_lighting_pipeline_layout;
  Context::Instance()->g_lighting_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, lighting_pipeline_info);

  vk::GraphicsPipelineCreateInfo bloom_pipeline_info = lighting_pipeline_info;
  std::vector<vk::Format> bloom_formats(Context::Instance()->g_bloom_mip_levels,
                                        Context::Instance()->g_gbuffer_format);
  vk::PipelineRenderingCreateInfo bloom_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(bloom_formats.size()),
      .pColorAttachmentFormats = bloom_formats.data(),
  };
  vk::PipelineShaderStageCreateInfo
      bloom_downsample_pipeline_shader_stage_create_info[2] = {
          {
              .stage = vk::ShaderStageFlagBits::eVertex,
              .module = shader_module,
              .pName = "vertBloomDownsample",
              .pSpecializationInfo = nullptr,
          },
          {
              .stage = vk::ShaderStageFlagBits::eFragment,
              .module = shader_module,
              .pName = "fragBloomDownsample",
              .pSpecializationInfo = nullptr,
          },
      };
  vk::PipelineShaderStageCreateInfo
      bloom_upsample_pipeline_shader_stage_create_info[2] = {
          {
              .stage = vk::ShaderStageFlagBits::eVertex,
              .module = shader_module,
              .pName = "vertBloomUpsample",
              .pSpecializationInfo = nullptr,
          },
          {
              .stage = vk::ShaderStageFlagBits::eFragment,
              .module = shader_module,
              .pName = "fragBloomUpsample",
              .pSpecializationInfo = nullptr,
          },
      };
  std::vector<vk::PipelineColorBlendAttachmentState>
      bloom_color_blend_attachments(Context::Instance()->g_bloom_mip_levels,
                                    opaque_blend_attachment);
  vk::PipelineColorBlendStateCreateInfo bloom_color_blend_info{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount =
          static_cast<uint32_t>(bloom_color_blend_attachments.size()),
      .pAttachments = bloom_color_blend_attachments.data(),
  };
  std::vector<vk::PushConstantRange> bloom_push_constant_range{{
      .stageFlags = vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size = sizeof(BloomPushConstants),
  }};
  vk::PipelineLayoutCreateInfo bloom_pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*Context::Instance()->g_descriptor_set_layout,
      .pushConstantRangeCount =
          static_cast<uint32_t>(bloom_push_constant_range.size()),
      .pPushConstantRanges = bloom_push_constant_range.data(),
  };
  Context::Instance()->g_bloom_downsample_pipeline_layout =
      vk::raii::PipelineLayout(Context::Instance()->g_device,
                               bloom_pipeline_layout_info);
  bloom_pipeline_info.pNext = &bloom_pipeline_rending_info;
  bloom_pipeline_info.pStages =
      bloom_downsample_pipeline_shader_stage_create_info;
  bloom_pipeline_info.pColorBlendState = &bloom_color_blend_info;
  bloom_pipeline_info.layout =
      Context::Instance()->g_bloom_downsample_pipeline_layout;
  Context::Instance()->g_bloom_downsample_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, bloom_pipeline_info);
  Context::Instance()->g_bloom_upsample_pipeline_layout =
      vk::raii::PipelineLayout(Context::Instance()->g_device,
                               bloom_pipeline_layout_info);
  bloom_pipeline_info.pStages =
      bloom_upsample_pipeline_shader_stage_create_info;
  bloom_pipeline_info.layout =
      Context::Instance()->g_bloom_upsample_pipeline_layout;
  Context::Instance()->g_bloom_upsample_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, bloom_pipeline_info);

  vk::GraphicsPipelineCreateInfo shodowmap_pipeline_info = pipeline_info;
  vk::PipelineRenderingCreateInfo shodowmap_pipeline_rending_info{
      .colorAttachmentCount = 0,
      .depthAttachmentFormat = Context::Instance()->g_shadowmap_image_format,
  };
  vk::PipelineShaderStageCreateInfo
      shodowmap_pipeline_shader_stage_create_info[2] = {
          {
              .stage = vk::ShaderStageFlagBits::eVertex,
              .module = shader_module,
              .pName = "vertShadowmap",
              .pSpecializationInfo = nullptr,
          },
          {
              .stage = vk::ShaderStageFlagBits::eFragment,
              .module = shader_module,
              .pName = "fragShadowmap",
              .pSpecializationInfo = nullptr,
          },
      };
  vk::PipelineRasterizationStateCreateInfo shadowmap_rasterization_create_info{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasConstantFactor = 1.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth = 1.0f,
  };
  Context::Instance()->g_shadowmap_pipeline_layout = vk::raii::PipelineLayout(
      Context::Instance()->g_device, pipeline_layout_info);
  shodowmap_pipeline_info.pNext = &shodowmap_pipeline_rending_info;
  shodowmap_pipeline_info.pStages = shodowmap_pipeline_shader_stage_create_info;
  shodowmap_pipeline_info.pRasterizationState =
      &shadowmap_rasterization_create_info;
  shodowmap_pipeline_info.layout =
      Context::Instance()->g_shadowmap_pipeline_layout;
  Context::Instance()->g_shadowmap_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, shodowmap_pipeline_info);

  vk::GraphicsPipelineCreateInfo particle_pipeline_info = pipeline_info;
  Context::Instance()->g_particle_pipeline_layout = vk::raii::PipelineLayout(
      Context::Instance()->g_device, pipeline_layout_info);
  vk::PipelineRenderingCreateInfo particle_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(graphsic_formats.size()),
      .pColorAttachmentFormats = graphsic_formats.data(),
      .depthAttachmentFormat = Context::Instance()->g_depth_image_format,
  };
  particle_pipeline_info.pNext = &particle_pipeline_rending_info;
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
  particle_pipeline_info.pStages = particle_pipeline_shader_stage_create_info;
  binding_desc = Particle::GetBindingDescription();
  auto particle_attribute_desc = Particle::GetAttributeDescription();
  vk::PipelineVertexInputStateCreateInfo particle_vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding_desc,
      .vertexAttributeDescriptionCount = particle_attribute_desc.size(),
      .pVertexAttributeDescriptions = particle_attribute_desc.data(),
  };
  particle_pipeline_info.pVertexInputState = &particle_vertex_input_info;
  input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::ePointList};
  particle_pipeline_info.pInputAssemblyState = &input_assembly_info;
  vk::PipelineDepthStencilStateCreateInfo particle_depth_stencil_info{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::False,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False,
  };
  particle_pipeline_info.pDepthStencilState = &particle_depth_stencil_info;
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
  particle_pipeline_info.pColorBlendState = &particle_color_blend_info;
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
