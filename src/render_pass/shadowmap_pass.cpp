#include "shadowmap_pass.h"

#include "context.h"
#include "descriptor_set.h"
#include "memory.h"
#include "swapchain.h"
#include "utils.h"

namespace {
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
}  // namespace

void ShadowmapPass::UpdateResources() {
  CreateShadowmapResources();
  SwapChainManager::RegisterRecreateFunction(CreateShadowmapResources);
}

void ShadowmapPass::CreatePipeline(
    const vk::raii::ShaderModule& shader_module) {
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
  vk::PipelineViewportStateCreateInfo viewport_state_info{
      .viewportCount = 1,
      .pViewports = nullptr,
      .scissorCount = 1,
      .pScissors = nullptr,
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
  std::vector<vk::Format> graphsic_formats{
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
  };
  vk::PipelineRenderingCreateInfo shodowmap_pipeline_rending_info{
      .colorAttachmentCount = 0,
      .depthAttachmentFormat = Context::Instance()->g_shadowmap_image_format,
  };
  Context::Instance()->g_shadowmap_pipeline_layout = vk::raii::PipelineLayout(
      Context::Instance()->g_device, pipeline_layout_info);
  vk::GraphicsPipelineCreateInfo shodowmap_pipeline_info{
      .pNext = &shodowmap_pipeline_rending_info,
      .stageCount = 2,
      .pStages = shodowmap_pipeline_shader_stage_create_info,
      .pVertexInputState = &vertex_input_info,
      .pInputAssemblyState = &input_assembly_info,
      .pTessellationState = {},
      .pViewportState = &viewport_state_info,
      .pRasterizationState = &shadowmap_rasterization_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = &depth_stencil_info,
      .pColorBlendState = &color_blend_info,
      .pDynamicState = &dyanmic_state_create_info,
      .layout = Context::Instance()->g_shadowmap_pipeline_layout,
      .renderPass = nullptr,
      .subpass = {},
      .basePipelineHandle = {},
      .basePipelineIndex = {},
  };

  Context::Instance()->g_shadowmap_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, shodowmap_pipeline_info);
}

void ShadowmapPass::Draw(uint32_t image_index, uint32_t frame_index,
                         vk::Viewport viewport, vk::Rect2D scissor) {
  // shadowmap pass
  TransformImageLayout(Context::Instance()->g_shadowmap_image, frame_index,
                       vk::ImageLayout::eUndefined,
                       vk::ImageLayout::eDepthStencilAttachmentOptimal, {},
                       vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                           vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                       vk::PipelineStageFlagBits2::eTopOfPipe,
                       vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                           vk::PipelineStageFlagBits2::eLateFragmentTests,
                       vk::ImageAspectFlagBits::eDepth);
  vk::RenderingAttachmentInfo shadowmap_depth_info{
      .imageView = Context::Instance()->g_shadowmap_image_view,
      .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = vk::ClearDepthStencilValue{1.0f, 0},
  };
  vk::RenderingInfo shadowmap_rendering_info{
      .renderArea = {.offset = {0, 0},
                     .extent = {Context::Instance()->g_shadowmap_width,
                                Context::Instance()->g_shadowmap_height}},
      .layerCount = 1,
      .colorAttachmentCount = 0,
      .pColorAttachments = nullptr,
      .pDepthAttachment = &shadowmap_depth_info,
  };
  vk::Viewport shadowmap_viewport = vk::Viewport(
      0.0f, 0.0f, static_cast<float>(Context::Instance()->g_shadowmap_width),
      static_cast<float>(Context::Instance()->g_shadowmap_height), 0.0f, 1.0f);
  vk::Rect2D shadowmap_scissor =
      vk::Rect2D({0, 0}, {Context::Instance()->g_shadowmap_width,
                          Context::Instance()->g_shadowmap_height});
  Context::Instance()->g_command_buffer[frame_index].beginRendering(
      shadowmap_rendering_info);
  Context::Instance()->g_command_buffer[frame_index].bindPipeline(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_shadowmap_pipeline);
  Context::Instance()->g_command_buffer[frame_index].bindVertexBuffers(
      0, *Context::Instance()->g_vertex_buffer, {0});
  Context::Instance()->g_command_buffer[frame_index].bindIndexBuffer(
      *Context::Instance()->g_index_buffer, 0, vk::IndexType::eUint32);
  Context::Instance()->g_command_buffer[frame_index].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_shadowmap_pipeline_layout, 0,
      *Context::Instance()->g_descriptor_sets[frame_index], nullptr);
  Context::Instance()->g_command_buffer[frame_index].setViewport(
      0, shadowmap_viewport);
  Context::Instance()->g_command_buffer[frame_index].setScissor(
      0, shadowmap_scissor);
  Context::Instance()->g_command_buffer[frame_index].drawIndexed(
      Context::Instance()->g_index_in.size(), 1, 0, 0, 0);
  Context::Instance()->g_command_buffer[frame_index].endRendering();
}

void ShadowmapPass::UpdateDescriptorSetInfo() {
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(nullptr,
                              *Context::Instance()->g_shadowmap_image_view,
                              vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        9, vk::DescriptorType::eSampledImage,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
}
