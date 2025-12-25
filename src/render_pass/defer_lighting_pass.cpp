#include "defer_lighting_pass.h"

#include "context.h"
#include "descriptor_set.h"
#include "memory.h"
#include "swapchain.h"
#include "utils.h"

namespace {
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
}  // namespace

void DeferLightingPass::UpdateResources() {
  CreateDepthResources();
  CreateGbufferResources();
  SwapChainManager::RegisterRecreateFunction(CreateDepthResources);
  SwapChainManager::RegisterRecreateFunction(CreateGbufferResources);
}

void DeferLightingPass::CreatePipeline(
    const vk::raii::ShaderModule& shader_module) {
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
  vk::GraphicsPipelineCreateInfo lighting_pipeline_info{
      .pNext = &lighting_pipeline_rending_info,
      .stageCount = 2,
      .pStages = lighting_pipeline_shader_stage_create_info,
      .pVertexInputState = &lighting_vertex_input_info,
      .pInputAssemblyState = &lighting_input_assembly_info,
      .pTessellationState = {},
      .pViewportState = &viewport_state_info,
      .pRasterizationState = &lighting_rasterization_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = &lighting_depth_stencil_info,
      .pColorBlendState = &color_blend_info,
      .pDynamicState = &dyanmic_state_create_info,
      .layout = Context::Instance()->g_lighting_pipeline_layout,
      .renderPass = nullptr,
      .subpass = {},
      .basePipelineHandle = {},
      .basePipelineIndex = {},
  };
  Context::Instance()->g_lighting_pipeline = vk::raii::Pipeline(
      Context::Instance()->g_device, nullptr, lighting_pipeline_info);
}

void DeferLightingPass::Draw(uint32_t image_index, uint32_t frame_index,
                             vk::Viewport viewport, vk::Rect2D scissor) {
  TransformImageLayout(Context::Instance()->g_depth_image, frame_index,
                       vk::ImageLayout::eUndefined,
                       vk::ImageLayout::eDepthStencilAttachmentOptimal, {},
                       vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                           vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                       vk::PipelineStageFlagBits2::eTopOfPipe,
                       vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                           vk::PipelineStageFlagBits2::eLateFragmentTests,
                       vk::ImageAspectFlagBits::eDepth);
  TransformImageLayout(Context::Instance()->g_bloom_image, frame_index,
                       vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                       {}, vk::AccessFlagBits2::eColorAttachmentWrite,
                       vk::PipelineStageFlagBits2::eTopOfPipe,
                       vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                       vk::ImageAspectFlagBits::eColor, 0,
                       Context::Instance()->g_bloom_mip_levels);
  // https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_dynamic_rendering_local_read.html
  // can not change attachments inside renderpass, use superset and remapping
  std::vector<vk::RenderingAttachmentInfo> attachment_infos{
      {
          .imageView = Context::Instance()->g_gbuffer_color_image_view,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          // disable msaa in defer lighting
          // .resolveMode = vk::ResolveModeFlagBits::eAverage,
          // .resolveImageView =
          //     Context::Instance()->g_swapchain_image_views[image_index],
          // .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eDontCare,
          .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
      },
      {
          .imageView = Context::Instance()->g_gbuffer_position_image_view,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eDontCare,
          .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
      },
      {
          .imageView = Context::Instance()->g_gbuffer_normal_image_view,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eDontCare,
          .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
      },
      {
          .imageView = Context::Instance()->g_gbuffer_roughness_f0_image_view,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eDontCare,
          .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
      },
      {
          .imageView = Context::Instance()->g_bloom_image_view,
          .imageLayout = vk::ImageLayout::eGeneral,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
      },
  };
  vk::RenderingAttachmentInfo depth_info{
      .imageView = Context::Instance()->g_depth_image_view,
      .imageLayout = vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .clearValue = vk::ClearDepthStencilValue{1.0f, 0},
  };
  vk::RenderingInfo rendering_info{
      .renderArea = {.offset = {0, 0},
                     .extent = Context::Instance()->g_swapchain_extent},
      .layerCount = 1,
      .colorAttachmentCount = static_cast<uint32_t>(attachment_infos.size()),
      .pColorAttachments = attachment_infos.data(),
      .pDepthAttachment = &depth_info,
  };
  // graphsic pass
  TransformImageLayout(Context::Instance()->g_shadowmap_image, frame_index,
                       vk::ImageLayout::eDepthStencilAttachmentOptimal,
                       vk::ImageLayout::eDepthReadOnlyOptimal,
                       vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                           vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                       vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                       vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                           vk::PipelineStageFlagBits2::eLateFragmentTests,
                       vk::PipelineStageFlagBits2::eFragmentShader,
                       vk::ImageAspectFlagBits::eDepth);
  Context::Instance()->g_command_buffer[frame_index].beginRendering(
      rendering_info);
  Context::Instance()->g_command_buffer[frame_index].bindPipeline(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_graphics_pipeline);
  Context::Instance()->g_command_buffer[frame_index].bindVertexBuffers(
      0, *Context::Instance()->g_vertex_buffer, {0});
  Context::Instance()->g_command_buffer[frame_index].bindIndexBuffer(
      *Context::Instance()->g_index_buffer, 0, vk::IndexType::eUint32);
  Context::Instance()->g_command_buffer[frame_index].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, Context::Instance()->g_pipeline_layout,
      0, *Context::Instance()->g_descriptor_sets[frame_index], nullptr);
  Context::Instance()->g_command_buffer[frame_index].setViewport(0, viewport);
  Context::Instance()->g_command_buffer[frame_index].setScissor(0, scissor);
  Context::Instance()->g_command_buffer[frame_index].drawIndexed(
      Context::Instance()->g_index_in.size(), 1, 0, 0, 0);
  // lighting pass
  TransformImageLayout(Context::Instance()->g_depth_image, frame_index,
                       vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal,
                       vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal,
                       vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                           vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                       vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                       vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                           vk::PipelineStageFlagBits2::eLateFragmentTests,
                       vk::PipelineStageFlagBits2::eFragmentShader,
                       vk::ImageAspectFlagBits::eDepth);
  std::vector<uint32_t> lighting_attachment_locations{
      vk::AttachmentUnused, vk::AttachmentUnused, vk::AttachmentUnused,
      vk::AttachmentUnused, 0};
  Context::Instance()
      ->g_command_buffer[frame_index]
      .setRenderingAttachmentLocations(vk::RenderingAttachmentLocationInfo{
          .colorAttachmentCount =
              static_cast<uint32_t>(lighting_attachment_locations.size()),
          .pColorAttachmentLocations = lighting_attachment_locations.data(),
      });
  Context::Instance()->g_command_buffer[frame_index].bindPipeline(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_lighting_pipeline);
  Context::Instance()->g_command_buffer[frame_index].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_lighting_pipeline_layout, 0,
      *Context::Instance()->g_descriptor_sets[frame_index], nullptr);
  LightingPushConstants lighting_push_constants{
      .enable_ssao = Context::Instance()->g_enable_ssao};
  Context::Instance()
      ->g_command_buffer[frame_index]
      .pushConstants<LightingPushConstants>(
          Context::Instance()->g_lighting_pipeline_layout,
          vk::ShaderStageFlagBits::eFragment, 0, lighting_push_constants);
  Context::Instance()->g_command_buffer[frame_index].draw(4, 1, 0, 0);
  // barrier
  std::vector<vk::ImageMemoryBarrier2> barriers = {
      {
          .srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                          vk::PipelineStageFlagBits2::eLateFragmentTests,
          .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                          vk::PipelineStageFlagBits2::eLateFragmentTests,
          .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                           vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          .oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
          .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = Context::Instance()->g_depth_image,
          .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                               .baseMipLevel = 0,
                               .levelCount = 1,
                               .baseArrayLayer = 0,
                               .layerCount = 1},
      },
  };
  vk::DependencyInfo dependency_info{
      .dependencyFlags = vk::DependencyFlagBits::eByRegion,
      .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
      .pImageMemoryBarriers = barriers.data(),
  };
  Context::Instance()->g_command_buffer[frame_index].pipelineBarrier2(
      dependency_info);
  // particle pass
  Context::Instance()->g_command_buffer[frame_index].bindPipeline(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_particle_pipeline);
  Context::Instance()->g_command_buffer[frame_index].bindVertexBuffers(
      0,
      *Context::Instance()
           ->g_particle_buffer[Context::Instance()->g_frame_in_flight - 1],
      {0});
  Context::Instance()->g_command_buffer[frame_index].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_particle_pipeline_layout, 0,
      *Context::Instance()->g_descriptor_sets[frame_index], nullptr);
  Context::Instance()->g_command_buffer[frame_index].draw(
      Context::Instance()->kParticleCount, 1, 0, 0);
  Context::Instance()->g_command_buffer[frame_index].endRendering();
}

void DeferLightingPass::UpdateDescriptorSetInfo() {
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(nullptr,
                              *Context::Instance()->g_gbuffer_color_image_view,
                              vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        5, vk::DescriptorType::eInputAttachment,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(
          nullptr, *Context::Instance()->g_gbuffer_position_image_view,
          vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        6, vk::DescriptorType::eInputAttachment,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(nullptr,
                              *Context::Instance()->g_gbuffer_normal_image_view,
                              vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        7, vk::DescriptorType::eInputAttachment,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(
          nullptr, *Context::Instance()->g_gbuffer_roughness_f0_image_view,
          vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        8, vk::DescriptorType::eInputAttachment,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(
          *Context::Instance()->g_depth_image_sampler,
          *Context::Instance()->g_depth_image_view,
          vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        10, vk::DescriptorType::eCombinedImageSampler,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
}
