#include "bloom_pass.h"

#include "context.h"
#include "descriptor_set.h"
#include "memory.h"
#include "swapchain.h"
#include "utils.h"

namespace {
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
}  // namespace

void BloomPass::UpdateResources() {
  CreateBloomResources();
  SwapChainManager::RegisterRecreateFunction(CreateBloomResources);
}

void BloomPass::CreatePipeline(const vk::raii::ShaderModule& shader_module) {
  std::vector dynamic_states = {vk::DynamicState::eViewport,
                                vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dyanmic_state_create_info = {
      .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
      .pDynamicStates = dynamic_states.data(),
  };
  vk::PipelineViewportStateCreateInfo viewport_state_info{
      .viewportCount = 1,
      .pViewports = nullptr,
      .scissorCount = 1,
      .pScissors = nullptr,
  };
  vk::PipelineMultisampleStateCreateInfo multisample_create_info{
      .rasterizationSamples = Context::Instance()->g_msaa_samples,
      .sampleShadingEnable = vk::False,
  };
  vk::PipelineColorBlendAttachmentState opaque_blend_attachment{
      .blendEnable = vk::False,
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  std::vector<vk::Format> graphsic_formats{
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
      Context::Instance()->g_gbuffer_format,
  };
  vk::PipelineVertexInputStateCreateInfo bloom_vertex_input_info{};
  vk::PipelineInputAssemblyStateCreateInfo bloom_input_assembly_info{
      .topology = vk::PrimitiveTopology::eTriangleStrip};
  vk::PipelineRasterizationStateCreateInfo bloom_rasterization_create_info{
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
  vk::PipelineDepthStencilStateCreateInfo bloom_depth_stencil_info{
      .depthTestEnable = vk::False,
      .depthWriteEnable = vk::False,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False,
  };
  std::vector<vk::Format> bloom_formats(Context::Instance()->g_bloom_mip_levels,
                                        Context::Instance()->g_gbuffer_format);
  vk::PipelineRenderingCreateInfo bloom_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(bloom_formats.size()),
      .pColorAttachmentFormats = bloom_formats.data(),
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
  vk::GraphicsPipelineCreateInfo bloom_pipeline_info{
      .pNext = &bloom_pipeline_rending_info,
      .stageCount = 2,
      .pStages = nullptr,
      .pVertexInputState = &bloom_vertex_input_info,
      .pInputAssemblyState = &bloom_input_assembly_info,
      .pTessellationState = {},
      .pViewportState = &viewport_state_info,
      .pRasterizationState = &bloom_rasterization_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = &bloom_depth_stencil_info,
      .pColorBlendState = &bloom_color_blend_info,
      .pDynamicState = &dyanmic_state_create_info,
      .layout = nullptr,
      .renderPass = nullptr,
      .subpass = {},
      .basePipelineHandle = {},
      .basePipelineIndex = {},
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
  bloom_pipeline_info.pStages =
      bloom_downsample_pipeline_shader_stage_create_info;
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
}

void BloomPass::Draw(uint32_t image_index, uint32_t frame_index,
                     vk::Viewport viewport, vk::Rect2D scissor) {
  // bloom pass
  vk::ImageMemoryBarrier bloom_barrier{
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = Context::Instance()->g_bloom_image,
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
  };
  bloom_barrier.subresourceRange.baseMipLevel = 0;
  bloom_barrier.subresourceRange.levelCount =
      Context::Instance()->g_bloom_mip_levels;
  bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
  bloom_barrier.newLayout = vk::ImageLayout::eGeneral;
  bloom_barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
  bloom_barrier.dstAccessMask =
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
  Context::Instance()->g_command_buffer[frame_index].pipelineBarrier(
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::PipelineStageFlagBits::eFragmentShader, {}, {}, nullptr,
      bloom_barrier);
  bloom_barrier.subresourceRange.levelCount = 1;
  std::vector<vk::RenderingAttachmentInfo> bloom_attachment_infos;
  for (int i = 0; i < Context::Instance()->g_bloom_mip_levels; ++i) {
    bloom_attachment_infos.emplace_back(vk::RenderingAttachmentInfo{
        .imageView = Context::Instance()->g_bloom_image_views[i],
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}});
  }

  vk::RenderingInfo bloom_rendering_info{
      .renderArea = {.offset = {0, 0},
                     .extent = Context::Instance()->g_swapchain_extent},
      .layerCount = 1,
      .colorAttachmentCount =
          static_cast<uint32_t>(bloom_attachment_infos.size()),
      .pColorAttachments = bloom_attachment_infos.data(),
      .pDepthAttachment = nullptr,
  };
  Context::Instance()->g_command_buffer[frame_index].beginRendering(
      bloom_rendering_info);
  Context::Instance()->g_command_buffer[frame_index].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_bloom_upsample_pipeline_layout, 0,
      *Context::Instance()->g_descriptor_sets[frame_index], nullptr);
  vk::Viewport bloom_viewport = viewport;
  Context::Instance()->g_command_buffer[frame_index].setScissor(0, scissor);
  std::vector<uint32_t> bloom_attachment_locations(
      Context::Instance()->g_bloom_mip_levels, vk::AttachmentUnused);
  int32_t mip_width = Context::Instance()->g_swapchain_extent.width,
          mip_height = Context::Instance()->g_swapchain_extent.height;
  std::vector<int32_t> bloom_widths, bloom_heights;
  bloom_widths.emplace_back(mip_width);
  bloom_heights.emplace_back(mip_height);
  Context::Instance()->g_command_buffer[frame_index].bindPipeline(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_bloom_downsample_pipeline);
  BloomPushConstants bloom_push_constants{.bloom_mip_level = 0,
                                          .bloom_factor = 0.0f};
  for (uint32_t i = 1; i < Context::Instance()->g_bloom_mip_levels; ++i) {
    mip_width = mip_width > 1 ? mip_width / 2 : 1;
    mip_height = mip_height > 1 ? mip_height / 2 : 1;
    bloom_widths.emplace_back(mip_width);
    bloom_heights.emplace_back(mip_height);
    bloom_viewport.width = mip_width;
    bloom_viewport.height = mip_height;
    Context::Instance()->g_command_buffer[frame_index].setViewport(
        0, bloom_viewport);
    bloom_push_constants.bloom_mip_level = i;
    Context::Instance()
        ->g_command_buffer[frame_index]
        .pushConstants<BloomPushConstants>(
            Context::Instance()->g_bloom_upsample_pipeline_layout,
            vk::ShaderStageFlagBits::eFragment, 0, bloom_push_constants);
    bloom_attachment_locations[i] = 0;
    Context::Instance()
        ->g_command_buffer[frame_index]
        .setRenderingAttachmentLocations(vk::RenderingAttachmentLocationInfo{
            .colorAttachmentCount =
                static_cast<uint32_t>(bloom_attachment_locations.size()),
            .pColorAttachmentLocations = bloom_attachment_locations.data(),
        });
    Context::Instance()->g_command_buffer[frame_index].draw(4, 1, 0, 0);
    bloom_attachment_locations[i] = vk::AttachmentUnused;
    bloom_barrier.subresourceRange.baseMipLevel = i;
    bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
    bloom_barrier.newLayout = vk::ImageLayout::eGeneral;
    bloom_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    bloom_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    Context::Instance()->g_command_buffer[frame_index].pipelineBarrier(
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::DependencyFlagBits::eByRegion, {}, nullptr, bloom_barrier);
  }
  Context::Instance()->g_command_buffer[frame_index].bindPipeline(
      vk::PipelineBindPoint::eGraphics,
      Context::Instance()->g_bloom_upsample_pipeline);
  bloom_push_constants.bloom_factor =
      Context::Instance()->kBloomRate / (Context::Instance()->kBloomRate + 1);
  for (int i = Context::Instance()->g_bloom_mip_levels - 2; i >= 0; --i) {
    bloom_viewport.width = bloom_widths[i];
    bloom_viewport.height = bloom_heights[i];
    Context::Instance()->g_command_buffer[frame_index].setViewport(
        0, bloom_viewport);
    bloom_push_constants.bloom_mip_level = i;
    Context::Instance()
        ->g_command_buffer[frame_index]
        .pushConstants<BloomPushConstants>(
            Context::Instance()->g_bloom_upsample_pipeline_layout,
            vk::ShaderStageFlagBits::eFragment, 0, bloom_push_constants);
    bloom_attachment_locations[i] = 0;
    Context::Instance()
        ->g_command_buffer[frame_index]
        .setRenderingAttachmentLocations(vk::RenderingAttachmentLocationInfo{
            .colorAttachmentCount =
                static_cast<uint32_t>(bloom_attachment_locations.size()),
            .pColorAttachmentLocations = bloom_attachment_locations.data(),
        });
    Context::Instance()->g_command_buffer[frame_index].draw(4, 1, 0, 0);
    bloom_attachment_locations[i] = vk::AttachmentUnused;
    bloom_barrier.subresourceRange.baseMipLevel = i;
    bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
    bloom_barrier.newLayout = vk::ImageLayout::eGeneral;
    bloom_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    bloom_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    Context::Instance()->g_command_buffer[frame_index].pipelineBarrier(
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::DependencyFlagBits::eByRegion, {}, nullptr, bloom_barrier);
  }
  Context::Instance()->g_command_buffer[frame_index].endRendering();
}

void BloomPass::UpdateDescriptorSetInfo() {
  {
    std::vector<vk::DescriptorImageInfo> image_info;
    for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
      image_info.emplace_back(nullptr, *Context::Instance()->g_bloom_image_view,
                              vk::ImageLayout::eGeneral);
    }
    DescriptorSetManager::RegisterDescriptorSetInfo(
        11, vk::DescriptorType::eSampledImage,
        vk::ShaderStageFlagBits::eFragment, image_info, {});
  }
}
