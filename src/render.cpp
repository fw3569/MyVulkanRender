#include "render.h"

#include "context.h"
#include "memory.h"

void RecordCommandBuffer(uint32_t image_index, uint32_t frame_index,
                         vk::Viewport viewport, vk::Rect2D scissor) {
  Context::Instance()->g_command_buffer[frame_index].begin({});
  TransformImageLayout(Context::Instance()->g_swapchain_images[image_index],
                       frame_index, vk::ImageLayout::eUndefined,
                       vk::ImageLayout::eTransferDstOptimal, {},
                       vk::AccessFlagBits2::eTransferWrite,
                       vk::PipelineStageFlagBits2::eTopOfPipe,
                       vk::PipelineStageFlagBits2::eTransfer);
  TransformImageLayout(Context::Instance()->g_color_image, frame_index,
                       vk::ImageLayout::eUndefined,
                       vk::ImageLayout::eColorAttachmentOptimal, {},
                       vk::AccessFlagBits2::eColorAttachmentWrite,
                       vk::PipelineStageFlagBits2::eTopOfPipe,
                       vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                       vk::ImageAspectFlagBits::eColor);
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
      // {
      //   .imageView = Context::Instance()->g_color_image_view,
      //   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      //   .resolveMode = vk::ResolveModeFlagBits::eAverage,
      //   .resolveImageView =
      //   Context::Instance()->g_swapchain_image_views[image_index],
      //   .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      //   .loadOp = vk::AttachmentLoadOp::eClear,
      //   .storeOp = vk::AttachmentStoreOp::eStore,
      //   .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
      // },
      {
          .imageView = Context::Instance()->g_gbuffer_color_image_view,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
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
  vk::Viewport shadermap_viewport = vk::Viewport(
      0.0f, 0.0f, static_cast<float>(Context::Instance()->g_shadowmap_width),
      static_cast<float>(Context::Instance()->g_shadowmap_height), 0.0f, 1.0f);
  vk::Rect2D shadermap_scissor =
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
      0, shadermap_viewport);
  Context::Instance()->g_command_buffer[frame_index].setScissor(
      0, shadermap_scissor);
  Context::Instance()->g_command_buffer[frame_index].drawIndexed(
      Context::Instance()->g_index_in.size(), 1, 0, 0, 0);
  Context::Instance()->g_command_buffer[frame_index].endRendering();
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
  if (Context::Instance()->g_enable_bloom) {
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
    vk::RenderingInfo bloom_rendering_info = rendering_info;
    bloom_rendering_info.colorAttachmentCount =
        static_cast<uint32_t>(bloom_attachment_infos.size());
    bloom_rendering_info.pColorAttachments = bloom_attachment_infos.data();
    bloom_rendering_info.pDepthAttachment = nullptr;
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

  bloom_barrier.subresourceRange.baseMipLevel = 0;
  bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
  bloom_barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
  bloom_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
  bloom_barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
  Context::Instance()->g_command_buffer[frame_index].pipelineBarrier(
      vk::PipelineStageFlagBits::eFragmentShader,
      vk::PipelineStageFlagBits::eTransfer, {}, {}, nullptr, bloom_barrier);
  vk::ImageBlit image_blit{
      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .srcOffsets =
          std::array{
              vk::Offset3D{0, 0, 0},
              vk::Offset3D{static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.width),
                           static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.height),
                           1},
          },
      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .dstOffsets =
          std::array{
              vk::Offset3D{0, 0, 0},
              vk::Offset3D{static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.width),
                           static_cast<int32_t>(
                               Context::Instance()->g_swapchain_extent.height),
                           1},
          },
  };
  Context::Instance()->g_command_buffer[frame_index].blitImage(
      Context::Instance()->g_bloom_image, vk::ImageLayout::eTransferSrcOptimal,
      Context::Instance()->g_swapchain_images[image_index],
      vk::ImageLayout::eTransferDstOptimal, image_blit, vk::Filter::eLinear);

  TransformImageLayout(Context::Instance()->g_swapchain_images[image_index],
                       frame_index, vk::ImageLayout::eTransferDstOptimal,
                       vk::ImageLayout::eColorAttachmentOptimal,
                       vk::AccessFlagBits2::eTransferWrite,
                       vk::AccessFlagBits2::eColorAttachmentWrite,
                       vk::PipelineStageFlagBits2::eTransfer,
                       vk::PipelineStageFlagBits2::eColorAttachmentOutput);
  std::vector<vk::RenderingAttachmentInfo> imgui_attachment_infos{{
      .imageView = Context::Instance()->g_swapchain_image_views[image_index],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eLoad,
      .storeOp = vk::AttachmentStoreOp::eStore,
  }};
  vk::RenderingInfo imgui_rendering_info = rendering_info;
  imgui_rendering_info.colorAttachmentCount =
      static_cast<uint32_t>(imgui_attachment_infos.size());
  imgui_rendering_info.pColorAttachments = imgui_attachment_infos.data();
  imgui_rendering_info.pDepthAttachment = nullptr;
  Context::Instance()->g_command_buffer[frame_index].beginRendering(
      imgui_rendering_info);
  ImGui_ImplVulkan_RenderDrawData(
      ImGui::GetDrawData(),
      *Context::Instance()->g_command_buffer[frame_index]);
  Context::Instance()->g_command_buffer[frame_index].endRendering();

  TransformImageLayout(Context::Instance()->g_swapchain_images[image_index],
                       frame_index, vk::ImageLayout::eColorAttachmentOptimal,
                       vk::ImageLayout::ePresentSrcKHR,
                       vk::AccessFlagBits2::eColorAttachmentWrite, {},
                       vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                       vk::PipelineStageFlagBits2::eBottomOfPipe);
  Context::Instance()->g_command_buffer[frame_index].end();
}

void CreateSyncObjects() {
  for (uint32_t i = 0; i < Context::Instance()->g_frame_in_flight; ++i) {
    Context::Instance()->g_present_complete_semaphore.emplace_back(
        vk::raii::Semaphore(Context::Instance()->g_device,
                            vk::SemaphoreCreateInfo()));
    Context::Instance()->g_render_finished_semaphore.emplace_back(
        vk::raii::Semaphore(Context::Instance()->g_device,
                            vk::SemaphoreCreateInfo()));
    Context::Instance()->g_draw_fence.emplace_back(
        vk::raii::Fence(Context::Instance()->g_device,
                        {.flags = vk::FenceCreateFlagBits::eSignaled}));
  }
  vk::SemaphoreTypeCreateInfo timeline_type_info{
      .semaphoreType = vk::SemaphoreType::eTimeline,
      .initialValue = 0,
  };
  Context::Instance()->g_particle_compute_semaphore = vk::raii::Semaphore(
      Context::Instance()->g_device, {.pNext = &timeline_type_info});
}
