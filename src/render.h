#pragma once

#include <cstdint>

#include "context.h"
#include "third_part/vulkan_headers.h"

void RecordCommandBuffer(
    uint32_t image_index, uint32_t frame_index,
    vk::Viewport viewport = vk::Viewport(
        0.0f, 0.0f,
        static_cast<float>(Context::Instance()->g_swapchain_extent.width),
        static_cast<float>(Context::Instance()->g_swapchain_extent.height),
        0.0f, 1.0f),
    vk::Rect2D scissor = vk::Rect2D({0, 0},
                                    Context::Instance()->g_swapchain_extent));
void CreateSyncObjects();
