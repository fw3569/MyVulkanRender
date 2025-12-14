#pragma once

#include "context.h"
#include "third_part/vulkan_headers.h"

vk::raii::CommandBuffer BeginOneTimeCommandBuffer();
void EndOneTimeCommandBuffer(vk::raii::CommandBuffer& command_buffer);
void CreateCommandPool();
void CreateCommandBuffer();
