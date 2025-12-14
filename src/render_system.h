#pragma once

#include <vector>

#include "third_part/vulkan_headers.h"

void CreateUboBuffer();
void CreateParticleResources();
vk::Format FindSupportFormat(const std::vector<vk::Format>& candidates,
                             vk::ImageTiling tiling,
                             vk::FormatFeatureFlags flags);
vk::Format FindSupportDepthFormat();
void CreateColorResources();
void CreateDepthResources();
void CreateShadowmapResources();
void CreateGbufferResources();
void CreateBloomResources();
