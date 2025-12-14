#pragma once

#include "context.h"
#include "third_part/vulkan_headers.h"
#include "utils.h"

vk::SampleCountFlagBits GetMaxUsableSampleCount();
void PickPhysicalDevice();
void CreateLogicalDevice();
