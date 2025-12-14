#pragma once

#include <cstdint>

#include "third_part/vulkan_headers.h"

void GenerateMipmaps(const vk::raii::Image& image, vk::Format format,
                     int32_t width, int32_t height, uint32_t mip_levels);
void CreateTextureImage();
void CreateTextureSampler();
void LoadMesh();
void UpdateMesh();
void CreateVertexBuffer();
void CreateIndexBuffer();
