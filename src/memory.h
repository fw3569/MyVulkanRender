#pragma once

#include <cstdint>

#include "command_buffer.h"
#include "context.h"
#include "third_part/vulkan_headers.h"

uint32_t FindMemoryType(uint32_t type_filter,
                        vk::MemoryPropertyFlags properties);
void CreateBuffer(uint32_t size, vk::BufferUsageFlags usage,
                  vk::SharingMode sharing_mode,
                  vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer,
                  vk::raii::DeviceMemory& memory);
void CopyBuffer(vk::raii::Buffer& src_buffer, vk::raii::Buffer& dst_buffer,
                uint32_t size);
void CreateImage(uint32_t width, uint32_t height, uint32_t mip_levels,
                 vk::SampleCountFlagBits sample_count, vk::Format format,
                 vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                 vk::MemoryPropertyFlags properties, vk::raii::Image& image,
                 vk::raii::DeviceMemory& memory);
void CopyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image,
                       uint32_t width, uint32_t height);
vk::raii::ImageView CreateImageView(const vk::Image& image,
                                    uint32_t base_mip_level,
                                    uint32_t mip_levels, vk::Format format,
                                    vk::ImageAspectFlagBits aspect);
void TransformImageLayout(
    const vk::Image& image, uint32_t command_buffer_index,
    vk::ImageLayout old_layout, vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor,
    uint32_t base_mip_level = 0, uint32_t level_count = 1);
void TransformImageLayoutImmediately(
    const vk::raii::Image& image, vk::ImageLayout old_layout,
    vk::ImageLayout new_layout, vk::AccessFlags src_access_mask,
    vk::AccessFlags dst_access_mask, vk::PipelineStageFlags src_stage_mask,
    vk::PipelineStageFlags dst_stage_mask,
    vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor,
    uint32_t base_mip_level = 0, uint32_t level_count = 1);
