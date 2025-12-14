#pragma once

#include "third_part/glm_headers.h"
#include "third_part/vulkan_headers.h"

struct Vertex {
  alignas(16) glm::vec3 position;
  alignas(16) glm::vec4 roughness_f0;
  alignas(16) glm::vec3 normal;
  float metallic;
  glm::vec2 tex_coord;

  static vk::VertexInputBindingDescription GetBindingDescription();
  static std::array<vk::VertexInputAttributeDescription, 5>
  GetAttributeDescription();
  bool operator==(const Vertex& other) const;
};

struct UniformBufferObject {
  alignas(16) glm::mat4 modu;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
  struct Light {
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 intensities;
  };
  alignas(16) Light light;
  alignas(16) glm::vec3 camera_pos;
  alignas(16) glm::mat4 light_view;
  alignas(16) glm::mat4 light_proj;
  alignas(8) glm::vec2 shadowmap_resolution;
  alignas(8) glm::vec2 shadowmap_scale;
};
// push_constants size should be multiple of 4
struct LightingPushConstants {
  int32_t enable_ssao;
};

struct BloomPushConstants {
  uint32_t bloom_mip_level;
  float bloom_factor;
};

struct Particle {
  alignas(16) glm::vec3 pos;
  alignas(16) glm::vec3 v;
  alignas(16) glm::vec3 color;
  static vk::VertexInputBindingDescription GetBindingDescription();
  static std::array<vk::VertexInputAttributeDescription, 3>
  GetAttributeDescription();
};
struct ParticleUbo {
  float delta_time = 0;
};

namespace std {
template <>
struct hash<Vertex> {
  size_t operator()(Vertex const& vertex) const;
};
}  // namespace std