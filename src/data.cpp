#include "data.h"

vk::VertexInputBindingDescription Vertex::GetBindingDescription() {
  return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
}
std::array<vk::VertexInputAttributeDescription, 5>
Vertex::GetAttributeDescription() {
  return {
      vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat,
                                          offsetof(Vertex, position)},
      vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32A32Sfloat,
                                          offsetof(Vertex, roughness_f0)},
      vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32B32Sfloat,
                                          offsetof(Vertex, normal)},
      vk::VertexInputAttributeDescription{3, 0, vk::Format::eR32Sfloat,
                                          offsetof(Vertex, metallic)},
      vk::VertexInputAttributeDescription{4, 0, vk::Format::eR32G32Sfloat,
                                          offsetof(Vertex, tex_coord)}};
}
bool Vertex::operator==(const Vertex& other) const {
  return position == other.position && roughness_f0 == other.roughness_f0 &&
         normal == other.normal && metallic == other.metallic &&
         tex_coord == other.tex_coord;
}

namespace std {
size_t hash<Vertex>::operator()(Vertex const& vertex) const {
  return ((hash<glm::vec3>()(vertex.position) ^
           (hash<glm::vec3>()(vertex.roughness_f0) << 1)) >>
          1) ^
         (hash<glm::vec2>()(vertex.tex_coord) << 1);
}
}  // namespace std

vk::VertexInputBindingDescription Particle::GetBindingDescription() {
  return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
}
std::array<vk::VertexInputAttributeDescription, 3>
Particle::GetAttributeDescription() {
  return {vk::VertexInputAttributeDescription{
              0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Particle, pos)},
          vk::VertexInputAttributeDescription{
              1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Particle, v)},
          vk::VertexInputAttributeDescription{
              2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Particle, color)}};
}
