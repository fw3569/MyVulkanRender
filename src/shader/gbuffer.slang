#include "global_data.slangh"

[[vk::binding(1, 0)]]
Sampler2D texture;

struct VertexOutput {
  float4 sv_position : SV_Position;
  float3 world_pos;
  float4 roughness_f0;
  float3 normal;
  float metallic;
  float2 tex_coord;
};

// generate gbuffer
[shader("vertex")]
VertexOutput vertMain(VertexIn vertex_in) {
  VertexOutput output;
  float4 world_pos = mul(ubo.modu, float4(vertex_in.position, 1.0));
  output.sv_position = mul(ubo.proj, mul(ubo.view, world_pos));
  output.roughness_f0 = vertex_in.roughness_f0;
  output.world_pos = world_pos.xyz;
  output.normal = mul(ubo.modu, float4(vertex_in.normal, 1.0)).xyz;
  output.metallic = vertex_in.metallic;
  output.tex_coord = vertex_in.tex_coord;
  return output;
}

struct Gbuffer {
  float4 texture_color : SV_Target0;
  float4 position : SV_Target1;
  // normal and metallic
  float4 normal : SV_Target2;
  float4 roughness_f0 : SV_Target3;
};
[shader("fragment")]
Gbuffer fragMain(VertexOutput vertex) {
  float4 texture_color = texture.Sample(vertex.tex_coord);
  if (texture_color.a < 0.1)
    discard;
  Gbuffer gbuffer;
  gbuffer.texture_color = texture_color;
  gbuffer.position = float4(vertex.world_pos, 1.0f);
  gbuffer.normal = float4(vertex.normal, vertex.metallic);
  gbuffer.roughness_f0 = vertex.roughness_f0;
  return gbuffer;
}
