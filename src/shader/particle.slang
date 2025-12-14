#include "global_data.slangh"

struct PointOutput {
  float4 sv_position : SV_Position;
  float3 color : Color;
  float size : SV_PointSize;
};
struct PointInput {
  float4 sv_position : SV_Position;
  float3 color : Color;
  float2 coord : SV_PointCoord;
};

struct ParticleUniform {
  float delta_time;
}
[[vk::binding(2, 0)]]
ConstantBuffer<ParticleUniform> particle_ubo;

struct Particle {
  float3 pos;
  float3 v;
  float3 color;
};
[[vk::binding(3, 0)]]
StructuredBuffer<Particle> particle_in;
[[vk::binding(4, 0)]]
RWStructuredBuffer<Particle> particle_out;

// compute particles
[shader("compute")]
[numthreads(256, 1, 1)]
void compParticle(uint3 thread_id: SV_DispatchThreadID) {
  particle_out[thread_id.x].pos =
      particle_in[thread_id.x].pos +
      particle_ubo.delta_time * particle_in[thread_id.x].v;
  particle_out[thread_id.x].color = particle_in[thread_id.x].color;
}

// draw particles
[shader("vertex")]
PointOutput vertParticle(Particle particle, uint vid: SV_VertexID) {
  PointOutput output;
  float4 view_pos = mul(ubo.view, mul(ubo.modu, float4(particle.pos, 1.0f)));
  output.sv_position = mul(ubo.proj, view_pos);
  output.color = particle.color;
  output.size = 10.0f / length(view_pos.xyz);
  return output;
}

[shader("fragment")]
float4 fragParticle(PointInput vertex) : SV_Target {
  return float4(vertex.color, 0.5f - distance(vertex.coord, float2(0.5f)));
}
