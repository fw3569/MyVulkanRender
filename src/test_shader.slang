struct VertexIn {
    float3 position;
    float3 color;
    float2 tex_coord;
};

struct UniformBufferObject {
    float4x4 modu;
    float4x4 view;
    float4x4 proj;
};
// [[vk::binding(1, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

Sampler2D texture;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 color;
    float2 tex_coord;
};

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
ConstantBuffer<ParticleUniform> particle_ubo;

struct Particle {
    float3 pos;
    float3 v;
    float3 color;
};
StructuredBuffer<Particle> particle_in;
RWStructuredBuffer<Particle> particle_out;

[shader("vertex")]
VertexOutput vertMain(VertexIn vertex_in) {
    VertexOutput output;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.modu, float4(vertex_in.position, 1.0))));
    output.color = vertex_in.color;
    output.tex_coord = vertex_in.tex_coord;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput vertex) : SV_Target {
    return texture.Sample(vertex.tex_coord);
    // return float4(vertex.tex_coord,1.0f,1.0f);
}

[shader("compute")]
[numthreads(256,1,1)]
void compParticle(uint3 thread_id: SV_DispatchThreadID) {
    particle_out[thread_id.x].pos = particle_in[thread_id.x].pos + particle_ubo.delta_time * particle_in[thread_id.x].v;
    particle_out[thread_id.x].color = particle_in[thread_id.x].color;
}

[shader("vertex")]
PointOutput vertParticle(Particle particle, uint vid : SV_VertexID) {
    PointOutput output;
    float4 view_pos = mul(ubo.view, mul(ubo.modu, float4(particle.pos, 1.0f)));
    output.sv_position = mul(ubo.proj, view_pos);
    output.color = particle.color;
    output.size = 10.0f/length(view_pos.xyz);
    return output;
}

[shader("fragment")]
float4 fragParticle(PointInput vertex) : SV_Target {
    return float4(vertex.color, 0.5f-distance(vertex.coord,float2(0.5f)));
}