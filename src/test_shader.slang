struct VertexIn {
    float3 position;
    float4 diffuse_specular;
    float3 normal;
    float2 tex_coord;
};

struct Light {
    float3 pos;
    float3 intensities;
};
// static const int shininess = 32;
struct UniformBufferObject {
    float4x4 modu;
    float4x4 view;
    float4x4 proj;
    Light light;
    float3 camera_pos;
};
// [[vk::binding(1, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

Sampler2D texture;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 world_pos;
    float4 diffuse_specular;
    float3 normal;
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


float3 blinn_phong(float3 pos, float3 texture_color, float4 diffuse_specular, float3 normal) {
    normal = normalize(normal);
    float3 to_light = ubo.light.pos - pos;
    float3 intensities_in = ubo.light.intensities / pow(length(to_light), 2);
    to_light = normalize(to_light);
    float3 to_camera = normalize(ubo.camera_pos - pos);
    float3 h = normalize(to_camera + to_light);
    float3 intensities_out = diffuse_specular.xyz * intensities_in * max(dot(to_light, normal), 0) + (float3(1.0f) - diffuse_specular.xyz) * intensities_in * pow(max(dot(h, normal), 0), diffuse_specular.w);
    return texture_color * intensities_out;
}

[shader("vertex")]
VertexOutput vertMain(VertexIn vertex_in) {
    VertexOutput output;
    float4 world_pos = mul(ubo.modu, float4(vertex_in.position, 1.0));
    output.sv_position = mul(ubo.proj, mul(ubo.view, world_pos));
    output.diffuse_specular = vertex_in.diffuse_specular;
    output.world_pos = world_pos.xyz;
    output.normal = mul(ubo.modu, float4(vertex_in.normal, 1.0)).xyz;
    output.tex_coord = vertex_in.tex_coord;
    return output;
}

struct Gbuffer {
    float4 texture_color : SV_Target0;
    float4 position : SV_Target1;
    float4 normal : SV_Target2;
    float4 diffuse_specular : SV_Target3;
};
[shader("fragment")]
Gbuffer fragMain(VertexOutput vertex) {
    float4 texture_color = texture.Sample(vertex.tex_coord);
    if (texture_color.a < 0.1) discard;
    // return float4(blinn_phong(vertex.world_pos,texture_color.rgb,vertex.diffuse_specular,vertex.normal),texture_color.a);
    Gbuffer gbuffer;
    gbuffer.texture_color = texture_color;
    gbuffer.position = float4(vertex.world_pos,1.0f);
    gbuffer.normal = float4(vertex.normal,1.0f);
    gbuffer.diffuse_specular = vertex.diffuse_specular;
    return gbuffer;
}

[[vk::input_attachment_index(5)]] SubpassInput<float4> gbuffer_color;
[[vk::input_attachment_index(6)]] SubpassInput<float4> gbuffer_position;
[[vk::input_attachment_index(7)]] SubpassInput<float4> gbuffer_normal;
[[vk::input_attachment_index(8)]] SubpassInput<float4> gbuffer_diffuse_specular;
[shader("vertex")]
float4 vertLighting(int vid: SV_VertexID) : SV_Position {
    return float4(vid%2*2-1.0f, vid/2*2-1.0f, 0.0f, 1.0f);
}
[shader("fragment")]
float4 fragLighting() : SV_Target {
    float4 texture_color = gbuffer_color.SubpassLoad();
    if (texture_color.a < 0.1) discard;
    return float4(blinn_phong(gbuffer_position.SubpassLoad().xyz,texture_color.rgb,gbuffer_diffuse_specular.SubpassLoad(),gbuffer_normal.SubpassLoad().xyz),texture_color.a);
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