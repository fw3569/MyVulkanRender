static const float PI = 3.141592653589793;
struct VertexIn {
    float3 position;
    float4 roughness_f0;
    float3 normal;
    float metallic;
    float2 tex_coord;
};

struct Light {
    float3 pos;
    float3 intensities;
};
struct UniformBufferObject {
    float4x4 modu;
    float4x4 view;
    float4x4 proj;
    Light light;
    float3 camera_pos;
    float4x4 light_view;
    float4x4 light_proj;
    float2 shadowmap_resolution;
    float2 shadowmap_scale;
};
// [[vk::binding(1, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

Sampler2D texture;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 world_pos;
    float4 roughness_f0;
    float3 normal;
    float metallic;
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


float3 blinn_phong(float3 pos, float3 texture_color, float4 diffuse_specular, float3 normal, float ao_weight, float shadow_weight) {
    normal = normalize(normal);
    float3 to_light = ubo.light.pos - pos;
    float3 intensities_in = ubo.light.intensities / pow(length(to_light), 2);
    float3 ambient_light = 0.2 * intensities_in * ao_weight;
    float3 direct_light = 0.8 * intensities_in * shadow_weight;
    to_light = normalize(to_light);
    float3 to_camera = normalize(ubo.camera_pos - pos);
    float3 h = normalize(to_camera + to_light);
    float3 diffuse_out = (ambient_light + diffuse_specular.xyz * direct_light * max(dot(to_light, normal), 0)) * texture_color;
    float3 specular_out = (float3(1.0f) - diffuse_specular.xyz) * direct_light * pow(max(dot(h, normal), 0), diffuse_specular.w);
    float3 intensities_out = diffuse_out + specular_out;
    return intensities_out;
}

float3 cook_torrance(float3 pos, float3 texture_color, float4 roughness_f0, float3 normal, float metallic, float ao_weight, float shadow_weight) {
    normal = normalize(normal);
    float3 light_vec = ubo.light.pos - pos;
    float3 intensities_in = ubo.light.intensities / pow(length(light_vec), 2);
    float3 ambient_light_in = 0.2 * intensities_in * ao_weight;
    float3 direct_light_in = 0.8 * intensities_in * shadow_weight;
    light_vec = normalize(light_vec);
    float3 view_vec = normalize(ubo.camera_pos - pos);
    float3 h = normalize(view_vec + light_vec);
    float cosnl = dot(normal, view_vec);
    float coshv = dot(h, view_vec);
    float coshl = dot(h, light_vec);
    // fresnel-schlick
    float3 fresnel = roughness_f0.yzw + (float3(1.0f) - roughness_f0.yzw) * pow((1 - coshv), 5);
    float3 ambient_light_out = (1 - fresnel) * (1 - metallic) / PI * texture_color * ambient_light_in * 0.5;
    if (cosnl <= 0.0f) {
        return ambient_light_out;
    }
    // ggx
    float a = pow(roughness_f0.x, 4);
    float normal_distribution = a / (PI * pow(1 + pow(cosnl, 2) * (a - 1), 2));
    // schlick-ggx
    float k = pow(roughness_f0.x + 1, 2) / 8;
    float geometry_occlusion = coshv * coshl / ((coshv * (1 - k) + k) * (coshl * (1 - k) + k));
    float3 diffuse_light_out = (1 - fresnel) * (1 - metallic) / PI * texture_color * direct_light_in * cosnl;
    float3 specular_light_out = lerp(fresnel, texture_color, metallic) * normal_distribution * geometry_occlusion / (4 * coshv * coshl) * direct_light_in;
    float3 intensities_out = ambient_light_out + diffuse_light_out + specular_light_out;
    return intensities_out;
}

// generate shadow map
[shader("vertex")]
float4 vertShadowmap(VertexIn vertex_in) : SV_Position {
    float4 pos = mul(ubo.light_proj, mul(ubo.light_view, mul(ubo.modu, float4(vertex_in.position, 1.0))));
    return pos;
}
[shader("fragment")]
void fragShadowmap() {
    return;
}

[[vk::binding(9, 0)]]
Texture2D<float32_t> shadowmap;
[[vk::binding(10, 0)]]
Sampler2D<float32_t> image_depth;

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
    if (texture_color.a < 0.1) discard;
    Gbuffer gbuffer;
    gbuffer.texture_color = texture_color;
    gbuffer.position = float4(vertex.world_pos, 1.0f);
    gbuffer.normal = float4(vertex.normal, vertex.metallic);
    gbuffer.roughness_f0 = vertex.roughness_f0;
    return gbuffer;
}

[[vk::input_attachment_index(5)]] SubpassInput<float4> gbuffer_color;
[[vk::input_attachment_index(6)]] SubpassInput<float4> gbuffer_position;
[[vk::input_attachment_index(7)]] SubpassInput<float4> gbuffer_normal;
// roughness, f0.rgb
[[vk::input_attachment_index(8)]] SubpassInput<float4> gbuffer_roughness_f0;
struct LightingPushConstants {
    bool use_ssao;
}
[vk::push_constant]
ConstantBuffer<LightingPushConstants> lighting_push_constants;
// deferred shading
float random( float2 p )
{
    float2 K1 = float2(
        23.14069263277926f, // e^pi (Gelfond's constant)
        2.665144142690225f  // 2^sqrt(2) (Gelfondâ€“Schneider constant)
    );
    return fract(cos(dot(p, K1)) * 12345.6789f);
}
[shader("vertex")]
float4 vertLighting(int vid : SV_VertexID) : SV_Position {
    return float4(vid%2*2-1.0f, vid/2*2-1.0f, 0.0f, 1.0f);
}
[shader("fragment")]
float4 fragLighting(float4 pos: SV_Position) : SV_Target {
    float4 texture_color = gbuffer_color.SubpassLoad();
    if (texture_color.a < 0.1) discard;
    float3 world_pos = gbuffer_position.SubpassLoad().xyz;
    float4 normal_metallic = gbuffer_normal.SubpassLoad();
    float3 normal = normalize(normal_metallic.xyz);
    float metallic = normal_metallic.w;
    float shadow_map_weight = 1.0f;
    if (true) { // if enable shadow
        float4 light_space_pos = mul(ubo.light_proj, mul(ubo.light_view, float4(world_pos + 1e-2 * normal * distance(ubo.light.pos, world_pos), 1.0f)));
        light_space_pos = light_space_pos / float4(light_space_pos.w);
        light_space_pos.xy = (light_space_pos.xy + 1.0f) / 2.0f * ubo.shadowmap_resolution;
        if (light_space_pos.x >= 0 && light_space_pos.x < ubo.shadowmap_resolution.x && light_space_pos.y >= 0 && light_space_pos.y < ubo.shadowmap_resolution.y) {
            int2 point0 = int2(light_space_pos.xy);
            float block_light_z = 0.0f;
            for (int x = -2; x <= 2; ++x) {
                for (int y = -2; y <= 2; ++y) {
                    block_light_z += shadowmap[point0 + int2(x, y)];
                }
            }
            block_light_z /= 25.0f;
            if (block_light_z < light_space_pos.z) {
                shadow_map_weight = 0.0f;
                int pcf_size = min(max(int((light_space_pos.z - block_light_z) * 2.0f / block_light_z), 2), 4);
                for (int x = -pcf_size; x <= pcf_size; ++x) {
                    for (int y = -pcf_size + abs(x); abs(x) + abs(y) <= pcf_size; ++y) {
                        shadow_map_weight += float(shadowmap[point0 + int2(x, y)] + 1e-2 >= light_space_pos.z);
                    }
                }
                shadow_map_weight /= (pcf_size + 1) * pcf_size * 2 + 1;
            }
        }
    }
    float ssao_weight = 1.0f;
    if (lighting_push_constants.use_ssao) { // if enable ssao
        ssao_weight = 0.0f;
        float3 normal_z = normal;
        float3 normal_x;
        float3 normal_y;
        float random_num1 = world_pos.x;
        float random_num2 = world_pos.y;
        if (normal_z.z >= 1.0f - 1e-07) {
            normal_x = float3(1.0f, 0.0f, 0.0f);
            normal_y = float3(0.0f, 1.0f, 0.0f);
        } else {
            normal_x = normalize(cross(normal_z, float3(0.0f, 0.0f, 1.0f)));
            normal_y = cross(normal_z, normal_x);
        }
        static const int ssao_sample_count = 16;
        for (int i = 0; i < ssao_sample_count; ++i) {
            random_num1 = random(float2(random_num1, random_num2));
            random_num2 = random(float2(random_num1, random_num2));
            // float theta = asin(random_num1); // distribute as cosx in axis
            float theta = acos(1 - 2 * random_num1) / 2; // pdf cosx in sphere
            float phi = random_num2 * 6.283185307179586f;
            float3 sample_dir = sin(theta) * cos(phi) * normal_x + sin(theta) * sin(phi) * normal_y + cos(theta) * normal_z;
            static const float sampel_dis = 0.05f;
            float3 sample_point = world_pos + sample_dir * sampel_dis;
            float4 sample_point_ndc = mul(ubo.proj, mul(ubo.view, float4(sample_point, 1.0f)));
            sample_point_ndc /= sample_point_ndc.w;
            sample_point_ndc.xy = (sample_point_ndc.xy + 1.0f) / 2.0f;
            float depth = image_depth.Sample(sample_point_ndc.xy);
            ssao_weight += float(depth >= sample_point_ndc.z);
        }
        ssao_weight /= ssao_sample_count;
    }
    // float4 outlight = float4(blinn_phong(world_pos, texture_color.rgb, gbuffer_diffuse_specular.SubpassLoad(), normal, ssao_weight, shadow_map_weight), texture_color.a);
    float4 outlight = float4(cook_torrance(world_pos, texture_color.rgb, gbuffer_roughness_f0.SubpassLoad(), normal, metallic, ssao_weight, shadow_map_weight), texture_color.a);
    return outlight;
}

// bloom pass
[[vk::binding(11, 0)]]
Texture2D bloom_image;
struct BloomPushConstants {
    uint32_t bloom_mip_level;
    float bloom_factor;
}
[vk::push_constant]
ConstantBuffer<BloomPushConstants> bloom_push_constants;
[shader("vertex")]
float4 vertBloomDownsample(int vid : SV_VertexID) : SV_Position {
    return float4(vid%2*2-1.0f, vid/2*2-1.0f, 0.0f, 1.0f);
}
[shader("fragment")]
float4 fragBloomDownsample(float4 pos: SV_Position) : SV_Target {
    uint32_t level = bloom_push_constants.bloom_mip_level;
    float4 sum = 0.0f;
    int2 point0 = int2(pos.xy) * 2;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float4 color = bloom_image.mips[level - 1][point0 + int2(i, j)];
            if (color.r + color.g + color.b > 1.0f) {
                sum += color;
            }
        }
    }
    return sum / 4.0f;
}
[shader("vertex")]
float4 vertBloomUpsample(int vid : SV_VertexID) : SV_Position {
    return float4(vid%2*2-1.0f, vid/2*2-1.0f, 0.0f, 1.0f);
}
[shader("fragment")]
float4 fragBloomUpsample(float4 pos: SV_Position) : SV_Target {
    uint32_t level = bloom_push_constants.bloom_mip_level;
    float bloom_factor = bloom_push_constants.bloom_factor;
    float4 sum = 0.0f;
    int2 point0 = int2(pos.xy) / 2;
    int n = 1;
    static const float gaussian_blur[3][3] = {
        {0.0947416582101747, 0.1183180127031206, 0.0947416582101747},
        {0.1183180127031206, 0.14776131634681883, 0.1183180127031206},
        {0.0947416582101747, 0.1183180127031206, 0.0947416582101747}
    };
    for (int i = -n; i <= n; ++i) {
        for (int j = -n; j <= n; ++j) {
            sum += bloom_image.mips[level + 1][point0 + int2(i, j)]*gaussian_blur[i+n][j+n];
        }
    }
    return sum * bloom_factor + bloom_image.mips[level][int2(pos.xy)];
}

// compute particles
[shader("compute")]
[numthreads(256,1,1)]
void compParticle(uint3 thread_id : SV_DispatchThreadID) {
    particle_out[thread_id.x].pos = particle_in[thread_id.x].pos + particle_ubo.delta_time * particle_in[thread_id.x].v;
    particle_out[thread_id.x].color = particle_in[thread_id.x].color;
}

// draw particles
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
