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