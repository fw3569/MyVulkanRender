const static float2 position[3] ={
  {0.0,-0.5},
  {0.5,0.5},
  {-0.5,0.5}
};

const static float3 colors[3] ={
  {1.0,0.0,0.0},
  {0.0,1.0,0.0},
  {0.0,0.0,1.0}
};

struct VertexOutput {
    float3 color : Color;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vertMain(uint vid: SV_VertexID) {
    VertexOutput output;
    output.color = colors[vid];
    output.sv_position = float4(position[vid],0.0,1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(float3 color :Color) :SV_Target {
    return float4(color,1.0);
}