#include "global_data.slangh"

// generate shadow map
[shader("vertex")]
float4 vertShadowmap(VertexIn vertex_in) : SV_Position {
  float4 pos =
      mul(ubo.light_proj,
          mul(ubo.light_view, mul(ubo.modu, float4(vertex_in.position, 1.0))));
  return pos;
}
[shader("fragment")]
void fragShadowmap() {
  return;
}
