#include "global_data.slangh"

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
float4 vertBloomDownsample(int vid: SV_VertexID) : SV_Position {
  return float4(vid % 2 * 2 - 1.0f, vid / 2 * 2 - 1.0f, 0.0f, 1.0f);
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
float4 vertBloomUpsample(int vid: SV_VertexID) : SV_Position {
  return float4(vid % 2 * 2 - 1.0f, vid / 2 * 2 - 1.0f, 0.0f, 1.0f);
}
[shader("fragment")]
float4 fragBloomUpsample(float4 pos: SV_Position) : SV_Target {
  uint32_t level = bloom_push_constants.bloom_mip_level;
  float bloom_factor = bloom_push_constants.bloom_factor;
  float4 sum = 0.0f;
  int2 point0 = int2(pos.xy) / 2;
  int n = 1;
  static const float gaussian_blur[3][3] = {
    { 0.0947416582101747, 0.1183180127031206, 0.0947416582101747 },
    { 0.1183180127031206, 0.14776131634681883, 0.1183180127031206 },
    { 0.0947416582101747, 0.1183180127031206, 0.0947416582101747 }
  };
  for (int i = -n; i <= n; ++i) {
    for (int j = -n; j <= n; ++j) {
      sum += bloom_image.mips[level + 1][point0 + int2(i, j)] *
             gaussian_blur[i + n][j + n];
    }
  }
  return sum * bloom_factor + bloom_image.mips[level][int2(pos.xy)];
}
