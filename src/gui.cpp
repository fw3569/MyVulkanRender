#include "gui.h"

#include "context.h"

namespace {
void FramebufferSizeCallback(GLFWwindow* window, int /* width */,
                             int /* height */) {
  Context* context_ptr = (Context*)glfwGetWindowUserPointer(window);
  std::lock_guard lock(context_ptr->g_window_resized_mtx);
  context_ptr->g_window_resized = true;
}
}  // namespace

void Gui::InitWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  Context::Instance()->g_window = glfwCreateWindow(
      Context::Instance()->kWindowWeight, Context::Instance()->kWindowHeight,
      "Vulkan", nullptr, nullptr);
  glfwSetWindowUserPointer(Context::Instance()->g_window, Context::Instance());
  glfwSetFramebufferSizeCallback(Context::Instance()->g_window,
                                 FramebufferSizeCallback);
}

void Gui::CreateaSurface() {
  VkSurfaceKHR surface;
  if (glfwCreateWindowSurface(*Context::Instance()->g_vk_instance,
                              Context::Instance()->g_window, nullptr,
                              &surface) == 0) {
    Context::Instance()->g_surface =
        vk::raii::SurfaceKHR(Context::Instance()->g_vk_instance, surface);
  } else {
    throw std::runtime_error("failed to create window surface!");
  }
}

void Gui::InitImGui() {
  Context::Instance()->g_imgui_context = ImGui::CreateContext();
  vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eSampler, 1000},
      {vk::DescriptorType::eCombinedImageSampler, 1000},
      {vk::DescriptorType::eSampledImage, 1000},
      {vk::DescriptorType::eStorageImage, 1000},
      {vk::DescriptorType::eUniformTexelBuffer, 1000},
      {vk::DescriptorType::eStorageTexelBuffer, 1000},
      {vk::DescriptorType::eUniformBuffer, 1000},
      {vk::DescriptorType::eStorageBuffer, 1000},
      {vk::DescriptorType::eUniformBufferDynamic, 1000},
      {vk::DescriptorType::eStorageBufferDynamic, 1000},
      {vk::DescriptorType::eInputAttachment, 100}};
  vk::DescriptorPoolCreateInfo pool_info = {
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = 1000,
      .poolSizeCount = (uint32_t)std::size(pool_sizes),
      .pPoolSizes = pool_sizes};
  Context::Instance()->g_imgui_pool =
      vk::raii::DescriptorPool(Context::Instance()->g_device, pool_info);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = *Context::Instance()->g_vk_instance;
  init_info.PhysicalDevice = *Context::Instance()->g_physical_device;
  init_info.Device = *Context::Instance()->g_device;
  init_info.Queue = *Context::Instance()->g_queue;
  init_info.DescriptorPool = *Context::Instance()->g_imgui_pool;
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.UseDynamicRendering = true;
  init_info.PipelineRenderingCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  VkFormat format =
      static_cast<VkFormat>(Context::Instance()->g_swapchain_image_format);
  init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &format;
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&init_info);
  ImGui_ImplVulkan_CreateFontsTexture();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForVulkan(Context::Instance()->g_window, true);
}
void Gui::Cleanup() {
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(Context::Instance()->g_window);
  glfwTerminate();
}

bool Gui::Closed() {
  return glfwWindowShouldClose(Context::Instance()->g_window);
}

void Gui::Update() {
  glfwPollEvents();

  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGui::Begin("Variables");
  if (ImGui::BeginTable("VariablesTable", 2)) {
    ImGui::TableSetupColumn("##Col0", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Roughness:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::SliderFloat("##RoughnessSlider",
                       &Context::Instance()->g_pbr_roughness, 0.0f, 1.0f,
                       "%.2f");
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("F0:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::SliderFloat("##F0Slider", &Context::Instance()->g_pbr_f0, 0.0f, 1.0f,
                       "%.2f");
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Metallic:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::SliderFloat("##MetallicSlider", &Context::Instance()->g_pbr_metallic,
                       0.0f, 1.0f, "%.2f");
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Light:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::SliderFloat("##LightSlider", &Context::Instance()->g_light_intensity,
                       0.0f, 20.0f, "%.2f");
    ImGui::EndTable();
  }
  ImGui::Checkbox("SSAO", &Context::Instance()->g_enable_ssao);
  ImGui::Checkbox("Bloom", &Context::Instance()->g_enable_bloom);
  ImGui::End();
  ImGui::Render();
}
