#include "main.h"

void Init() {}
void Cleanup() { Context::Cleanup(); }

int main(int argc, char** argv) {
  std::cout << "start" << std::endl;
  Init();
  ExitGuard exit_guard(Cleanup);
  try {
    Application app;
    app.Run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "stop" << std::endl;
  return EXIT_SUCCESS;
}
