#include "utils.h"

#include <fstream>

std::vector<char> ReadFile(const char* file_path) {
  std::ifstream file(file_path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }
  std::vector<char> buffer(file.tellg());
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  file.close();
  return buffer;
}

ExitGuard::ExitGuard(std::function<void()> func) : func(func) {}

ExitGuard::~ExitGuard() { func(); }
