#pragma once

#include <functional>
#include <iostream>
#include <vector>

#ifndef SHADER_FILE_PATH
#define SHADER_FILE_PATH ""
#endif
#ifndef DATA_FILE_PATH
#define DATA_FILE_PATH ""
#endif

#ifndef NDEBUG
#define LOG(...)                                       \
  {                                                    \
    std::cout << __FILE__ << ":" << __LINE__ << " : "; \
    std::vector<std::string> v{__VA_ARGS__};           \
    for (const std::string& msg : v) {                 \
      std::cout << msg;                                \
    }                                                  \
    std::cout << std::endl;                            \
  }
#else
#define LOG(...) ;
#endif

std::vector<char> ReadFile(const char* file_path);

class ExitGuard {
 public:
  ExitGuard(std::function<void()> func);
  ~ExitGuard();

 private:
  std::function<void()> func;
};
