#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "fmt.hpp"
#include "prof.hpp"
#include "vml.hpp"

#define NUM 100000

void render(const std::string &file_path, float t, float dt) {
  PROF_FUNC_ARGS("render", file_path, t, dt);
  vml::vec3 camera_pos(-6.0 * std::cos(t), 0.0, -6.0 * std::cos(t));
  vml::mat4 view =
      inverse(look_at(camera_pos, vml::vec3(0.0), vml::vec3(0.0, 1.0, 0.0)));
}

int main(int argc, char *argv[]) { return 0; }
