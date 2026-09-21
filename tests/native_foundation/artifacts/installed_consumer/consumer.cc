#include <GlassHelix/artifacts/nf1_replay.hh>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace glasshelix::artifacts;
  if (argc != 3) return 2;
  const auto identity = runtime_identity();
  nf1_replay_record record{};
  record.status = nf1_run_status::succeeded_output;
  record.logical_inputs = {.25f, .5f, .1f, .2f};
  record.logical_output = {.3f};
  record.registered_blocks = {"local-add#10@1"};
  record.numerical_policy = "f32-rne-propagate";
  record.glasshelix_source = identity.glasshelix_source;
  record.cellerator_source = identity.cellerator_source;
  std::string error;
  if (!write_nf1_replay(argv[1], record, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  return std::system((std::string(argv[2]) + " " + argv[1]).c_str()) == 0 ? 0 : 1;
}
