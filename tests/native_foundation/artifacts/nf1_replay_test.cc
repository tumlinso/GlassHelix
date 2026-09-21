#include <GlassHelix/artifacts/nf1_replay.hh>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
using namespace glasshelix::artifacts;
namespace {
bool process_ok(const std::string& command) { return std::system(command.c_str()) == 0; }
void write_text(const std::string& path, const std::string& text) { std::ofstream out(path, std::ios::trunc); out << text << '\n'; }
}
int main(int argc, char** argv) {
  assert(argc == 5);
  const std::string path = argv[1], cli = argv[2], gh = argv[3], ce = argv[4];
  nf1_replay_record r{};
  r.status = nf1_run_status::succeeded_output;
  r.logical_inputs = {.25f, .5f, .1f, .2f}; r.logical_output = {.3f};
  r.registered_blocks = {"local-add#10@1"}; r.numerical_policy = "f32-rne-propagate";
  r.glasshelix_source = gh; r.cellerator_source = ce;
  std::string error;
  assert(write_nf1_replay(path, r, &error));
  assert(process_ok(cli + " " + path + " " + gh + " " + ce));
  assert(!process_ok(cli + " " + path + " " + ce + " " + gh));
  nf1_replay_record loaded{}; assert(read_nf1_replay(path, &loaded, &error));
  std::vector<float> result; assert(replay_defined_operation(loaded, &result, &error) && result[0] == .3f);
  loaded.logical_inputs[2] = .4f; loaded.logical_output[0] = .6f;
  assert(replay_defined_operation(loaded, &result, &error) && result[0] == .6f);
  loaded.status = nf1_run_status::pending; assert(!replay_defined_operation(loaded, &result, &error));
  loaded.status = nf1_run_status::failed; loaded.failure = "native-failure"; assert(!replay_defined_operation(loaded, &result, &error));
  const std::string line = "GH_NF1_REPLAY|1|succeeded_output|f32-rne-propagate|" + gh + "|" + ce + "||local-add#10@1|0.25,0.5,0.1,0.2|0.3";
  write_text(path, "broken"); assert(!read_nf1_replay(path, &loaded, &error));
  write_text(path, line + ","); assert(!read_nf1_replay(path, &loaded, &error));
  write_text(path, "GH_NF1_REPLAY|99|pending|f32-rne-propagate|" + gh + "|" + ce + "||||"); assert(!read_nf1_replay(path, &loaded, &error));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_output|f16|" + gh + "|" + ce + "||local-add#10@1|0.25,0.5,0.1,0.2|0.3"); assert(!read_nf1_replay(path, &loaded, &error));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_output|f32-rne-propagate|" + gh + "|" + ce + "||unknown#1@1|0.25,0.5,0.1,0.2|0.3"); assert(read_nf1_replay(path, &loaded, &error)); assert(!replay_defined_operation(loaded, &result, &error));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_empty|f32-rne-propagate|" + gh + "|" + ce + "||local-add#10@1||"); assert(read_nf1_replay(path, &loaded, &error)); assert(!replay_defined_operation(loaded, &result, &error));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_output|f32-rne-propagate|" + gh + "|" + ce + "||local-add#10@1|0.25,0.5,0.1,0.2|0.3\x01"); assert(!read_nf1_replay(path, &loaded, &error));
  std::cout << "artifact tests passed\n";
}
