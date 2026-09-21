#include <GlassHelix/artifacts/nf1_replay.hh>
#include <cassert>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
using namespace glasshelix::artifacts;
namespace {
bool process_ok(const std::string& command) { return std::system(command.c_str()) == 0; }
bool process_rejects(const std::string& command) { return !process_ok(command); }
void write_text(const std::string& path, const std::string& text) { std::ofstream out(path, std::ios::trunc); out << text << '\n'; }
}
int main(int argc, char** argv) {
  assert(argc == 3);
  const std::string path = argv[1], cli = argv[2];
  const auto identity = runtime_identity();
  const std::string gh = identity.glasshelix_source, ce = identity.cellerator_source;
  nf1_replay_record r{};
  r.status = nf1_run_status::succeeded_output;
  r.logical_inputs = {.25f, .5f, .1f, .2f}; r.logical_output = {.3f};
  r.registered_blocks = {"local-add#10@1"}; r.numerical_policy = "f32-rne-propagate";
  r.glasshelix_source = gh; r.cellerator_source = ce;
  std::string error;
  assert(write_nf1_replay(path, r, &error));
  const std::string invoke = cli + " " + path;
  assert(process_ok(cli + " --identity"));
  assert(process_ok(invoke));
  auto wrong_source = r; wrong_source.glasshelix_source[0] = gh[0] == '0' ? '1' : '0';
  assert(write_nf1_replay(path, wrong_source, &error) && process_rejects(invoke));
  assert(write_nf1_replay(path, r, &error));
  nf1_replay_record loaded{}; assert(read_nf1_replay(path, &loaded, &error));
  std::vector<float> result; assert(replay_defined_operation(loaded, &result, &error) && result[0] == .3f);
  loaded.logical_inputs[2] = .4f; loaded.logical_output[0] = .6f;
  assert(replay_defined_operation(loaded, &result, &error) && result[0] == .6f);
  loaded.status = nf1_run_status::pending; assert(!replay_defined_operation(loaded, &result, &error));
  loaded.status = nf1_run_status::failed; loaded.failure = "native-failure"; assert(!replay_defined_operation(loaded, &result, &error));
  auto non_ascii = r; non_ascii.status = nf1_run_status::failed; non_ascii.logical_output.clear();
  non_ascii.failure = std::string("bad") + static_cast<char>(0x80); assert(!validate(non_ascii, &error));
  nf1_replay_record roundtrip = r;
  roundtrip.logical_inputs = {std::nextafter(1.f, 2.f), std::numeric_limits<float>::max(), .1f, .2f};
  assert(write_nf1_replay(path, roundtrip, &error));
  nf1_replay_record roundtrip_loaded{}; assert(read_nf1_replay(path, &roundtrip_loaded, &error));
  assert(std::bit_cast<std::uint32_t>(roundtrip.logical_inputs[0]) ==
         std::bit_cast<std::uint32_t>(roundtrip_loaded.logical_inputs[0]));
  assert(std::bit_cast<std::uint32_t>(roundtrip.logical_inputs[1]) ==
         std::bit_cast<std::uint32_t>(roundtrip_loaded.logical_inputs[1]));
  const std::string line = "GH_NF1_REPLAY|1|succeeded_output|f32-rne-propagate|" + gh + "|" + ce + "||local-add#10@1|0.25,0.5,0.1,0.2|0.3";
  write_text(path, "broken"); assert(!read_nf1_replay(path, &loaded, &error) && process_rejects(invoke));
  write_text(path, line + ","); assert(!read_nf1_replay(path, &loaded, &error) && process_rejects(invoke));
  write_text(path, "GH_NF1_REPLAY|99|pending|f32-rne-propagate|" + gh + "|" + ce + "||||"); assert(!read_nf1_replay(path, &loaded, &error) && process_rejects(invoke));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_output|f16|" + gh + "|" + ce + "||local-add#10@1|0.25,0.5,0.1,0.2|0.3"); assert(!read_nf1_replay(path, &loaded, &error) && process_rejects(invoke));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_output|f32-rne-propagate|" + gh + "|" + ce + "||unknown#1@1|0.25,0.5,0.1,0.2|0.3"); assert(read_nf1_replay(path, &loaded, &error) && process_rejects(invoke)); assert(!replay_defined_operation(loaded, &result, &error));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_empty|f32-rne-propagate|" + gh + "|" + ce + "||local-add#10@1||"); assert(read_nf1_replay(path, &loaded, &error) && process_rejects(invoke)); assert(!replay_defined_operation(loaded, &result, &error));
  write_text(path, "GH_NF1_REPLAY|1|succeeded_output|f32-rne-propagate|" + gh + "|" + ce + "||local-add#10@1|0.25,0.5,0.1,0.2|0.3\x01"); assert(!read_nf1_replay(path, &loaded, &error) && process_rejects(invoke));
  std::remove(path.c_str()); assert(process_rejects(invoke));
  std::cout << "artifact tests passed\n";
}
