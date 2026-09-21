#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace glasshelix::artifacts {
enum class nf1_run_status : std::uint8_t { pending, failed, succeeded_empty, succeeded_output };
struct nf1_replay_record {
    static constexpr std::uint32_t schema_version = 1;
    std::uint32_t schema = schema_version;
    nf1_run_status status = nf1_run_status::pending;
    std::vector<float> logical_inputs{};
    std::vector<float> logical_output{};
    std::vector<std::string> registered_blocks{}; // stable-definition@revision
    std::string numerical_policy{};
    std::string glasshelix_source{};
    std::string cellerator_source{};
    std::string failure{};
};
struct nf1_runtime_identity {
    const char* glasshelix_source;
    const char* cellerator_source;
};
// This is deliberately a narrow NF1 record: it rejects control characters and
// separators instead of becoming a generic object serializer.  It never stores
// a device pointer, stream, generation, or ephemeral native handle.
bool validate(const nf1_replay_record&, std::string* error = nullptr) noexcept;
bool write_nf1_replay(const std::string& path, const nf1_replay_record&, std::string* error = nullptr) noexcept;
bool read_nf1_replay(const std::string& path, nf1_replay_record*, std::string* error = nullptr) noexcept;
// A portable logical replay probe for examples and validation. Pending/failed
// records refuse; succeeded_empty intentionally returns an empty result.
bool replay_defined_operation(const nf1_replay_record&, std::vector<float>* result, std::string* error = nullptr) noexcept;
nf1_runtime_identity runtime_identity() noexcept;
bool replay_matches_runtime(const nf1_replay_record&, std::string* error = nullptr) noexcept;
const char* status_name(nf1_run_status) noexcept;
}
