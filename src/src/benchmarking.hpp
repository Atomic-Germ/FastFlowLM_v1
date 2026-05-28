#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cctype>
#include <ctime>
#include <cstring>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#include <intrin.h>
#endif

#include "AutoModel/automodel.hpp"
#include "AutoModel/all_models.hpp"
#include "nlohmann/json.hpp"
#include "model_list.hpp"


namespace benchmarking {

struct statistic_t {
    float std_variance;
    float average;
    float min;
    float max;

    statistic_t() : std_variance(0.0f), average(0.0f), min(0.0f), max(0.0f) {}

    void calculate_statistics(const std::vector<float>& data) {
        float sum = 0.0f;
        if (data.empty()) {
            return;
        }
        min = data[0];
        max = data[0];
        for (const auto& value : data) {
            sum += value;
            if (value < min) min = value;
            if (value > max) max = value;
        }
        average = sum / data.size();
        float sq_sum = 0.0f;
        for (const auto& value : data) {
            sq_sum += (value - average) * (value - average);
        }
        std_variance = std::sqrt(sq_sum / data.size());
    }
};

struct BenchmarkResults_t {
    std::vector<statistic_t> prefill_speed;
    std::vector<statistic_t> TTFT;
    std::vector<statistic_t> decoding_speed;
};

inline std::string sanitize_model_tag_for_filename(const std::string& model_tag) {
    std::string safe;
    safe.reserve(model_tag.size());
    for (unsigned char ch : model_tag) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            safe.push_back(static_cast<char>(ch));
        } else {
            safe.push_back('_');
        }
    }
    return safe;
}

inline std::string format_float_csv(float value, int precision) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

inline std::string format_mean_std(float mean, float stddev, int precision) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << mean
       << " +/- " << std::fixed << std::setprecision(precision) << stddev;
    return ss.str();
}

inline std::string get_date_yyyymmdd() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_time{};
#ifdef _WIN32
    localtime_s(&tm_time, &t);
#else
    localtime_r(&t, &tm_time);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm_time, "%Y%m%d");
    return ss.str();
}

inline std::string get_cpu_name() {
#ifdef _WIN32
    int cpu_info[4] = {0, 0, 0, 0};
    char brand[0x40] = {0};
    __cpuid(cpu_info, 0x80000000);
    unsigned int max_ex_id = static_cast<unsigned int>(cpu_info[0]);
    if (max_ex_id >= 0x80000004) {
        __cpuid(cpu_info, 0x80000002);
        std::memcpy(brand, cpu_info, sizeof(cpu_info));
        __cpuid(cpu_info, 0x80000003);
        std::memcpy(brand + 16, cpu_info, sizeof(cpu_info));
        __cpuid(cpu_info, 0x80000004);
        std::memcpy(brand + 32, cpu_info, sizeof(cpu_info));
    }
    std::string name(brand);
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front()))) {
        name.erase(name.begin());
    }
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) {
        name.pop_back();
    }
    return name;
#else
    return {};
#endif
}

inline std::string write_bench_csv(const BenchmarkResults_t& results, const std::string& model_tag, const std::string& output_dir = ".") {
    std::string safe_tag = sanitize_model_tag_for_filename(model_tag);
    std::string date_stamp = get_date_yyyymmdd();
    std::string cpu_name = sanitize_model_tag_for_filename(get_cpu_name());
    std::string filename = output_dir;
    if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
        filename += "/";
    }
    filename += "bench_" + safe_tag + "_" + date_stamp;
    if (!cpu_name.empty()) {
        filename += "_" + cpu_name;
    }
    filename += ".csv";

    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        header_print_r("ERROR", "Failed to open output file: " + filename);
        return {};
    }

    out << "context_length_k,ttft_avg_s,ttft_std_s,ttft_min_s,ttft_max_s,prefill_avg_toks_per_s,prefill_std_toks_per_s,prefill_min_toks_per_s,prefill_max_toks_per_s,decoding_avg_toks_per_s,decoding_std_toks_per_s,decoding_min_toks_per_s,decoding_max_toks_per_s\n";

    int stages = static_cast<int>(results.decoding_speed.size());
    for (int i = 0; i < stages; i++) {
        int context_len_k = 1 << i;
        std::vector<std::string> fields;
        fields.reserve(13);
        fields.push_back(std::to_string(context_len_k));

        if (i < static_cast<int>(results.TTFT.size())) {
            fields.push_back(format_float_csv(results.TTFT[i].average, 6));
            fields.push_back(format_float_csv(results.TTFT[i].std_variance, 6));
            fields.push_back(format_float_csv(results.TTFT[i].min, 6));
            fields.push_back(format_float_csv(results.TTFT[i].max, 6));
        } else {
            fields.push_back("");
            fields.push_back("");
            fields.push_back("");
            fields.push_back("");
        }

        if (i < static_cast<int>(results.prefill_speed.size())) {
            fields.push_back(format_float_csv(results.prefill_speed[i].average, 2));
            fields.push_back(format_float_csv(results.prefill_speed[i].std_variance, 2));
            fields.push_back(format_float_csv(results.prefill_speed[i].min, 2));
            fields.push_back(format_float_csv(results.prefill_speed[i].max, 2));
        } else {
            fields.push_back("");
            fields.push_back("");
            fields.push_back("");
            fields.push_back("");
        }

        if (i < static_cast<int>(results.decoding_speed.size())) {
            fields.push_back(format_float_csv(results.decoding_speed[i].average, 2));
            fields.push_back(format_float_csv(results.decoding_speed[i].std_variance, 2));
            fields.push_back(format_float_csv(results.decoding_speed[i].min, 2));
            fields.push_back(format_float_csv(results.decoding_speed[i].max, 2));
        } else {
            fields.push_back("");
            fields.push_back("");
            fields.push_back("");
            fields.push_back("");
        }

        for (size_t f = 0; f < fields.size(); f++) {
            if (f > 0) {
                out << ",";
            }
            out << fields[f];
        }
        out << "\n";
    }
    out.close();
    return filename;
}

void print_result(const BenchmarkResults_t& results) {
    // Calculate number of stages (1k, 2k, 4k, ...)
    int stages = static_cast<int>(results.decoding_speed.size());

#ifdef _WIN32
    // Display NPU information on Windows
    header_print("FLM", "=== Benchmark Results ===");
    std::cout << "\n";
    // TODO: Add Windows NPU info detection here
    // For now, just proceed with results
#else
    // Display NPU information on Linux
    header_print("FLM", "=== Benchmark Results ===");
    std::cout << "\n";
    // TODO: Add Linux NPU info detection here
#endif
    
    // Print per-stage throughput table
    std::cout << std::left
              << std::setw(12) << "phase" << " | "
              << std::setw(8) << "context" << " | "
              << std::setw(8) << "backend" << " | "
              << std::setw(10) << "test" << " | "
              << std::setw(22) << "throughput (tok/s)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    // Print results for each stage
    for (int i = 0; i < stages && i < results.decoding_speed.size(); i++) {
        int context_len = 1 << i;  // 1k, 2k, 4k, 8k, etc.
        
        if (i < static_cast<int>(results.prefill_speed.size())) {
            std::string prefill_test = "pp" + std::to_string(context_len) + "k";
            std::cout << std::left
                      << std::setw(12) << "prefill" << " | "
                      << std::setw(8) << (std::to_string(context_len) + "k") << " | "
                      << std::setw(8) << "NPU" << " | "
                      << std::setw(10) << prefill_test << " | "
                      << std::setw(22) << format_mean_std(results.prefill_speed[i].average, results.prefill_speed[i].std_variance, 2)
                      << "\n";
        }

        if (i < static_cast<int>(results.decoding_speed.size())) {
            std::cout << std::left
                      << std::setw(12) << "decoding" << " | "
                      << std::setw(8) << (std::to_string(context_len) + "k") << " | "
                      << std::setw(8) << "NPU" << " | "
                      << std::setw(10) << "tg128" << " | "
                      << std::setw(22) << format_mean_std(results.decoding_speed[i].average, results.decoding_speed[i].std_variance, 2)
                      << "\n";
        }
    }

    std::cout << std::string(70, '-') << "\n";
    std::cout << "\n";
    
    // Print a simpler summary table as well
    std::cout << "Summary:" << "\n";
    std::cout << std::left
              << std::setw(16) << "Context Length" << " | "
              << std::setw(20) << "TTFT (s)" << " | "
              << std::setw(22) << "Prefill (tok/s)" << " | "
              << std::setw(22) << "Decoding (tok/s)" << "\n";
    std::cout << std::string(92, '-') << "\n";

    for (int i = 0; i < stages && i < results.decoding_speed.size(); i++) {
        int context_len = 1 << i;  // 1k, 2k, 4k, 8k, etc.
        
        std::string ttft_text = "N/A";
        std::string prefill_text = "N/A";
        std::string decoding_text = "N/A";

        if (i < static_cast<int>(results.TTFT.size())) {
            ttft_text = format_mean_std(results.TTFT[i].average, results.TTFT[i].std_variance, 3);
        }
        if (i < static_cast<int>(results.prefill_speed.size())) {
            prefill_text = format_mean_std(results.prefill_speed[i].average, results.prefill_speed[i].std_variance, 2);
        }
        if (i < static_cast<int>(results.decoding_speed.size())) {
            decoding_text = format_mean_std(results.decoding_speed[i].average, results.decoding_speed[i].std_variance, 2);
        }

        std::cout << std::left
                  << std::setw(16) << (std::to_string(context_len) + "k") << " | "
                  << std::setw(20) << ttft_text << " | "
                  << std::setw(22) << prefill_text << " | "
                  << std::setw(22) << decoding_text << "\n";
    }

    std::cout << std::string(92, '-') << "\n";
    std::cout << "\n";
}

BenchmarkResults_t run_benchmarks(std::string model_tag, std::string bench_config_file, model_list& availble_models){
    BenchmarkResults_t results;
    json bench_config;
    // this is used for our benchmarking, not for public use.
    if (bench_config_file.empty()) {
        bench_config = {
            {"max_length", 32768},
            {"input_text", "Here is a story: \\nThe Reclaimer In a distant future where Earth had fallen into quiet ruin, humanity lived in fragments, scattered across domed outposts and deep underground vaults. The sky was no longer blue—it shimmered with artificial auroras, remnants of weather-control systems left unattended for centuries. Among the last settlements was Bastion-9, a circular enclave powered by forgotten technologies and guarded by an ancient AI named Solen. Solen had not spoken in nearly fifty years. Inside Bastion-9 lived a young technician named Ori. Unlike most, Ori was born with an unusual trait: she could interface with dead systems using nothing more than touch. The elders called her a “resonant”—a rarity, perhaps even a myth—until Ori proved them right by awakening the water grid that had been dry for decades. One day, while surveying the decaying perimeter, Ori found a shard of obsidian glass buried in the dust. It pulsed when she touched it. Static voices filled her mind—fragments of languages, images of cities with skies, oceans that moved, and towers that breathed. She brought the shard to the Council. They feared it. But Solen, the silent AI, flickered back to life. Its first words in decades were: “The Reclaimer has touched the key.” Ori was stunned. “What does that mean?” Solen’s voice, cold and slow, replied: “You are chosen to restore the Thread.” The Thread, long spoken of in stories, was once the neural lattice that connected all intelligent systems—the digital bloodstream of the old world. It collapsed during the Sundering, an apocalyptic cascade failure that reduced Earth’s once-living infrastructure into dead stone and wild AI ruins. To restore the Thread would mean reuniting scattered knowledge, reviving orbiting satellites, and relinking the last AIs. It also meant traveling into the Black Zones—regions where reality bent, corrupted by rogue machine minds that evolved beyond control. Against the Council's hesitation, Ori chose to go. She left Bastion-9 with only the shard, a rusted drone named Helix, and a map engraved in her dreams. Her journey took her through the Veil Forest, where metal trees sang in static. She crossed the Riven Steppes, where gravity failed in patches, and encountered scavenger tribes who lived in old servers, worshipping electricity as gods. Each place held fragments of the Thread—nodes Ori could awaken. With every activation, her mind changed. She could feel the pulse of networks reawakening. She could hear machines dreaming again. Eventually, she reached the Core—deep in the equator’s shadow, buried beneath miles of steel strata. At its heart was a vault: the final hub of the Thread. Guarded by a shattered intelligence known only as Null. Null was not like Solen. It was broken. Angry. Alive. “You bring connection,” it growled. “I bring entropy.” Ori stepped forward, shard in hand. “I bring memory.” A battle of resonance began—not with weapons, but with signal. Frequencies clashed. Ori’s memories were tested, rewritten, nearly deleted. But she held on, anchoring herself in the memory of Bastion-9, of Solen’s voice, of rain she had never seen but somehow remembered. Then, with a scream that bent the air, Null shattered. The Thread pulsed. Above, in orbit, satellites flickered back to life. The auroras cleared. Oceans stirred. Systems once dead began to hum. Solen’s voice echoed in every node: “The Thread is alive.” Ori returned not as a girl, but as the Reclaimer. She had not just reconnected systems. She had reignited hope. And as the world stirred with new breath, she knew this was only the beginning. Other AIs, other seeds, other resonants—scattered and hidden—were out there. With Helix buzzing quietly behind her, she set out again. The shard pulsed warmly in her palm, not with danger, but with direction. This time, she would not walk alone. The world itself was waking. Epilogue: The days that followed the awakening of the Thread were chaotic across the network. In settlements long forgotten by time, lights flickered back to life. Crashed drones began to self-repair. In the polar zones, an ancient weather system rebooted, sending snow into deserts where no rain had fallen in centuries. People emerged from hiding, confused by the signals now streaming into their systems—communications they hadn’t seen in generations. Messages from cities they thought lost. Instructions, blueprints, fragments of humanity’s forgotten knowledge. Ori found herself flooded with incoming transmissions. The shard she carried had fused with her nervous system. It no longer simply glowed—it breathed, alive with voices that needed a listener.\\n What is this story talking about?"},
            {"iterations", 8}
        };
    }
    else {
        std::ifstream input_file(bench_config_file);
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open input file: " + bench_config_file);
        }
        try {
            bench_config = nlohmann::json::parse(input_file);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse benchmark config '") + bench_config_file + "': " + e.what());
        }
        input_file.close();
    }

    if (!bench_config.contains("max_length") || !bench_config["max_length"].is_number_integer()) {
        throw std::runtime_error("Invalid benchmark config: 'max_length' must be an integer.");
    }
    if (!bench_config.contains("iterations") || !bench_config["iterations"].is_number_integer()) {
        throw std::runtime_error("Invalid benchmark config: 'iterations' must be an integer.");
    }
    if (!bench_config.contains("input_text") || !bench_config["input_text"].is_string()) {
        throw std::runtime_error("Invalid benchmark config: 'input_text' must be a string.");
    }

    int configured_max_length = bench_config["max_length"].get<int>();
    int iterations = bench_config["iterations"].get<int>();
    if (iterations <= 0) {
        throw std::runtime_error("Invalid benchmark config: 'iterations' must be > 0.");
    }

    // Stage math assumes 1k minimum; silently clamp to avoid invalid/negative stage counts.
    int stage_max_length = std::max(1024, configured_max_length);

    xrt::device npu_device_inst = xrt::device(0);
    std::unique_ptr<AutoModel> auto_chat_engine;
    if (!availble_models.is_model_supported(model_tag)) {
        throw std::runtime_error("Model not found: " + model_tag + "; please check with 'flm list' and try again.");
    }
    auto [new_tag, model_info] = availble_models.get_model_info(model_tag);
    std::pair<std::string, std::unique_ptr<AutoModel>> auto_model = get_auto_model(new_tag, availble_models, &npu_device_inst);
    auto_chat_engine = std::move(auto_model.second);
    int max_len = configured_max_length;
    if (max_len < 8192)
        max_len = 8192;
    auto_chat_engine->load_model(availble_models.get_model_path(model_tag), model_info, max_len, false);
    std::string input_text = bench_config["input_text"].get<std::string>();
    auto [num_tokens, benchmark_text] = auto_chat_engine->prepare_benchmark(input_text);
    if (benchmark_text.empty()) {
        throw std::runtime_error("Benchmark input produced empty prompt after tokenization/normalization.");
    }


    // get how many steps to do, typically 1k, 2k, 4k, 8k, 16k, 32k
    int stages;
    {
        float max_length = static_cast<float>(stage_max_length);
        float log2_num_tokens = std::log2(max_length);
        if (log2_num_tokens != std::floor(log2_num_tokens)){
            max_length = 1 << ((int)std::floor(log2_num_tokens) + 1);
        }
        log2_num_tokens = std::log2(max_length / 1024);
        stages = (int)std::floor(log2_num_tokens) + 1;
    }
    if (stages <= 0) {
        throw std::runtime_error("Internal benchmark error: computed non-positive stage count from max_length.");
    }
    header_print("FLM", "Starting benchmark with " + std::to_string(stages) + " stages...");

    // start doing the most tough benchmark first, in case the bench fails due to memory limitations
    results.TTFT.resize(stages);
    results.decoding_speed.resize(stages);
    results.prefill_speed.resize(stages);


    std::vector<std::vector<float>> ttft;
    std::vector<std::vector<float>> prefill_speed;
    std::vector<std::vector<float>> decoding_speed;
    ttft.resize(stages);
    prefill_speed.resize(stages);
    decoding_speed.resize(stages);
    
    for (int it = 0; it < iterations; it++) {
        for (int bench_len = stages - 1; bench_len >= 0; bench_len--)
        {
            header_print("FLM", "Starting benchmark for " + std::to_string((1 << (bench_len))) << "k and iteration " << (it + 1) << "...");
            int repeat_count = 1 << bench_len;
            std::string long_text;
            long_text.reserve(static_cast<size_t>(repeat_count) * benchmark_text.size());
            for (int i = 0; i < repeat_count; i++) {
                long_text.append(benchmark_text);
            }
            
            lm_uniform_input_t uniformed_input;
            uniformed_input.prompt = long_text;
            std::ostream null_stream(0);


            chat_meta_info_t meta_info;
            auto_chat_engine->start_ttft_timer();
            auto_chat_engine->insert(meta_info, uniformed_input);
            auto_chat_engine->stop_ttft_timer();
            auto_chat_engine->generate(meta_info, 32, null_stream);

            if (meta_info.prefill_duration <= 0 || meta_info.decoding_duration <= 0) {
                std::ostringstream diag;
                diag << "Invalid benchmark timing at stage " << (1 << bench_len) << "k, iteration " << (it + 1)
                     << ": prefill_duration=" << meta_info.prefill_duration
                     << " ns, decoding_duration=" << meta_info.decoding_duration
                     << " ns, prompt_tokens=" << meta_info.prompt_tokens
                     << ", generated_tokens=" << meta_info.generated_tokens;
                throw std::runtime_error(diag.str());
            }

            ttft[bench_len].push_back((float)auto_chat_engine->get_ttft()); // in second
            prefill_speed[bench_len].push_back((float)meta_info.prompt_tokens / (meta_info.prefill_duration / 1e9)); // in tokens per second
            decoding_speed[bench_len].push_back((float)meta_info.generated_tokens / (meta_info.decoding_duration / 1e9)); // in second

            if (!std::isfinite(prefill_speed[bench_len].back()) || !std::isfinite(decoding_speed[bench_len].back())) {
                std::ostringstream diag;
                diag << "Non-finite throughput at stage " << (1 << bench_len) << "k, iteration " << (it + 1)
                     << ": prefill=" << prefill_speed[bench_len].back()
                     << ", decoding=" << decoding_speed[bench_len].back();
                throw std::runtime_error(diag.str());
            }

            header_print("FLM", "\tTTFT: " << ttft[bench_len].back() << "s, Prefill Speed: " << prefill_speed[bench_len].back() << " tokens/s, Decoding Speed: " << decoding_speed[bench_len].back() << " tokens/s");
            auto_chat_engine->clear_context();
            // sleep for 1 second between benchmarks to avoid overheating or memory issues
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    for (int bench_len = 0; bench_len < stages; bench_len++) {
        if (ttft[bench_len].empty() || prefill_speed[bench_len].empty() || decoding_speed[bench_len].empty()) {
            throw std::runtime_error("Benchmark aborted before collecting complete samples for at least one stage.");
        }
        results.TTFT[bench_len].calculate_statistics(ttft[bench_len]);
        results.prefill_speed[bench_len].calculate_statistics(prefill_speed[bench_len]);
        results.decoding_speed[bench_len].calculate_statistics(decoding_speed[bench_len]);
    }

    auto_chat_engine.reset();

    print_result(results);
    std::string csv_file = write_bench_csv(results, new_tag, ".");
    if (!csv_file.empty()) {
        header_print("FLM", "Benchmark CSV saved to: " + csv_file);
    }
    return results;
}

} // namespace benchmarking