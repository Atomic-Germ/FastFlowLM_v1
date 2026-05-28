/// \file modeling_gpt_oss.cpp
/// \brief modeling_gpt_oss class
/// \author FastFlowLM Team
/// \date 2025-10-01
/// \version 0.9.24
/// \note This is a source file for the gpt-oss class
#include "AutoModel/modeling_gpt_oss.hpp"   

static std::string normalize_tool_args(const std::string& tool_name, const std::string& raw_args) {
    if (raw_args.empty()) {
        return raw_args;
    }

    try {
        auto j = nlohmann::json::parse(raw_args);
        if (!j.is_object()) {
            return raw_args;
        }

        if (tool_name == "bash") {
            if (!j.contains("command")) {
                if (j.contains("arg") && j["arg"].is_string()) {
                    std::string cmd = j["arg"].get<std::string>();
                    // Avoid converting clearly non-command multiline content.
                    if (!cmd.empty() && cmd.find('\n') == std::string::npos && cmd.find('\r') == std::string::npos) {
                        j["command"] = cmd;
                        j.erase("arg");
                    }
                }
                else if (j.contains("arg") && j["arg"].is_object()) {
                    auto arg_obj = j["arg"];
                    if (arg_obj.contains("command") && arg_obj["command"].is_string()) {
                        j["command"] = arg_obj["command"].get<std::string>();
                        j.erase("arg");
                    }
                }
                else if (j.contains("result") && j["result"].is_string()) {
                    std::string cmd = j["result"].get<std::string>();
                    if (!cmd.empty() && cmd.find('\n') == std::string::npos && cmd.find('\r') == std::string::npos) {
                        j["command"] = cmd;
                    }
                }
            }
        }

        return j.dump();
    } catch (...) {
        return raw_args;
    }
}


GPT_OSS::GPT_OSS(xrt::device* npu_device_inst) : AutoModel(npu_device_inst, "gpt-oss") {}

void GPT_OSS::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->model_path = model_path;
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    this->lm_engine = std::make_unique<gpt_oss_npu>(*this->lm_config, this->npu.get(), this->MAX_L);
    this->lm_engine->load_weights(*this->q4nx);
    this->q4nx.reset();
    this->tokenizer = std::make_unique<Tokenizer>(model_path);

    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 10;
    config.top_p = 0.95;
    config.min_p = 0.1;
    config.temperature = 0.6;
    config.rep_penalty = 1.05;
    config.freq_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void GPT_OSS::setup_tokenizer(std::string model_path){
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string GPT_OSS::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    nlohmann::ordered_json templated_messages = messages;

    if (!tools.empty()) {
        // Some GPT-OSS templates do not reliably render tool definitions.
        // Inject an explicit system instruction with the exact call format
        // expected by our parser/runtime to prevent fabricated answers.
        bool has_tool_result = false;
        for (auto& msg : messages) {
            if (msg.contains("role") && msg["role"].is_string() && msg["role"].get<std::string>() == "tool") {
                has_tool_result = true;
                break;
            }
        }

        std::string tool_system =
            "# Tools\n\n"
            "You MUST use tools when the user request requires external data, files, or command execution.\n"
            "Never fabricate command outputs or file listings.\n\n"
            "Available functions:\n";
        for (auto& tool : tools) {
            tool_system += "- " + tool.dump() + "\n";
        }
        tool_system +=
            "\nWhen calling a function, output ONLY this format:\n"
            "<|channel|>final <|constrain|>functions.<function_name><|message|>{\"arg\":\"value\"}<|call|>\n"
            "Do not output normal assistant content before or instead of a function call when a tool is needed.\n"
            "For bash calls, arguments MUST include key \"command\" as a string.\n"
            "Do NOT use \"arg\" as the top-level key for bash.";

        if (has_tool_result) {
            tool_system +=
                "\n\n# Tool result grounding\n"
                "A tool result is present in the conversation.\n"
                "Your final answer MUST be grounded only in that tool output.\n"
                "Do not add files, paths, or values that are not explicitly in the tool output.\n"
                "If details are missing, say they are not shown instead of guessing.";
        }

        nlohmann::ordered_json modified = nlohmann::ordered_json::array();
        if (!messages.empty() && messages[0].contains("role") && messages[0]["role"] == "system") {
            nlohmann::ordered_json sys = messages[0];
            std::string existing = (messages[0].contains("content") && messages[0]["content"].is_string())
                ? messages[0]["content"].get<std::string>() : "";
            sys["content"] = tool_system + (existing.empty() ? "" : "\n\n" + existing);
            modified.push_back(sys);
            for (size_t i = 1; i < messages.size(); ++i) {
                modified.push_back(messages[i]);
            }
        }
        else {
            modified.push_back({{"role", "system"}, {"content", tool_system}});
            for (auto& msg : messages) {
                modified.push_back(msg);
            }
        }
        templated_messages = modified;
    }

    inputs.messages = templated_messages;
    if (!tools.empty()) {
        inputs.tools = tools;
    }
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    inputs.extra_context["reasoning_effort"] = this->reasoning_effort;
    inputs.extra_context["model_identity"] = this->model_identity;
    inputs.extra_context["role"] = this->role;

    return this->chat_tmpl->apply(inputs);
}
bool GPT_OSS::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled)
{
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        this->tools = input.tools;
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages, input.tools);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    
    // hardware
    int restore_idx = -1;
    gpt_oss_npu *gpt_oss_engine = dynamic_cast<gpt_oss_npu*>(this->lm_engine.get());
    if (meta_info.restore_allowed) {
        restore_idx = gpt_oss_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }
    bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = gpt_oss_engine->checkpoint();

    return success;
}

std::string GPT_OSS::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    os << "<|start|>" << std::flush;
    os << "assistant" << std::flush;
    std::vector<int> sampled_tokens;
    std::string result;
    if (length_limit > 0){
        sampled_tokens.reserve(length_limit);
    }
    else{
        sampled_tokens.reserve(4096);
    }
    assert(this->last_token != -1);

    stop_reason_t reason = EOT_DETECTED;
    int last_sampled_token = this->last_token;

 
    token_history.push_back(last_token);
    if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1){
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;

    }
    if (this->is_eos(last_sampled_token)){
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
        return result;
    }
    while (this->total_tokens < this->MAX_L){
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            // reset stream content 
            buffer_.clear();
            current_mode_ = StreamEventType::CONTENT;
            tool_name_.clear();
            is_in_tool_block_ = false;
            break;
        }
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->sampler->sample(y);
        this->profiler_list[SAMPLING_TIME].stop(1);
        this->total_tokens++;
        last_sampled_token = sampled_token;

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)){ // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        token_history.push_back(sampled_token);
        if (this->is_eos(sampled_token)){
            meta_info.generated_tokens++;
            this->lm_engine->forward(last_sampled_token);
            break;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
            reason = MAX_LENGTH_REACHED;
            break;
        }
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);

    os << "<|end|>" << std::flush;
    return "<|start|>assistant" + result + "<|end|>";
}

/*
// std::string GPT_OSS::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
//     const std::string MARKER_TOOL_START = "<|start|>assistant<|channel|>commentary";
//     bool is_constrained_mode = false;

//     os << "<|start|>" << std::flush;
//     os << "assistant" << std::flush;
//     std::string result = this->_shared_generate(meta_info, length_limit, os, is_cancelled);
//     //std::vector<int> sampled_tokens;
//     //std::string result;
//     //if (length_limit > 0) {
//     //    sampled_tokens.reserve(length_limit);
//     //}
//     //else {
//     //    sampled_tokens.reserve(4096);
//     //}
//     //assert(this->last_token != -1);

//     //stop_reason_t reason = EOT_DETECTED;
//     //int last_sampled_token = this->last_token;
//     //this->token_history.push_back(this->last_token);
//     //auto decoding_start_time = time_utils::now();
//     //if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1) {
//     //    std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
//     //    result += token_str;
//     //    os << token_str << std::flush;

//     //}
//     //if (this->is_eos(last_sampled_token)) {
//     //    return result;
//     //}
//     //this->profiler_list[TKOEN_DECODE_TIME].stop(1);
//     //if (this->total_tokens >= this->MAX_L) {
//     //    header_print("WARNING", "Max length reached, stopping generation...");
//     //    reason = MAX_LENGTH_REACHED;
//     //    return result;
//     //}
//     //while (this->total_tokens < this->MAX_L) {
//     //    if (is_cancelled()) {
//     //        reason = CANCEL_DETECTED;
//     //        // reset stream content 
//     //        buffer_.clear();
//     //        current_mode_ = StreamEventType::CONTENT;
//     //        waiting_for_header_ = true;
//     //        break;
//     //    }
//     //    this->profiler_list[DECODING_TIME].start();
//     //    buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
//     //    this->profiler_list[DECODING_TIME].stop(1);

//     //    this->profiler_list[SAMPLING_TIME].start();
//     //    // if MARKER_TOOL_START found 
//     //    
//     //    if (!is_constrained_mode) {
//     //        if (result.length() >= MARKER_TOOL_START.length() &&
//     //            result.compare(result.length() - MARKER_TOOL_START.length(), MARKER_TOOL_START.length(), MARKER_TOOL_START) == 0) {
//     //            is_constrained_mode = true;
//     //            reset_tool_grammar_state();
//     //            header_print("INFO", "<|start|>assistant<|channel|>commentary detected! Switching to Constrained Decoding.");
//     //        }
//     //    }

//     //    int sampled_token;
//     //    if (is_constrained_mode) {
//     //        sampled_token = _sample_in_tool(y);
//     //        if (current_state == STATE_COMPLETE) {
//     //            is_constrained_mode = false;
//     //            header_print("INFO", "Constrained Decoding complete, returning to normal mode");
//     //        }
//     //    }
//     //    else
//     //        sampled_token = this->sampler->sample(y);
//     //    this->profiler_list[SAMPLING_TIME].stop(1);
//     //    this->total_tokens++;
//     //    last_sampled_token = sampled_token;

//     //    this->profiler_list[TKOEN_DECODE_TIME].start();
//     //    this->profiler_list[TKOEN_DECODE_TIME].stop(1);
//     //    if (this->is_normal_token(sampled_token)) { // filter out special tokens
//     //        std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
//     //        os << token_str << std::flush;
//     //        result += token_str;
//     //    }
//     //    this->token_history.push_back(sampled_token);
//     //    if (this->is_eos(sampled_token)) {
//     //        this->lm_engine->forward(last_sampled_token);
//     //        break;
//     //    }
//     //    meta_info.generated_tokens++;
//     //    if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
//     //        reason = MAX_LENGTH_REACHED;
//     //        break;
//     //    }
//     //}

//     //auto decoding_end_time = time_utils::now();
//     //meta_info.decoding_duration = (uint64_t)time_utils::duration_ns(decoding_start_time, decoding_end_time).first;
//     //meta_info.stop_reason = reason;
//     //if (this->total_tokens >= this->MAX_L) {
//     //    header_print("WARNING", "Max length reached, stopping generation...");
//     //}

//     os << "<|end|>" << std::flush;
//     return "<|start|>assistant" + result + "<|end|>";
// }
*/

std::string GPT_OSS::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    os << "<|start|>" << std::flush;
    os << "assistant" << std::flush;
    std::string result = this->_shared_generate(meta_info, length_limit, os);
    os << "<|end|>" << std::flush;
    return "<|start|>assistant" + result + "<|end|>";
}

NonStreamResult GPT_OSS::parse_nstream_content(const std::string response_text) {    
    NonStreamResult result;

    auto parse_function_envelope = [](const std::string& text, std::string& out_name, std::string& out_args) -> bool {
        std::string s = text;
        while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r' || s.front() == '\t')) {
            s.erase(s.begin());
        }
        while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t')) {
            s.pop_back();
        }
        if (s.empty()) {
            return false;
        }

        try {
            auto j = nlohmann::json::parse(s);
            if (!j.is_object()) {
                return false;
            }

            if (j.contains("type") && j["type"].is_string() && j["type"].get<std::string>() == "function" &&
                j.contains("name") && j["name"].is_string()) {
                out_name = j["name"].get<std::string>();
                if (j.contains("parameters")) {
                    out_args = j["parameters"].is_string() ? j["parameters"].get<std::string>() : j["parameters"].dump();
                } else {
                    out_args = "{}";
                }
                out_args = normalize_tool_args(out_name, out_args);
                return true;
            }

            if (j.contains("name") && j["name"].is_string() && j.contains("arguments")) {
                out_name = j["name"].get<std::string>();
                out_args = j["arguments"].is_string() ? j["arguments"].get<std::string>() : j["arguments"].dump();
                out_args = normalize_tool_args(out_name, out_args);
                return true;
            }
        } catch (...) {}

        return false;
    };

    const std::string think_start_tag = "<|start|>assistant<|channel|>analysis<|message|>";
    const std::string think_start_tag_short = "<|channel|>analysis<|message|>";
    const std::string think_end_tag = "<|end|>";
    const std::string final_start_tag = "<|start|>assistant<|channel|>final<|message|>";
    const std::string final_start_tag_short = "<|channel|>final<|message|>";
    const std::string final_end_tag = "<|end|>";

    // --- Parse reasoning content ---
    size_t t_start = response_text.find(think_start_tag);
    size_t think_tag_len = think_start_tag.size();
    if (t_start == std::string::npos) {
        t_start = response_text.find(think_start_tag_short);
        think_tag_len = think_start_tag_short.size();
    }
    size_t t_end = response_text.find(think_end_tag, t_start == std::string::npos ? 0 : t_start + think_tag_len);

    if (t_start != std::string::npos && t_end != std::string::npos) {
        t_start += think_tag_len;
        result.reasoning_content = response_text.substr(t_start, t_end - t_start);
    }

    // --- Parse final content ---
    size_t f_start = response_text.find(final_start_tag);
    size_t final_tag_len = final_start_tag.size();
    if (f_start == std::string::npos) {
        f_start = response_text.find(final_start_tag_short);
        final_tag_len = final_start_tag_short.size();
    }
    size_t f_end = response_text.find(final_end_tag, f_start == std::string::npos ? 0 : f_start + final_tag_len);

    if (f_start != std::string::npos && f_end != std::string::npos) {
        f_start += final_tag_len;
        result.content = response_text.substr(f_start, f_end - f_start);
        if (parse_function_envelope(result.content, result.tool_name, result.tool_args)) {
            result.content.clear();
            return result;
        }
    }

    // Parse harmony-style tool calls:
    // <|start|>assistant<|channel|>commentary to=functions.<name> ... <|message|>{...}<|call|><|end|>
    const std::string tool_start_tag = "<|start|>assistant<|channel|>commentary to=functions.";
    const std::string tool_start_tag_short = "<|channel|>commentary to=functions.";
    const std::string tool_start_tag_constrain = "<|start|>assistant<|channel|>final <|constrain|>functions.";
    const std::string tool_start_tag_constrain_no_space = "<|start|>assistant<|channel|>final<|constrain|>functions.";
    const std::string tool_start_tag_constrain_short = "<|channel|>final <|constrain|>functions.";
    const std::string tool_start_tag_constrain_short_no_space = "<|channel|>final<|constrain|>functions.";
    const std::string tool_start_tag_plain = "<|start|>assistant<|channel|>final functions.";
    const std::string tool_start_tag_plain_double_space = "<|start|>assistant<|channel|>final  functions.";
    const std::string tool_start_tag_plain_short = "<|channel|>final functions.";
    const std::string tool_start_tag_plain_short_double_space = "<|channel|>final  functions.";
    const std::string tool_msg_split = "<|message|>";
    const std::string tool_call_marker = "<|call|>";
    size_t tc_start = std::string::npos;
    size_t tool_tag_len = 0;

    auto update_tool_start = [&](const std::string& marker) {
        size_t pos = response_text.find(marker);
        if (pos != std::string::npos && (tc_start == std::string::npos || pos < tc_start)) {
            tc_start = pos;
            tool_tag_len = marker.size();
        }
    };

    update_tool_start(tool_start_tag);
    update_tool_start(tool_start_tag_short);
    update_tool_start(tool_start_tag_constrain);
    update_tool_start(tool_start_tag_constrain_no_space);
    update_tool_start(tool_start_tag_constrain_short);
    update_tool_start(tool_start_tag_constrain_short_no_space);
    update_tool_start(tool_start_tag_plain);
    update_tool_start(tool_start_tag_plain_double_space);
    update_tool_start(tool_start_tag_plain_short);
    update_tool_start(tool_start_tag_plain_short_double_space);

    size_t tc_end = response_text.find("<|end|>", tc_start == std::string::npos ? 0 : tc_start + tool_tag_len);
    if (tc_start != std::string::npos && tc_end != std::string::npos) {
        std::string tool_block = response_text.substr(tc_start + tool_tag_len, tc_end - (tc_start + tool_tag_len));
        size_t split_pos = tool_block.find(tool_msg_split);
        if (split_pos != std::string::npos) {
            std::string meta_part = tool_block.substr(0, split_pos);
            size_t name_end = meta_part.find_first_of(" <\n\r\t");
            result.tool_name = meta_part.substr(0, name_end);

            std::string args_part = tool_block.substr(split_pos + tool_msg_split.length());
            size_t call_pos = args_part.find(tool_call_marker);
            if (call_pos != std::string::npos) {
                args_part = args_part.substr(0, call_pos);
            }
            while (!args_part.empty() && (args_part.back() == ' ' || args_part.back() == '\n' || args_part.back() == '\r' || args_part.back() == '\t')) {
                args_part.pop_back();
            }
            while (!args_part.empty() && (args_part.front() == ' ' || args_part.front() == '\n' || args_part.front() == '\r' || args_part.front() == '\t')) {
                args_part.erase(args_part.begin());
            }
            if (!args_part.empty()) {
                try {
                    result.tool_args = nlohmann::json::parse(args_part).dump();
                } catch (...) {
                    result.tool_args = args_part;
                }
                result.tool_args = normalize_tool_args(result.tool_name, result.tool_args);
            }
            return result;
        }
        else {
            // Compact legacy form: functions.<name>{...}<|call|>
            size_t json_start = tool_block.find('{');
            if (json_start != std::string::npos) {
                std::string name_part = tool_block.substr(0, json_start);
                size_t name_end = name_part.find_first_of(" <\n\r\t");
                result.tool_name = name_part.substr(0, name_end);

                std::string args_part = tool_block.substr(json_start);
                size_t call_pos = args_part.find(tool_call_marker);
                if (call_pos != std::string::npos) {
                    args_part = args_part.substr(0, call_pos);
                }
                while (!args_part.empty() && (args_part.back() == ' ' || args_part.back() == '\n' || args_part.back() == '\r' || args_part.back() == '\t')) {
                    args_part.pop_back();
                }
                if (!args_part.empty()) {
                    try {
                        result.tool_args = nlohmann::json::parse(args_part).dump();
                    } catch (...) {
                        result.tool_args = args_part;
                    }
                    result.tool_args = normalize_tool_args(result.tool_name, result.tool_args);
                    return result;
                }
            }
        }
    }

    // Fallback parser for generic <tool_call>{"name":...,"arguments":...}</tool_call> format.
    const std::string xml_tool_start = "<tool_call>";
    const std::string xml_tool_end = "</tool_call>";
    size_t xml_start = response_text.find(xml_tool_start);
    size_t xml_end = response_text.find(xml_tool_end, xml_start == std::string::npos ? 0 : xml_start + xml_tool_start.size());
    if (xml_start != std::string::npos && xml_end != std::string::npos) {
        std::string json_str = response_text.substr(xml_start + xml_tool_start.size(), xml_end - (xml_start + xml_tool_start.size()));
        try {
            auto j = nlohmann::json::parse(json_str);
            if (j.contains("name")) {
                result.tool_name = j["name"].get<std::string>();
            }
            if (j.contains("arguments")) {
                result.tool_args = j["arguments"].is_string() ? j["arguments"].get<std::string>() : j["arguments"].dump();
                result.tool_args = normalize_tool_args(result.tool_name, result.tool_args);
            }
        } catch (...) {}
    }

    return result;
}

int GPT_OSS::_sample_in_tool(buffer<bf16>& logits) {
    // to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
    std::vector<int> allowed_tokens;
    switch (current_state) {
        case STATE_EXPECT_TO_FUNCTIONS: {
            allowed_tokens = get_to_functions_tokens();
            break;
        }
        case STATE_EXPECT_FUNCTION_NAME: {
            allowed_tokens = get_function_name_tokens();
            break;
        }
        case STATE_EXPECT_CONSTRAIN: {
            auto constrain_tokens = tokenizer->encode(" <|constrain|>");
            auto message_tokens = tokenizer->encode("json");
            allowed_tokens.insert(allowed_tokens.end(),
                constrain_tokens.begin(), constrain_tokens.end());
            allowed_tokens.insert(allowed_tokens.end(),
                message_tokens.begin(), message_tokens.end());
            break;
        }
        case STATE_EXPECT_MESSAGE: {
            auto tokens = tokenizer->encode("<|message|>");
            allowed_tokens = { tokens[0] };
            break;
        }
        //case STATE_IN_JSON: {
        //    header_print("JSONSSS", "SSSS");
        //    return this->sampler->sample(logits);
        //    //allowed_tokens = constrain_by_json_schema(logits);
        //    //break;
        //}
    }
    mask_logits(logits, allowed_tokens);
    int token_id = this->sampler->sample(logits);
    update_state(token_id);
    return token_id;
    //return this->sampler->sample(logits);
}

std::vector<int> GPT_OSS::get_to_functions_tokens() {
    const std::string target = " to=functions.";

    std::string remaining = target.substr(accumulated_text.length());

    auto remaining_tokens = tokenizer->encode(remaining);

    if (remaining_tokens.empty()) {
        return {};
    }

    return { remaining_tokens[0] };
}

std::vector<int> GPT_OSS::get_function_name_tokens() {
    std::vector<int> allowed;

    for (const auto& tool : tools) {
        std::string name = tool["function"]["name"];

        auto tokens = this->tokenizer->encode(name);
        for (int token : tokens) {
            allowed.push_back(token);
        }
    }

    return allowed;
}

void GPT_OSS::update_state(int token_id) {
    std::string token_str = tokenizer->decode({ token_id });
    accumulated_text += token_str;

    switch (current_state) {
        case STATE_EXPECT_TO_FUNCTIONS: {
            if (accumulated_text.find("to=functions.") != std::string::npos) {
                current_state = STATE_EXPECT_FUNCTION_NAME;
                accumulated_text.clear();
                header_print("TOOLCALLING", "Completed 'to=functions.', expecting function name");
            }
            break;
        }
        case STATE_EXPECT_FUNCTION_NAME: {
            //bool found = false;
            for (const auto& tool : tools) {
                std::string name = tool["function"]["name"];
                if (accumulated_text == name) {
                    current_state = STATE_EXPECT_CONSTRAIN;
                    accumulated_text.clear();
                    //found = true;
                    header_print("TOOLCALLING", "Matched function: " + name);
                    break;
                }
            }
            break;
        }
        case STATE_EXPECT_CONSTRAIN: {
            if (token_str.find("json") != std::string::npos) {
                current_state = STATE_EXPECT_MESSAGE;
                accumulated_text.clear();
                header_print("TOOLCALLING", "Starting MESSAGE generation");
            }
            break;
        }
        case STATE_EXPECT_MESSAGE: {
            if (token_str.find("<|message|>") != std::string::npos) {
                current_state = STATE_COMPLETE;
                accumulated_text.clear();
            }
            break;
        }
    }
}

void GPT_OSS::mask_logits(buffer<bf16>& logits, const std::vector<int>& allowed_tokens) {
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    std::unordered_set<int> allowed_set(allowed_tokens.begin(), allowed_tokens.end());

    for (int i = 0; i < logits.size(); i++) {
        if (allowed_set.find(i) == allowed_set.end()) {
            logits[i] = static_cast<bf16>(NEG_INF);
        }
    }
}

StreamResult GPT_OSS::parse_stream_content(const std::string content) {
    //header_print("GPTOSSHERE", content);

    auto parse_function_envelope = [](const std::string& text, std::string& out_name, std::string& out_args) -> bool {
        std::string s = text;
        while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r' || s.front() == '\t')) {
            s.erase(s.begin());
        }
        while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t')) {
            s.pop_back();
        }
        if (s.empty()) {
            return false;
        }

        try {
            auto j = nlohmann::json::parse(s);
            if (!j.is_object()) {
                return false;
            }

            if (j.contains("type") && j["type"].is_string() && j["type"].get<std::string>() == "function" &&
                j.contains("name") && j["name"].is_string()) {
                out_name = j["name"].get<std::string>();
                if (j.contains("parameters")) {
                    out_args = j["parameters"].is_string() ? j["parameters"].get<std::string>() : j["parameters"].dump();
                } else {
                    out_args = "{}";
                }
                out_args = normalize_tool_args(out_name, out_args);
                return true;
            }

            if (j.contains("name") && j["name"].is_string() && j.contains("arguments")) {
                out_name = j["name"].get<std::string>();
                out_args = j["arguments"].is_string() ? j["arguments"].get<std::string>() : j["arguments"].dump();
                out_args = normalize_tool_args(out_name, out_args);
                return true;
            }
        } catch (...) {}

        return false;
    };

    const std::string MARKER_REASONING = "<|start|>assistant<|channel|>analysis<|message|>";
    const std::string MARKER_REASONING_SHORT = "<|channel|>analysis<|message|>";
    const std::string MARKER_NORMAL = "<|start|>assistant<|channel|>final<|message|>";
    const std::string MARKER_NORMAL_SHORT = "<|channel|>final<|message|>";
    const std::string MARKER_END = "<|end|>";    

    // Markers for Tool Calling
    const std::string MARKER_TOOL_START = "<|start|>assistant<|channel|>commentary to=functions.";
    const std::string MARKER_TOOL_START_SHORT = "<|channel|>commentary to=functions.";
    const std::string MARKER_TOOL_START_CONSTRAIN = "<|start|>assistant<|channel|>final <|constrain|>functions.";
    const std::string MARKER_TOOL_START_CONSTRAIN_NO_SPACE = "<|start|>assistant<|channel|>final<|constrain|>functions.";
    const std::string MARKER_TOOL_START_CONSTRAIN_SHORT = "<|channel|>final <|constrain|>functions.";
    const std::string MARKER_TOOL_START_CONSTRAIN_SHORT_NO_SPACE = "<|channel|>final<|constrain|>functions.";
    const std::string MARKER_TOOL_START_PLAIN = "<|start|>assistant<|channel|>final functions.";
    const std::string MARKER_TOOL_START_PLAIN_DOUBLE_SPACE = "<|start|>assistant<|channel|>final  functions.";
    const std::string MARKER_TOOL_START_PLAIN_SHORT = "<|channel|>final functions.";
    const std::string MARKER_TOOL_START_PLAIN_SHORT_DOUBLE_SPACE = "<|channel|>final  functions.";
    //const std::string MARKER_TOOL_START = "<|start|>assistant<|channel|>commentary <|constrain|>functions.";
    const std::string MARKER_TOOL_SPLIT = "<|message|>"; 
    const std::string MARKER_TOOL_END = "<|end|>";   

    StreamResult result;
    buffer_ += content; // Append new chunk to buffer

    // for test
    //result.content = content;
    //result.type = current_mode_;
    //return result;
    // for test

    while (true) {
        if (waiting_for_header_) {
            size_t pos_reason = buffer_.find(MARKER_REASONING);
            size_t pos_reason_short = buffer_.find(MARKER_REASONING_SHORT);
            size_t pos_normal = buffer_.find(MARKER_NORMAL);
            size_t pos_normal_short = buffer_.find(MARKER_NORMAL_SHORT);
            size_t pos_tool = buffer_.find(MARKER_TOOL_START);
            size_t pos_tool_short = buffer_.find(MARKER_TOOL_START_SHORT);
            size_t pos_tool_constrain = buffer_.find(MARKER_TOOL_START_CONSTRAIN);
            size_t pos_tool_constrain_no_space = buffer_.find(MARKER_TOOL_START_CONSTRAIN_NO_SPACE);
            size_t pos_tool_constrain_short = buffer_.find(MARKER_TOOL_START_CONSTRAIN_SHORT);
            size_t pos_tool_constrain_short_no_space = buffer_.find(MARKER_TOOL_START_CONSTRAIN_SHORT_NO_SPACE);
            size_t pos_tool_plain = buffer_.find(MARKER_TOOL_START_PLAIN);
            size_t pos_tool_plain_double_space = buffer_.find(MARKER_TOOL_START_PLAIN_DOUBLE_SPACE);
            size_t pos_tool_plain_short = buffer_.find(MARKER_TOOL_START_PLAIN_SHORT);
            size_t pos_tool_plain_short_double_space = buffer_.find(MARKER_TOOL_START_PLAIN_SHORT_DOUBLE_SPACE);

            // Find which marker appears first
            size_t pos_first = std::string::npos;
            StreamEventType next_mode = StreamEventType::REASONING; // default
            size_t marker_len = 0;

            if (pos_reason != std::string::npos) {
                pos_first = pos_reason;
                next_mode = StreamEventType::REASONING;
                marker_len = MARKER_REASONING.length();
            }
            if (pos_reason_short != std::string::npos && (pos_first == std::string::npos || pos_reason_short < pos_first)) {
                pos_first = pos_reason_short;
                next_mode = StreamEventType::REASONING;
                marker_len = MARKER_REASONING_SHORT.length();
            }
            if (pos_normal != std::string::npos && (pos_first == std::string::npos || pos_normal < pos_first)) {
                pos_first = pos_normal;
                next_mode = StreamEventType::CONTENT;
                marker_len = MARKER_NORMAL.length();
            }
            if (pos_normal_short != std::string::npos && (pos_first == std::string::npos || pos_normal_short < pos_first)) {
                pos_first = pos_normal_short;
                next_mode = StreamEventType::CONTENT;
                marker_len = MARKER_NORMAL_SHORT.length();
            }
            if (pos_tool != std::string::npos && (pos_first == std::string::npos || pos_tool < pos_first)) {
                pos_first = pos_tool;
                next_mode = StreamEventType::WAITING; // Special mode for buffering tool
                marker_len = MARKER_TOOL_START.length();
            }
            if (pos_tool_short != std::string::npos && (pos_first == std::string::npos || pos_tool_short < pos_first)) {
                pos_first = pos_tool_short;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_SHORT.length();
            }
            if (pos_tool_constrain != std::string::npos && (pos_first == std::string::npos || pos_tool_constrain < pos_first)) {
                pos_first = pos_tool_constrain;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_CONSTRAIN.length();
            }
            if (pos_tool_constrain_no_space != std::string::npos && (pos_first == std::string::npos || pos_tool_constrain_no_space < pos_first)) {
                pos_first = pos_tool_constrain_no_space;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_CONSTRAIN_NO_SPACE.length();
            }
            if (pos_tool_constrain_short != std::string::npos && (pos_first == std::string::npos || pos_tool_constrain_short < pos_first)) {
                pos_first = pos_tool_constrain_short;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_CONSTRAIN_SHORT.length();
            }
            if (pos_tool_constrain_short_no_space != std::string::npos && (pos_first == std::string::npos || pos_tool_constrain_short_no_space < pos_first)) {
                pos_first = pos_tool_constrain_short_no_space;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_CONSTRAIN_SHORT_NO_SPACE.length();
            }
            if (pos_tool_plain != std::string::npos && (pos_first == std::string::npos || pos_tool_plain < pos_first)) {
                pos_first = pos_tool_plain;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_PLAIN.length();
            }
            if (pos_tool_plain_double_space != std::string::npos && (pos_first == std::string::npos || pos_tool_plain_double_space < pos_first)) {
                pos_first = pos_tool_plain_double_space;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_PLAIN_DOUBLE_SPACE.length();
            }
            if (pos_tool_plain_short != std::string::npos && (pos_first == std::string::npos || pos_tool_plain_short < pos_first)) {
                pos_first = pos_tool_plain_short;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_PLAIN_SHORT.length();
            }
            if (pos_tool_plain_short_double_space != std::string::npos && (pos_first == std::string::npos || pos_tool_plain_short_double_space < pos_first)) {
                pos_first = pos_tool_plain_short_double_space;
                next_mode = StreamEventType::WAITING;
                marker_len = MARKER_TOOL_START_PLAIN_SHORT_DOUBLE_SPACE.length();
            }

            // found
            if (pos_first != std::string::npos) {
                waiting_for_header_ = false;
                current_mode_ = next_mode;

                // Remove garbage before the marker and the marker itself
                buffer_.erase(0, pos_first + marker_len);
                continue;
            }
            else {
                result.type = StreamEventType::WAITING;
                return result;
            }
        }


        if (current_mode_ == StreamEventType::WAITING) {
            size_t pos_end = buffer_.find(MARKER_TOOL_END);

            if (pos_end != std::string::npos) {
                // Format: Name <|constrain|>... <|message|> {JSON}
                std::string full_tool_str = buffer_.substr(0, pos_end);

                size_t pos_split = full_tool_str.find(MARKER_TOOL_SPLIT);
                if (pos_split != std::string::npos) {
                    // 1. Extract Name (Left side)
                    std::string meta_part = full_tool_str.substr(0, pos_split);
                    // Name is typically the first word
                    size_t name_end = meta_part.find_first_of(" <");
                    result.tool_name = meta_part.substr(0, name_end);

                    // 2. Extract JSON (Right side)
                    std::string args_part = full_tool_str.substr(pos_split + MARKER_TOOL_SPLIT.length());
                    size_t call_pos = args_part.find("<|call|>");
                    if (call_pos != std::string::npos) {
                        args_part = args_part.substr(0, call_pos);
                    }
                    while (!args_part.empty() && (args_part.back() == ' ' || args_part.back() == '\n' || args_part.back() == '\r' || args_part.back() == '\t')) {
                        args_part.pop_back();
                    }
                    while (!args_part.empty() && (args_part.front() == ' ' || args_part.front() == '\n' || args_part.front() == '\r' || args_part.front() == '\t')) {
                        args_part.erase(args_part.begin());
                    }
                    try {
                        result.tool_args_str = nlohmann::json::parse(args_part).dump();
                    } catch (...) {
                        result.tool_args_str = args_part;
                    }
                    result.tool_args_str = normalize_tool_args(result.tool_name, result.tool_args_str);

                    // 3. Set Result
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));
                }
                else {
                    // Compact legacy form: functions.<name>{...}<|call|>
                    size_t json_start = full_tool_str.find('{');
                    if (json_start != std::string::npos) {
                        std::string name_part = full_tool_str.substr(0, json_start);
                        size_t name_end = name_part.find_first_of(" <\n\r\t");
                        result.tool_name = name_part.substr(0, name_end);

                        std::string args_part = full_tool_str.substr(json_start);
                        size_t call_pos = args_part.find("<|call|>");
                        if (call_pos != std::string::npos) {
                            args_part = args_part.substr(0, call_pos);
                        }
                        while (!args_part.empty() && (args_part.back() == ' ' || args_part.back() == '\n' || args_part.back() == '\r' || args_part.back() == '\t')) {
                            args_part.pop_back();
                        }
                        try {
                            result.tool_args_str = nlohmann::json::parse(args_part).dump();
                        } catch (...) {
                            result.tool_args_str = args_part;
                        }
                        result.tool_args_str = normalize_tool_args(result.tool_name, result.tool_args_str);
                        result.type = StreamEventType::TOOL_DONE;
                        result.tool_id = "call_" + std::to_string(std::time(nullptr));
                    }
                    else {
                        // Fallback if format is wrong
                        result.content = "[Tool Parse Error]";
                        result.type = StreamEventType::CONTENT;
                    }
                }

                // Clean up and reset
                buffer_.erase(0, pos_end + MARKER_TOOL_END.length());
                waiting_for_header_ = true;

                // Return immediately, don't stream rest
                return result;
            }
            else {
                // Tool block not finished yet. Wait.
                result.type = StreamEventType::WAITING;
                return result;
            }
        }
        else {
            size_t pos_end = buffer_.find(MARKER_END);

            if (pos_end != std::string::npos) {
                // End marker found
                std::string segment = buffer_.substr(0, pos_end);
                if (current_mode_ == StreamEventType::CONTENT && parse_function_envelope(segment, result.tool_name, result.tool_args_str)) {
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));
                }
                else {
                    result.content += segment; // Output content before marker
                    result.type = current_mode_;
                }

                // Remove content + marker, ready for next section
                buffer_ = buffer_.substr(pos_end + MARKER_END.length());
                waiting_for_header_ = true; // Go back to waiting for next header
            }
            else {
                // No end marker, flush everything immediately
                if (current_mode_ == StreamEventType::CONTENT && parse_function_envelope(buffer_, result.tool_name, result.tool_args_str)) {
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));
                }
                else {
                    result.content += buffer_;
                    result.type = current_mode_;
                }
                buffer_.clear();
                break; // Done with this chunk
            }
        }
    }

    return result;
}

StreamResult GPT_OSS::parse_stream_content_final(const std::string content) {
    if (!content.empty()) {
        return parse_stream_content(content);
    }

    // If the model ended without emitting <|end|>, force-close pending blocks
    // so a detected tool header can still become TOOL_DONE.
    if (!buffer_.empty() || current_mode_ == StreamEventType::WAITING) {
        return parse_stream_content("<|end|>");
    }

    StreamResult result;
    result.type = StreamEventType::WAITING;
    return result;
}