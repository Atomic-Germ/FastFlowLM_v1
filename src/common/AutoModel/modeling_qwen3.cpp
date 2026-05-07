/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_qwen3.hpp"


/************              Qwen3 family            **************/
Qwen3::Qwen3(xrt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3") {}

void Qwen3::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // lm_config->model_type == qwen3
    this->lm_engine = std::make_unique<qwen3_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_think = (model_info["size"] == 600000000)? false : true;
    this->enable_tool = (model_info["size"] > 1700000000)? true : false;

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Qwen3::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    if (this->is_first_prompt == false) {
        tokens.insert(tokens.begin(), 198);
    }
    this->is_first_prompt = false;

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string Qwen3::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Qwen3::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

NonStreamResult Qwen3::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string name, arguments;
    std::string content, reasoning_content;

    std::string think_start_tag = "<think>";
    std::string think_end_tag = "</think>";
    std::string tool_start_tag = "<tool_call>";
    std::string tool_end_tag = "</tool_call>";

    size_t think_start_pos = response_text.find(think_start_tag);
    size_t think_end_pos = response_text.find(think_end_tag);
    size_t tool_start_pos = response_text.find(tool_start_tag);
    size_t tool_end_pos = response_text.find(tool_end_tag);

    bool is_reasoning = !(think_start_pos == std::string::npos || think_end_pos == std::string::npos);
    bool is_tool = !(tool_start_pos == std::string::npos || tool_end_pos == std::string::npos);
    bool is_content = !is_tool;

    if (is_reasoning) {
        // Find reasoning part
        think_start_pos += think_start_tag.length();
        std::string reasoning_str = response_text.substr(think_start_pos, think_end_pos - think_start_pos);
        result.reasoning_content = reasoning_str;
    }

    if (is_tool) {
        // Find tool calling part
        tool_start_pos += tool_start_tag.length();
        std::string json_str = response_text.substr(tool_start_pos, tool_end_pos - tool_start_pos);
        // Parse "name" 
        std::string key_name = "\"name\": \"";
        size_t name_start = json_str.find(key_name);
        if (name_start != std::string::npos) {
            name_start += key_name.length();
            size_t name_end = json_str.find("\"", name_start);
            if (name_end != std::string::npos) {
                name = json_str.substr(name_start, name_end - name_start);
            }
        }
        // Parse "arguments"
        std::string key_args = "\"arguments\":";
        size_t args_pos = json_str.find(key_args);
        if (args_pos != std::string::npos) {
            size_t brace_start = json_str.find("{", args_pos);
            size_t brace_end = json_str.rfind("}"); // Find the last closing brace

            if (brace_start != std::string::npos && brace_end != std::string::npos && brace_end > brace_start) {
                arguments = json_str.substr(brace_start, brace_end - brace_start);
            }
        }

        result.tool_name = name;
        result.tool_args = arguments;

    }
    else if (is_content) {
        std::string content_str = response_text.substr(think_end_pos + think_end_tag.length());
        result.content = content_str;
    }

    return result;
}

StreamResult Qwen3::parse_stream_content(const std::string content) {
    return _shared_think_tool_calling_pasrsed(content);
}

/************              Qwen3_IT family            **************/
Qwen3_IT::Qwen3_IT(xrt::device* npu_device_inst) : AutoModel(npu_device_inst) {}

void Qwen3_IT::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);

    // lm_config->model_type == qwen3
    this->lm_engine = std::make_unique<qwen3_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Qwen3_IT::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_IT::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    if (!tools.empty())
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3_IT::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    if (this->is_first_prompt == false) {
        tokens.insert(tokens.begin(), 198);
    }
    this->is_first_prompt = false;
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string Qwen3_IT::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Qwen3_IT::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

// Non-stream
NonStreamResult Qwen3_IT::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string name, arguments;

    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";

    size_t start_pos = response_text.find(start_tag);
    size_t end_pos = response_text.find(end_tag);

    if (start_pos == std::string::npos || end_pos == std::string::npos) {
        // pure content
        result.content = response_text;
        return result;
    }

    start_pos += start_tag.length();
    std::string json_str = response_text.substr(start_pos, end_pos - start_pos);

    // Parse "name" 
    std::string key_name = "\"name\": \"";
    size_t name_start = json_str.find(key_name);
    if (name_start != std::string::npos) {
        name_start += key_name.length();
        size_t name_end = json_str.find("\"", name_start);
        if (name_end != std::string::npos) {
            name = json_str.substr(name_start, name_end - name_start);
        }
    }

    // Parse "arguments"
    std::string key_args = "\"arguments\":";
    size_t args_pos = json_str.find(key_args);
    if (args_pos != std::string::npos) {
        size_t brace_start = json_str.find("{", args_pos);
        size_t brace_end = json_str.rfind("}"); // Find the last closing brace

        if (brace_start != std::string::npos && brace_end != std::string::npos && brace_end > brace_start) {
            arguments = json_str.substr(brace_start, brace_end - brace_start);
        }
    }

    result.tool_name = name;
    result.tool_args = arguments;

    return result;
}

// Stream
StreamResult Qwen3_IT::parse_stream_content(const std::string content) {
    std::string tool_start_tag = "<tool_call>";
    std::string tool_end_tag = "</tool_call>";

    StreamResult result;
    result.type = StreamEventType::CONTENT; 

    if (content.find(tool_start_tag) != std::string::npos) {
        is_in_tool_block_ = true;
        tool_name_.clear();
        result.type = StreamEventType::WAITING;
        return result;
    }

    if (content.find("</tool_call>") != std::string::npos) {
        is_in_tool_block_ = false;

        try {
            auto j = nlohmann::json::parse(tool_name_);

            result.type = StreamEventType::TOOL_DONE;
            //result.tool_id = generate_id();
            result.tool_id = "call_" + std::to_string(std::time(nullptr));

            if (j.contains("name")) {
                result.tool_name = j["name"].get<std::string>();
            }

            if (j.contains("arguments")) {
                if (j["arguments"].is_string()) {
                    result.tool_args_str = j["arguments"].get<std::string>();
                }
                else {
                    result.tool_args_str = j["arguments"].dump();
                }
            }
        }
        catch (...) {
            result.type = StreamEventType::CONTENT;
            result.content = "[Error parsing tool call]";
        }
        return result;
    }

    if (is_in_tool_block_) {
        tool_name_ += content;
        result.type = StreamEventType::WAITING; 
        return result;
    }

    result.content = content;
    return result;

}

/************              Qwen3_TK family            **************/
Qwen3_TK::Qwen3_TK(xrt::device* npu_device_inst) : AutoModel(npu_device_inst) {}

void Qwen3_TK::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);

    // lm_config->model_type == qwen3
    this->lm_engine = std::make_unique<qwen3_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Qwen3_TK::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
    this->think_marker_id = this->tokenizer->encode("<think>")[0];
}

std::string Qwen3_TK::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3_TK::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
      if (this->is_first_prompt == false) {
        tokens.insert(tokens.begin(), 198);
    }
    this->is_first_prompt = false;
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string Qwen3_TK::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    os << "<think>\n\n";
    result = this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    result = "<think>\n\n" + result;
    return result;
}

std::string Qwen3_TK::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

NonStreamResult Qwen3_TK::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string name, arguments;
    std::string content, reasoning_content;

    std::string think_start_tag = "<think>";
    std::string think_end_tag = "</think>";
    std::string tool_start_tag = "<tool_call>";
    std::string tool_end_tag = "</tool_call>";

    size_t think_start_pos = response_text.find(think_start_tag);
    size_t think_end_pos = response_text.find(think_end_tag);
    size_t tool_start_pos = response_text.find(tool_start_tag);
    size_t tool_end_pos = response_text.find(tool_end_tag);

    bool is_reasoning = !(think_start_pos == std::string::npos || think_end_pos == std::string::npos);
    bool is_tool = !(tool_start_pos == std::string::npos || tool_end_pos == std::string::npos);
    bool is_content = !is_tool;

    if (is_reasoning) {
        // Find reasoning part
        think_start_pos += think_start_tag.length();
        std::string reasoning_str = response_text.substr(think_start_pos, think_end_pos - think_start_pos);
        result.reasoning_content = reasoning_str;
    }

    if (is_tool) {
        // Find tool calling part
        tool_start_pos += tool_start_tag.length();
        std::string json_str = response_text.substr(tool_start_pos, tool_end_pos - tool_start_pos);
        // Parse "name" 
        std::string key_name = "\"name\": \"";
        size_t name_start = json_str.find(key_name);
        if (name_start != std::string::npos) {
            name_start += key_name.length();
            size_t name_end = json_str.find("\"", name_start);
            if (name_end != std::string::npos) {
                name = json_str.substr(name_start, name_end - name_start);
            }
        }
        // Parse "arguments"
        std::string key_args = "\"arguments\":";
        size_t args_pos = json_str.find(key_args);
        if (args_pos != std::string::npos) {
            size_t brace_start = json_str.find("{", args_pos);
            size_t brace_end = json_str.rfind("}"); // Find the last closing brace

            if (brace_start != std::string::npos && brace_end != std::string::npos && brace_end > brace_start) {
                arguments = json_str.substr(brace_start, brace_end - brace_start);
            }
        }

        result.tool_name = name;
        result.tool_args = arguments;

    }
    else if (is_content) {
        std::string content_str = response_text.substr(think_end_pos + think_end_tag.length());
        result.content = content_str;
    }

    return result;
}


StreamResult Qwen3_TK::parse_stream_content(const std::string content) {
    return _shared_think_tool_calling_pasrsed(content);
}

/************              DeepSeek_r1_0528_8b family            **************/
DeepSeek_r1_0528_8b::DeepSeek_r1_0528_8b(xrt::device* npu_device_inst) : AutoModel(npu_device_inst) {}

void DeepSeek_r1_0528_8b::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // model_type == llama
    this->lm_engine = std::make_unique<qwen3_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;
    config.rep_penalty = 1.05;
    config.freq_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void DeepSeek_r1_0528_8b::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string DeepSeek_r1_0528_8b::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.extra_context = this->extra_context;

    if (!tools.empty()) {
        // The DeepSeek-R1-0528 template has no {%- if tools %} section, so tool
        // definitions must be injected manually as a system message.
        // The model's tokenizer vocab uses <tool_call>/<think> tokens (same as
        // Qwen3), so we use the same system-message format as Qwen3-Instruct.
        std::string tool_system =
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (auto& tool : tools) {
            tool_system += "\n" + tool.dump();
        }
        tool_system +=
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

        // Build modified messages with the tool-definition system message first.
        nlohmann::ordered_json modified_messages = nlohmann::ordered_json::array();
        if (!messages.empty() && messages[0]["role"] == "system") {
            // Prepend tool definitions to an existing system message.
            std::string existing = messages[0]["content"].is_string()
                ? messages[0]["content"].get<std::string>() : "";
            nlohmann::ordered_json sys = messages[0];
            sys["content"] = tool_system + (existing.empty() ? "" : "\n\n" + existing);
            modified_messages.push_back(sys);
            for (size_t i = 1; i < messages.size(); ++i)
                modified_messages.push_back(messages[i]);
        } else {
            modified_messages.push_back({{"role", "system"}, {"content", tool_system}});
            for (auto& msg : messages)
                modified_messages.push_back(msg);
        }
        inputs.messages = modified_messages;
    } else {
        inputs.messages = messages;
    }

    return this->chat_tmpl->apply(inputs);
}

bool DeepSeek_r1_0528_8b::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    if (this->is_first_prompt == false) {
        tokens.insert(tokens.begin(), 198);
    }
    this->is_first_prompt = false;
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string DeepSeek_r1_0528_8b::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result = this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    return result;
}

std::string DeepSeek_r1_0528_8b::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

NonStreamResult DeepSeek_r1_0528_8b::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    const std::string think_start_tag = "<think>";
    const std::string think_end_tag   = "</think>";
    // This model generates tool calls as a ```json code block containing
    // {"name": "...", "arguments": {...}} - matching the system message we inject.
    const std::string code_start_tag  = "```json";
    const std::string code_end_tag    = "```";

    size_t think_start_pos = response_text.find(think_start_tag);
    size_t think_end_pos   = response_text.find(think_end_tag);

    bool is_reasoning = !(think_start_pos == std::string::npos || think_end_pos == std::string::npos);

    if (is_reasoning) {
        size_t body_start = think_start_pos + think_start_tag.length();
        result.reasoning_content = response_text.substr(body_start, think_end_pos - body_start);
    }

    // Look for ```json block containing {"name": ..., "arguments": ...}
    size_t code_start_pos = response_text.find(code_start_tag);
    if (code_start_pos != std::string::npos) {
        size_t json_start = code_start_pos + code_start_tag.length();
        // skip newline after ```json
        if (json_start < response_text.size() && response_text[json_start] == '\n')
            json_start++;
        // find the closing ``` (must be on its own - search after json_start)
        size_t code_end_pos = response_text.find(code_end_tag, json_start);
        if (code_end_pos != std::string::npos) {
            std::string json_str = response_text.substr(json_start, code_end_pos - json_start);
            // trim trailing whitespace
            while (!json_str.empty() && (json_str.back() == '\n' || json_str.back() == '\r' || json_str.back() == ' '))
                json_str.pop_back();
            try {
                auto j = nlohmann::json::parse(json_str);
                if (j.contains("name") && j.contains("arguments")) {
                    result.tool_name = j["name"].get<std::string>();
                    result.tool_args = j["arguments"].is_string()
                        ? j["arguments"].get<std::string>()
                        : j["arguments"].dump();
                    return result;
                }
            } catch (...) {}
        }
    }

    // No tool call found - return as content
    size_t content_start = (think_end_pos != std::string::npos)
        ? think_end_pos + think_end_tag.length()
        : 0;
    result.content = response_text.substr(content_start);
    return result;
}


StreamResult DeepSeek_r1_0528_8b::parse_stream_content(const std::string content) {
    // This model outputs tool calls as:
    //   ```json\n{"name":"...","arguments":{...}}\n```
    // We detect that code block, parse it, and emit TOOL_DONE.
    // Outside of that, <think>/<think> is handled for reasoning.
    const std::string THINK_START = "<think>";
    const std::string THINK_END   = "</think>";
    const std::string CODE_START  = "```json";
    const std::string CODE_END    = "```";

    StreamResult result;
    buffer_ += content;

    while (true) {
        // ── Inside a buffered tool call code block ──────────────────────────
        if (is_in_tool_block_) {
            size_t end_pos = buffer_.find(CODE_END);
            if (end_pos != std::string::npos) {
                tool_name_ += buffer_.substr(0, end_pos);
                buffer_ = buffer_.substr(end_pos + CODE_END.length());
                is_in_tool_block_ = false;
                // trim trailing whitespace from accumulated JSON
                while (!tool_name_.empty() && (tool_name_.back() == '\n' || tool_name_.back() == '\r' || tool_name_.back() == ' '))
                    tool_name_.pop_back();
                try {
                    auto j = nlohmann::json::parse(tool_name_);
                    if (j.contains("name") && j.contains("arguments")) {
                        result.type = StreamEventType::TOOL_DONE;
                        result.tool_id = "call_" + std::to_string(std::time(nullptr));
                        result.tool_name = j["name"].get<std::string>();
                        result.tool_args_str = j["arguments"].is_string()
                            ? j["arguments"].get<std::string>()
                            : j["arguments"].dump();
                    } else {
                        // Not a tool call JSON - emit as plain content
                        result.type = StreamEventType::CONTENT;
                        result.content = CODE_START + "\n" + tool_name_ + "\n" + CODE_END;
                    }
                } catch (...) {
                    result.type = StreamEventType::CONTENT;
                    result.content = CODE_START + "\n" + tool_name_ + "\n" + CODE_END;
                }
                return result;
            } else {
                tool_name_ += buffer_;
                buffer_.clear();
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        // ── CONTENT mode: look for ```json tool call start or <think> ────────
        if (current_mode_ == StreamEventType::CONTENT) {
            size_t code_pos  = buffer_.find(CODE_START);
            size_t think_pos = buffer_.find(THINK_START);

            size_t first = std::min(code_pos, think_pos);
            if (first != std::string::npos) {
                if (first > 0) {
                    // Emit content that precedes the marker, but hold back
                    // any trailing bytes that could be a partial marker prefix.
                    size_t safe_end = first;
                    for (size_t len = std::min(CODE_START.length() - 1, first); len > 0; --len) {
                        if (buffer_.substr(first - len, len) == CODE_START.substr(0, len) ||
                            buffer_.substr(first - len, len) == THINK_START.substr(0, len)) {
                            safe_end = first - len;
                            break;
                        }
                    }
                    if (safe_end == 0) break; // nothing safe to emit yet
                    result.content = buffer_.substr(0, safe_end);
                    result.type = StreamEventType::CONTENT;
                    buffer_ = buffer_.substr(safe_end);
                    return result;
                }
                if (code_pos == 0) {
                    is_in_tool_block_ = true;
                    tool_name_.clear();
                    buffer_ = buffer_.substr(CODE_START.length());
                    if (!buffer_.empty() && buffer_[0] == '\n')
                        buffer_ = buffer_.substr(1);
                    result.type = StreamEventType::WAITING;
                    return result;
                }
                // think_pos == 0
                buffer_ = buffer_.substr(THINK_START.length());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        }

        // ── REASONING mode: look for </think> ───────────────────────────────
        if (current_mode_ == StreamEventType::REASONING) {
            size_t end_pos = buffer_.find(THINK_END);
            if (end_pos != std::string::npos) {
                if (end_pos > 0) {
                    result.content = buffer_.substr(0, end_pos);
                    result.type = StreamEventType::REASONING;
                    buffer_ = buffer_.substr(end_pos);
                    return result;
                }
                buffer_ = buffer_.substr(THINK_END.length());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }

        // ── No marker found: emit what is safely emittable ───────────────────
        if (!buffer_.empty()) {
            // Hold back trailing bytes that could be the start of a marker
            std::string candidates[] = {CODE_START, THINK_START, THINK_END};
            size_t hold = 0;
            for (auto& m : candidates) {
                for (size_t len = std::min(m.length() - 1, buffer_.length()); len > 0; --len) {
                    if (buffer_.substr(buffer_.length() - len) == m.substr(0, len)) {
                        hold = std::max(hold, len);
                        break;
                    }
                }
            }
            size_t emit_len = buffer_.length() - hold;
            if (emit_len > 0) {
                result.content = buffer_.substr(0, emit_len);
                result.type = current_mode_;
                buffer_ = buffer_.substr(emit_len);
                return result;
            }
        }
        break;
    }

    result.type = current_mode_;
    return result;
}