#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

engine::core::BackendType parse_backend(const std::string & backend) {
    if (backend == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (backend == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (backend == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (backend == "metal") {
        return engine::core::BackendType::Metal;
    }
    if (backend == "best") {
        return engine::core::BackendType::BestAvailable;
    }

    throw std::runtime_error("Unsupported backend: " + backend);
}

engine::runtime::AudioBuffer read_audio(
    const std::filesystem::path & path
) {
    const auto wav = engine::audio::read_wav_f32(path);

    return engine::runtime::AudioBuffer{
        wav.sample_rate,
        wav.channels,
        wav.samples,
    };
}

} // namespace

class OmniVoice {
public:
    OmniVoice(
        const std::string & model_path,
        const std::string & ref_audio,
        const std::string & ref_text,
        const std::string & backend,
        int device,
        int threads,
        const std::string & generator_weight_type,
        const std::string & audio_tokenizer_weight_type
    )
        : ref_audio_(read_audio(ref_audio)),
          ref_text_(ref_text) {

        auto registry = engine::runtime::make_default_registry();

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = std::filesystem::path(model_path);
        load_request.family_hint = std::string("omnivoice");

        model_ = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend);
        session_options.backend.device = device;
        session_options.backend.threads = threads;

        session_options.options[
            "omnivoice.generator_weight_type"
        ] = generator_weight_type;

        session_options.options[
            "omnivoice.audio_tokenizer_weight_type"
        ] = audio_tokenizer_weight_type;

        const engine::runtime::TaskSpec task{
            engine::runtime::VoiceTaskKind::Tts,
            engine::runtime::RunMode::Offline,
        };

        session_ = model_->create_task_session(
            task,
            session_options
        );

        offline_ = dynamic_cast<
            engine::runtime::IOfflineVoiceTaskSession *
        >(session_.get());

        if (offline_ == nullptr) {
            throw std::runtime_error(
                "OmniVoice session does not support offline inference"
            );
        }
    }

    py::dict generate(
        const std::string & text,
        const std::string & language,
        int seed,
        int steps,
        float guidance_scale,
        float speed
    ) {
        if (text.empty()) {
            throw std::runtime_error("Text must not be empty");
        }

        engine::runtime::TaskRequest request;

        request.text_input = engine::runtime::Transcript{
            text,
            language,
        };

        engine::runtime::VoiceReference reference;
        reference.audio = ref_audio_;

        engine::runtime::VoiceCondition voice;
        voice.speaker = std::move(reference);

        request.voice = std::move(voice);

        request.options["language"] = language;
        request.options["reference_text"] = ref_text_;
        request.options["seed"] = std::to_string(seed);
        request.options["num_inference_steps"] =
            std::to_string(steps);
        request.options["guidance_scale"] =
            std::to_string(guidance_scale);
        request.options["speed"] =
            std::to_string(speed);

        engine::runtime::TaskResult result;

        {
            // The same session is reused, so protect it from concurrent use.
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);

            session_->prepare(
                engine::runtime::build_preparation_request(request)
            );

            result = offline_->run(request);
        }

        if (!result.audio_output.has_value()) {
            throw std::runtime_error(
                "OmniVoice returned no audio output"
            );
        }

        const auto & audio = *result.audio_output;

        py::array_t<float> samples(audio.samples.size());
        std::memcpy(
            samples.mutable_data(),
            audio.samples.data(),
            audio.samples.size() * sizeof(float)
        );

        py::dict output;
        output["samples"] = std::move(samples);
        output["sample_rate"] = audio.sample_rate;
        output["channels"] = audio.channels;

        const double frames =
            audio.channels > 0
                ? static_cast<double>(audio.samples.size()) /
                      static_cast<double>(audio.channels)
                : 0.0;

        output["duration"] =
            audio.sample_rate > 0
                ? frames / static_cast<double>(audio.sample_rate)
                : 0.0;

        return output;
    }

private:
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> model_;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session_;

    engine::runtime::IOfflineVoiceTaskSession * offline_ = nullptr;

    engine::runtime::AudioBuffer ref_audio_;
    std::string ref_text_;

    std::mutex mutex_;
};

PYBIND11_MODULE(omnivoice_cpp, module) {
    module.doc() =
        "Simple Python binding for audio.cpp OmniVoice";

    py::class_<OmniVoice>(module, "OmniVoice")
        .def(
            py::init<
                const std::string &,
                const std::string &,
                const std::string &,
                const std::string &,
                int,
                int,
                const std::string &,
                const std::string &
            >(),
            py::arg("model_path"),
            py::arg("ref_audio"),
            py::arg("ref_text"),
            py::arg("backend") = "cuda",
            py::arg("device") = 0,
            py::arg("threads") = 4,
            py::arg("generator_weight_type") = "f16",
            py::arg("audio_tokenizer_weight_type") = "f16"
        )
        .def(
            "generate",
            &OmniVoice::generate,
            py::arg("text"),
            py::arg("language") = "Bengali",
            py::arg("seed") = 42,
            py::arg("steps") = 20,
            py::arg("guidance_scale") = 2.0F,
            py::arg("speed") = 1.0F
        );
}
