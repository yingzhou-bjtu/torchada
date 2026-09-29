#include <torch/extension.h>

#include <mufft.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace {

struct PlanKey {
    int n;
    int device;
    std::uint64_t stream;

    bool operator==(const PlanKey& other) const {
        return n == other.n && device == other.device && stream == other.stream;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const {
        return (static_cast<std::size_t>(key.n) << 32) ^
            static_cast<std::size_t>(key.device) ^
            static_cast<std::size_t>(key.stream);
    }
};

std::mutex plan_mutex;
std::unordered_map<PlanKey, mufftHandle, PlanKeyHash> plans;

mufftHandle get_plan(int n, int device, c10::musa::MUSAStream stream) {
    std::lock_guard<std::mutex> lock(plan_mutex);
    const PlanKey key{n, device, static_cast<std::uint64_t>(stream.id())};
    auto it = plans.find(key);
    if (it != plans.end()) {
        return it->second;
    }

    mufftHandle plan = nullptr;
    const mufftResult result = mufftPlan1d(&plan, n, MUFFT_C2C, 1);
    TORCH_CHECK(
        result == MUFFT_SUCCESS,
        "muFFT C2C plan creation failed for n=",
        n,
        ", result=",
        static_cast<int>(result));
    const mufftResult stream_result = mufftSetStream(plan, stream);
    TORCH_CHECK(
        stream_result == MUFFT_SUCCESS,
        "muFFT stream binding failed, result=",
        static_cast<int>(stream_result));
    plans.emplace(key, plan);
    return plan;
}

}  // namespace

torch::Tensor musa_irfft_graph(torch::Tensor spectrum, int64_t n) {
    TORCH_CHECK(
        spectrum.device().type() == c10::DeviceType::PrivateUse1,
        "musa_irfft_graph expects a MUSA tensor");
    TORCH_CHECK(
        spectrum.scalar_type() == torch::kComplexFloat,
        "musa_irfft_graph currently supports complex64 input only");
    TORCH_CHECK(spectrum.dim() == 3, "musa_irfft_graph expects [batch, frequency, frames]");

    spectrum = spectrum.contiguous();
    const int batch = static_cast<int>(spectrum.size(0));
    const int frequencies = static_cast<int>(spectrum.size(1));
    const int frames = static_cast<int>(spectrum.size(2));
    const int fft_size = static_cast<int>(n);
    TORCH_CHECK(
        frequencies == fft_size / 2 + 1,
        "frequency dimension must equal n / 2 + 1, got ",
        frequencies,
        " for n=",
        fft_size);

    auto transposed = spectrum.permute({0, 2, 1}).contiguous();
    auto full_spectrum = torch::zeros(
        {batch, frames, fft_size},
        spectrum.options());
    full_spectrum.slice(2, 0, frequencies).copy_(transposed);
    if (frequencies > 2) {
        auto mirrored = transposed.slice(2, 1, frequencies - 1).flip(2).conj();
        full_spectrum.slice(2, fft_size - (frequencies - 2), fft_size).copy_(mirrored);
    }

    auto inverse = torch::empty({batch, frames, fft_size}, spectrum.options());
    const auto stream = c10::musa::getCurrentMUSAStream(spectrum.device().index());
    mufftHandle plan = get_plan(fft_size, spectrum.device().index(), stream);
    mufftResult result = MUFFT_SUCCESS;

    for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
        for (int frame_idx = 0; frame_idx < frames; ++frame_idx) {
            auto* input = reinterpret_cast<mufftComplex*>(
                full_spectrum.data_ptr<c10::complex<float>>() +
                (batch_idx * frames + frame_idx) * fft_size);
            auto* output = reinterpret_cast<mufftComplex*>(
                inverse.data_ptr<c10::complex<float>>() +
                (batch_idx * frames + frame_idx) * fft_size);
            result = mufftExecC2C(plan, input, output, MUFFT_INVERSE);
            TORCH_CHECK(
                result == MUFFT_SUCCESS,
                "muFFT inverse execution failed, result=",
                static_cast<int>(result));
        }
    }

    return at::real(inverse.permute({0, 2, 1})) / fft_size;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "musa_irfft_graph",
        &musa_irfft_graph,
        "Graph-capturable inverse real FFT implemented with muFFT");
}
