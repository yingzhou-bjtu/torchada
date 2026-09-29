#include <torch/extension.h>

#include <mufft.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct PlanKey {
    int n;
    int transform_count;
    int device;
    std::uint64_t stream;

    bool operator==(const PlanKey& other) const {
        return n == other.n && transform_count == other.transform_count &&
            device == other.device && stream == other.stream;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const {
        return (static_cast<std::size_t>(key.n) << 32) ^
            static_cast<std::size_t>(key.transform_count) ^
            static_cast<std::size_t>(key.device) ^
            static_cast<std::size_t>(key.stream);
    }
};

struct PlanEntry {
    mufftHandle handle;
    c10::musa::MUSAStream stream;
};

std::mutex plan_mutex;
std::unordered_map<PlanKey, PlanEntry, PlanKeyHash> plans;

// A captured graph may replay a plan long after capture.  Plans therefore stay
// alive until explicit process cleanup instead of being evicted while graphs
// can still reference them.
mufftHandle get_plan(
    int n,
    int transform_count,
    int device,
    c10::musa::MUSAStream stream) {
    const PlanKey key{
        n,
        transform_count,
        device,
        static_cast<std::uint64_t>(stream.id())};
    {
        std::lock_guard<std::mutex> lock(plan_mutex);
        auto it = plans.find(key);
        if (it != plans.end()) {
            return it->second.handle;
        }
    }

    mufftHandle plan = nullptr;
    const mufftResult result = mufftPlanMany(
        &plan,
        1,
        &n,
        nullptr,
        1,
        n,
        nullptr,
        1,
        n,
        MUFFT_C2C,
        transform_count);
    TORCH_CHECK(
        result == MUFFT_SUCCESS,
        "muFFT C2C plan creation failed for n=",
        n,
        ", result=",
        static_cast<int>(result));
    const mufftResult stream_result = mufftSetStream(plan, stream);
    if (stream_result != MUFFT_SUCCESS) {
        mufftDestroy(plan);
        TORCH_CHECK(
            false,
            "muFFT stream binding failed, result=",
            static_cast<int>(stream_result));
    }

    mufftHandle cached_plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(plan_mutex);
        auto it = plans.find(key);
        if (it != plans.end()) {
            cached_plan = it->second.handle;
        } else {
            plans.emplace(key, PlanEntry{plan, stream});
        }
    }
    if (cached_plan != nullptr) {
        TORCH_CHECK(
            mufftDestroy(plan) == MUFFT_SUCCESS,
            "muFFT duplicate plan destruction failed");
        return cached_plan;
    }
    return plan;
}

void clear_plans() {
    std::vector<PlanEntry> entries;
    {
        std::lock_guard<std::mutex> lock(plan_mutex);
        entries.reserve(plans.size());
        for (const auto& entry : plans) {
            entries.push_back(entry.second);
        }
        plans.clear();
    }
    for (const auto& entry : entries) {
        entry.stream.synchronize();
        mufftDestroy(entry.handle);
    }
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
    TORCH_CHECK(fft_size > 0, "musa_irfft_graph expects a positive n");
    const int64_t transform_count_64 = static_cast<int64_t>(batch) * frames;
    TORCH_CHECK(
        transform_count_64 <= std::numeric_limits<int>::max(),
        "musa_irfft_graph has too many transforms");
    const int transform_count = static_cast<int>(transform_count_64);
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
    const auto plan = get_plan(
        fft_size,
        transform_count,
        spectrum.device().index(),
        stream);
    auto* input = reinterpret_cast<mufftComplex*>(full_spectrum.data_ptr<c10::complex<float>>());
    auto* output = reinterpret_cast<mufftComplex*>(inverse.data_ptr<c10::complex<float>>());
    const mufftResult result =
        mufftExecC2C(plan, input, output, MUFFT_INVERSE);
    TORCH_CHECK(
        result == MUFFT_SUCCESS,
        "muFFT inverse execution failed, result=",
        static_cast<int>(result));

    return at::real(inverse.permute({0, 2, 1})) / fft_size;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "musa_irfft_graph",
        &musa_irfft_graph,
        "Graph-capturable inverse real FFT implemented with muFFT");
    module.def("clear_mufft_plans", &clear_plans);
}
