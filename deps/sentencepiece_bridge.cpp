#include <sentencepiece_processor.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

int fail(const std::string &message) {
    last_error = message;
    return -1;
}
}

extern "C" {

void *reels_sp_create() {
    try {
        return new sentencepiece::SentencePieceProcessor();
    } catch (const std::exception &error) {
        last_error = error.what();
        return nullptr;
    }
}

void reels_sp_destroy(void *processor) {
    delete static_cast<sentencepiece::SentencePieceProcessor *>(processor);
}

int reels_sp_load(void *processor, const char *path) {
    if (!processor || !path) return fail("null SentencePiece processor or model path");
    const auto status =
        static_cast<sentencepiece::SentencePieceProcessor *>(processor)->Load(path);
    if (!status.ok()) return fail(status.ToString());
    last_error.clear();
    return 0;
}

int reels_sp_encode(void *processor, const char *text, int32_t *ids,
                    std::size_t capacity, std::size_t *length) {
    if (!processor || !text || !length)
        return fail("null argument passed to SentencePiece encode");
    std::vector<int> encoded;
    const auto status =
        static_cast<sentencepiece::SentencePieceProcessor *>(processor)
            ->Encode(std::string(text), &encoded);
    if (!status.ok()) return fail(status.ToString());
    *length = encoded.size();
    if (!ids) return 0;
    if (capacity < encoded.size()) return fail("SentencePiece output buffer is too small");
    for (std::size_t i = 0; i < encoded.size(); ++i)
        ids[i] = static_cast<int32_t>(encoded[i]);
    last_error.clear();
    return 0;
}

int reels_sp_piece_size(void *processor) {
    if (!processor) return fail("null SentencePiece processor");
    return static_cast<sentencepiece::SentencePieceProcessor *>(processor)
        ->GetPieceSize();
}

int reels_sp_pad_id(void *processor) {
    return processor
        ? static_cast<sentencepiece::SentencePieceProcessor *>(processor)->pad_id()
        : -1;
}

int reels_sp_eos_id(void *processor) {
    return processor
        ? static_cast<sentencepiece::SentencePieceProcessor *>(processor)->eos_id()
        : -1;
}

int reels_sp_unk_id(void *processor) {
    return processor
        ? static_cast<sentencepiece::SentencePieceProcessor *>(processor)->unk_id()
        : -1;
}

const char *reels_sp_last_error() {
    return last_error.c_str();
}

}
