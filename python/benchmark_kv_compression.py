"""Standalone lossless tensor compression benchmark; synthetic data is not H3 KV."""

import argparse
import json
import time

import blosc2
import torch
import zstandard


class KVCompressionBenchmark:
    """Measure exact round trips on bounded tensor samples without loading H3."""

    @staticmethod
    def tensors(value, name="sample"):
        """Visit tensors in a saved dictionary/list without interpreting their contents."""
        if torch.is_tensor(value):
            yield name, value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from KVCompressionBenchmark.tensors(item, name + "." + str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from KVCompressionBenchmark.tensors(item, name + "." + str(index))

    @staticmethod
    def run(tensor, name, mib):
        """Include byte reordering in timed, bit-exact compression/decompression."""
        tensor = tensor.detach().cpu().contiguous().view(-1)
        tensor = tensor[:int(mib * 1024 ** 2) // tensor.element_size()]
        raw = tensor.view(torch.uint8).numpy().tobytes()
        if not raw:
            return
        size = len(raw) / 1024 ** 2
        for level in (1, 3):
            compressor = zstandard.ZstdCompressor(level=level)
            codecs = [("zstd-%d" % level, compressor.compress, zstandard.ZstdDecompressor().decompress)]
            for filter_name in ("SHUFFLE", "BITSHUFFLE"):
                selected = getattr(blosc2.Filter, filter_name)
                codecs.append(("blosc-zstd-%d-%s" % (level, filter_name.lower()), lambda data, selected=selected: blosc2.compress(data, typesize=tensor.element_size(), clevel=level, filter=selected, codec=blosc2.Codec.ZSTD), blosc2.decompress))
            for codec, compress, decompress in codecs:
                started = time.perf_counter()
                packed = compress(raw)
                encode_seconds = time.perf_counter() - started
                started = time.perf_counter()
                restored = decompress(packed)
                decode_seconds = time.perf_counter() - started
                if restored != raw:
                    raise AssertionError("Lossless round trip failed: " + codec)
                print(json.dumps(dict(sample=name, dtype=str(tensor.dtype), raw_mib=size, codec=codec, saved_percent=100 * (1 - len(packed) / len(raw)), encode_mib_s=size / encode_seconds, decode_mib_s=size / decode_seconds, exact=True)), flush=True)


def main():
    """Benchmark a trusted tensor-only KV dump, or an explicitly synthetic baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Tensor-only .pt dump of actual KV; never a model checkpoint")
    parser.add_argument("--synthetic", action="store_true", help="Run seeded Gaussian BF16 baseline, not actual H3 KV")
    parser.add_argument("--mib", type=int, default=32)
    args = parser.parse_args()
    if bool(args.input) == args.synthetic or not 1 <= args.mib <= 512:
        parser.error("Choose exactly one of --input/--synthetic, and --mib between 1 and 512")
    torch.set_num_threads(2)
    blosc2.set_nthreads(2)
    if args.synthetic:
        generator = torch.Generator().manual_seed(163)
        samples = [("SYNTHETIC Gaussian BF16; not H3 KV", torch.randn(args.mib * 1024 ** 2 // 2, generator=generator).bfloat16())]
    else:
        samples = KVCompressionBenchmark.tensors(torch.load(args.input, map_location="cpu", weights_only=True, mmap=True))
    for name, tensor in samples:
        KVCompressionBenchmark.run(tensor, name, args.mib)


if __name__ == "__main__":
    main()
