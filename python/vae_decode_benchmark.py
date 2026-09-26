"""Time the MiniMax H3 video-VAE decode the HR Endless Sampler performs once per chunk.

The sampler calls `_decode_video_frames(vae, latent)` -> `vae.decode(latent)` for every
chunk's final pixels, so this benchmark times exactly that call. It measures wall time,
per-frame cost, and - on CPU - how many cores the decode actually keeps busy, which is
reported rather than assumed (a decode that silently runs single threaded is the usual
reason CPU offload looks worse than it should).

Thread pools are pinned before torch is imported. Import order matters: OpenMP/MKL read
their environment once, at library init.

Usage (from the ComfyUI root, so `comfy` is importable):
    ./tools/python.sh -s custom_nodes/ComfyUI-MiniMax-H3-Sampler-Unlimited/python/vae_decode_benchmark.py --device cpu
    ./tools/python.sh -s custom_nodes/ComfyUI-MiniMax-H3-Sampler-Unlimited/python/vae_decode_benchmark.py --device gpu
"""

import argparse
import os
import sys
import threading
import time

# Every pool the decode could hand work to, pinned before any torch/comfy import below.
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
THREADS = int(os.environ.get("VAE_BENCH_THREADS", os.cpu_count() or 1))
for name in THREAD_ENV:
    os.environ[name] = str(THREADS)

import torch  # noqa: E402  (must follow the thread pinning above)

DEFAULT_COMFY_ROOT = os.environ.get("COMFY_ROOT", "/NVME/comfyui/ComfyUI")
DEFAULT_VAE = os.environ.get("H3_VIDEO_VAE", "/NVME/comfyui/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors")
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


class ProcessCpuSampler:
    """Sample this process's consumed CPU time so parallel use can be proven, not assumed."""

    def __init__(self):
        self.started_wall = 0.0
        self.started_cpu = 0.0
        self.stopped_wall = 0.0
        self.stopped_cpu = 0.0

    @staticmethod
    def cpu_seconds():
        """Cumulative user+system CPU seconds from /proc/self/stat (fields 14 and 15)."""
        with open("/proc/self/stat", "r") as handle:
            text = handle.read()
        # The comm field is parenthesised and may contain spaces, so split after its last ')'.
        fields = text[text.rindex(")") + 2:].split()
        return (int(fields[11]) + int(fields[12])) / float(CLOCK_TICKS)

    def start(self):
        """Record the wall/CPU baseline immediately before the timed region."""
        self.started_cpu = self.cpu_seconds()
        self.started_wall = time.perf_counter()

    def stop(self):
        """Freeze the timed region and return (wall seconds, cores kept busy on average)."""
        self.stopped_wall = time.perf_counter()
        self.stopped_cpu = self.cpu_seconds()
        wall = self.stopped_wall - self.started_wall
        busy = (self.stopped_cpu - self.started_cpu) / wall if wall > 0 else 0.0
        return wall, busy


class ResidentSampler(threading.Thread):
    """Poll RSS while the decode runs so peak host memory is a measurement."""

    def __init__(self, interval=0.5, verbose_every=10.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = interval
        self.verbose_every = verbose_every
        self.stop_event = threading.Event()
        self.peak_rss = 0

    @staticmethod
    def rss_bytes():
        """Resident set size in bytes from /proc/self/statm (pages are 4 KiB on x86-64)."""
        with open("/proc/self/statm", "r") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    def run(self):
        """Sample until asked to stop; peak is all we keep, plus a periodic growth line."""
        started = time.perf_counter()
        next_report = started + self.verbose_every
        while not self.stop_event.is_set():
            current = self.rss_bytes()
            self.peak_rss = max(self.peak_rss, current)
            now = time.perf_counter()
            # A growth curve shows whether memory climbs per tile, which is the difference
            # between an infeasible decode and one that just needs a bigger budget.
            if now >= next_report:
                print("    +%.0fs rss: %.2f GiB" % (now - started, current / 1024 ** 3))
                sys.stdout.flush()
                next_report = now + self.verbose_every
            self.stop_event.wait(self.interval)

    def finish(self):
        """Stop sampling and wait for the thread to leave its loop."""
        self.stop_event.set()
        self.join(timeout=5)


class H3VaeDecodeBenchmark:
    """Decode real H3 chunks through ComfyUI's own VAE class, one device per run."""

    def __init__(self, vae_path, comfy_root, device_name, frames, height, width, cpu_free_gib=1.0):
        self.vae_path = vae_path
        self.comfy_root = comfy_root
        self.device_name = device_name
        self.frames = frames
        self.height = height
        self.width = width
        self.cpu_free_gib = cpu_free_gib
        self.vae = None
        self.latent = None

    @staticmethod
    def configure_attention():
        """Force a CPU-capable attention backend before ComfyUI binds one at import.

        comfy/ldm/modules/attention.py picks its optimized_attention once, at import time,
        in the order sage -> flash -> xformers -> pytorch. xformers is CUDA-only, and a
        library import parses no argv (comfy/options.py args_parsing stays False), so
        cpu_state is left at its GPU default and a later CPU decode binds attention_xformers
        and dies in memory_efficient_attention_forward. Setting the flag on the shared args
        object here, before comfy.model_management reads it, moves the chain to
        attention_pytorch, which is SDPA and runs on CPU.
        """
        import comfy.cli_args
        comfy.cli_args.args.use_pytorch_cross_attention = True

    @staticmethod
    def enable_dynamic_vram():
        """Mirror main.py's DynamicVRAM bring-up so a bare harness matches server residency.

        Without this the VAE builds a plain ModelPatcher, which keeps all ~4.85 GiB of fp16
        weights resident on a 16 GiB card and OOMs the tiled decode. The server streams them
        through aimdo instead, so the measurement has to ask for the same thing.
        """
        import comfy_aimdo.control
        import comfy.memory_management
        import comfy.model_patcher
        if not comfy_aimdo.control.init_devices([0]):
            print("aimdo init_devices failed; weights will be pinned resident")
            return False
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
        return True

    @staticmethod
    def limit_cpu_free_memory(gib):
        """Report a device-sized free-memory figure to CPU callers of get_free_memory.

        comfy/ldm/minimax/vae.py:592 sizes its per-call tile batch as
        min(4, get_free_memory(device) // 128 MiB). On CPU that call resolves to
        comfy/system_memory available, i.e. ~110 GiB here, so the decode always takes the
        largest batch of 4 tiles - each tile carrying a whole 17-frame 1088x256 fp32
        activation set - and the host OOM-killer ends the run around 100 GiB RSS. Reporting
        a real device budget instead makes the CPU run use the same batch a 16 GiB card
        would. This only patches the benchmark process; nothing in ComfyUI is modified.
        """
        import comfy.model_management
        budget = int(gib * 1024 ** 3)
        real = comfy.model_management.get_free_memory

        def capped(dev=None, torch_free_too=False):
            """Free memory, but never more than `budget` when the target device is the CPU."""
            if dev is not None and getattr(dev, "type", None) == "cpu":
                return (budget, budget) if torch_free_too else budget
            return real(dev, torch_free_too)

        comfy.model_management.get_free_memory = capped
        return budget

    def import_comfy(self):
        """Put the ComfyUI tree on the path and import the two modules the harness needs."""
        if self.comfy_root not in sys.path:
            sys.path.insert(0, self.comfy_root)
        # Only the CPU run needs the override: on GPU, production keeps ComfyUI's own choice.
        if self.device_name == "cpu":
            self.configure_attention()
        else:
            self.enable_dynamic_vram()
        import comfy.sd
        import comfy.utils
        return comfy.sd, comfy.utils

    def load_vae(self):
        """Build the VAE through its own constructor so dtype/device rules stay ComfyUI's."""
        comfy_sd, comfy_utils = self.import_comfy()
        if self.device_name == "cpu" and self.cpu_free_gib > 0:
            self.limit_cpu_free_memory(self.cpu_free_gib)
        device = torch.device("cuda:0") if self.device_name == "gpu" else torch.device("cpu")
        state_dict = comfy_utils.load_torch_file(self.vae_path, safe_load=True)
        self.vae = comfy_sd.VAE(sd=state_dict, device=device)
        # The sampler hands the VAE the chunk latent; geometry comes from the VAE's own
        # downscale formula rather than a hardcoded 16x/5x so this cannot silently drift.
        latent_frames = int(self.vae.downscale_ratio[0](self.frames))
        latent_height = self.height // int(self.vae.downscale_ratio[1])
        latent_width = self.width // int(self.vae.downscale_ratio[2])
        self.latent = torch.randn((1, self.vae.latent_channels, latent_frames, latent_height, latent_width), dtype=self.vae.vae_dtype)
        self.latent = self.latent.to(device=device)
        return self.vae, self.latent

    def describe(self):
        """One-line summary of what this run is about to decode."""
        return "device=%s dtype=%s latent=%s -> pixels=%s" % (self.vae.device, self.vae.vae_dtype, tuple(self.latent.shape), tuple(self.vae.first_stage_model.decode_output_shape(self.latent.shape)))

    @staticmethod
    def checkpoint(label):
        """Print resident memory at a stage boundary so an OOM can be blamed on a stage."""
        print("  [%s] rss: %.2f GiB" % (label, ResidentSampler.rss_bytes() / 1024 ** 3))
        sys.stdout.flush()

    def decode(self):
        """Run the sampler's exact decode call and return its timing and memory figures."""
        if self.device_name == "gpu":
            torch.cuda.reset_peak_memory_stats()
        self.checkpoint("before decode")
        cpu_sampler = ProcessCpuSampler()
        rss_sampler = ResidentSampler()
        rss_sampler.start()
        cpu_sampler.start()
        # This is the same entry point nodes.py:2893 uses - VAE.decode, not first_stage_model.
        pixels = self.vae.decode(self.latent)
        wall, busy = cpu_sampler.stop()
        rss_sampler.finish()
        peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if self.device_name == "gpu" else 0.0
        return {"wall": wall, "cores_busy": busy, "peak_rss_gib": rss_sampler.peak_rss / 1024 ** 3, "peak_vram_gib": peak_vram, "pixel_shape": tuple(pixels.shape), "pixel_sum": float(pixels.float().sum())}

    @staticmethod
    def report(frames, result, threads):
        """Print one result block in the log so a later reader can compare runs."""
        print("--- frames=%d ---" % frames)
        print("  wall: %.1f s   (%.2f s/frame)" % (result["wall"], result["wall"] / max(1, frames)))
        print("  pixels: %s" % (result["pixel_shape"],))
        print("  pixel checksum: %.4f" % result["pixel_sum"])
        if result["cores_busy"] > 0:
            print("  cores busy (avg over decode): %.2f of %d configured" % (result["cores_busy"], threads))
            print("  peak host RSS: %.2f GiB" % result["peak_rss_gib"])
        if result["peak_vram_gib"] > 0:
            print("  peak VRAM allocated: %.2f GiB" % result["peak_vram_gib"])
        sys.stdout.flush()


def main():
    """Parse arguments, then decode each requested frame count in ascending order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--frames", default="5,22,39", help="Comma-separated output frame counts to decode")
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--comfy-root", default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--cpu-free-gib", type=float, default=0.125, help="Free memory the CPU run reports to ComfyUI's tile-batch sizing (batch = free // 128 MiB, capped at 4); 0 leaves host RAM as the answer")
    parser.add_argument("--rlimit-gib", type=float, default=16.0, help="Cap this process's address space so a runaway allocation raises instead of being OOM-killed; 0 removes the cap, which can lock the machine")
    args = parser.parse_args()

    if args.rlimit_gib > 0:
        import resource
        limit = int(args.rlimit_gib * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        print("address space capped at %.2f GiB" % args.rlimit_gib)

    frame_counts = sorted(int(value) for value in args.frames.split(",") if value.strip())
    torch.set_num_threads(THREADS)
    print("torch %s | threads=%d (interop %d) | cpu capability=%s | mkldnn=%s" % (torch.__version__, torch.get_num_threads(), torch.get_num_interop_threads(), torch.backends.cpu.get_cpu_capability(), torch.backends.mkldnn.is_available()))
    print("comfy root: %s" % args.comfy_root)
    print("vae: %s" % args.vae)
    if args.device == "cpu":
        print("cpu tile-batch budget: %.2f GiB" % args.cpu_free_gib)

    # Warm-up on the smallest size only: the ascending order means each run also pays
    # for the previous ones' allocator growth, which is how a real render behaves.
    for frames in frame_counts:
        benchmark = H3VaeDecodeBenchmark(args.vae, args.comfy_root, args.device, frames, args.height, args.width, args.cpu_free_gib)
        benchmark.load_vae()
        benchmark.checkpoint("after load")
        print(benchmark.describe())
        result = benchmark.decode()
        benchmark.report(frames, result, THREADS)
        del benchmark
        if args.device == "gpu":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
