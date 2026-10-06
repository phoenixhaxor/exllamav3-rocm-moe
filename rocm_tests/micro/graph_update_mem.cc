// Does hipGraphExecKernelNodeSetParams on an instantiated graph allocate device memory that is only released with
// the exec? Capture one kernel; launch it N times without updates, then N times with an argument update before each
// launch, then destroy the exec. Prints this process's DRM memory (drm-memory-vram + gtt from /proc/self/fdinfo).
//   hipcc -O2 --offload-arch=<arch> -x hip graph_update_mem.cc -o gum && ./gum [N] [R]   (R: re-instantiate every R updates)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

static long drm_kib()
{
    long total = 0;
    DIR* d = opendir("/proc/self/fdinfo");
    struct dirent* e;
    while ((e = readdir(d)))
    {
        if (e->d_name[0] == '.') continue;
        std::string p = std::string("/proc/self/fdinfo/") + e->d_name;
        FILE* f = fopen(p.c_str(), "r");
        if (!f) continue;
        char line[256];
        bool amd = false;
        long v = 0, x;
        while (fgets(line, sizeof line, f))
        {
            if (strstr(line, "drm-driver:") && strstr(line, "amdgpu")) amd = true;
            if (sscanf(line, "drm-memory-vram: %ld", &x) == 1) v += x;
            if (sscanf(line, "drm-memory-gtt: %ld", &x) == 1) v += x;
        }
        fclose(f);
        if (amd) total += v;
    }
    closedir(d);
    return total;
}

// 16 pointer arguments, roughly the size of a Triton attention kernel's argument block
__global__ void k(int* p, int v, int* a1, int* a2, int* a3, int* a4, int* a5, int* a6, int* a7, int* a8,
                  int* a9, int* a10, int* a11, int* a12, int* a13, int* a14)
{
    if (threadIdx.x == 0) p[0] = v;
}

int main(int argc, char** argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    int* d;
    CK(hipMalloc(&d, 64));
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipGraph_t g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    k<<<1, 64, 0, s>>>(d, 0, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
    CK(hipStreamEndCapture(s, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    size_t n = 1;
    hipGraphNode_t node;
    CK(hipGraphGetNodes(g, &node, &n));
    hipKernelNodeParams kp;
    CK(hipGraphKernelNodeGetParams(node, &kp));
    int v = 0;
    void* args[] = { &d, &v, &d, &d, &d, &d, &d, &d, &d, &d, &d, &d, &d, &d, &d, &d };
    kp.kernelParams = args;

    for (int i = 0; i < iters; ++i) CK(hipGraphLaunch(ge, s));
    CK(hipStreamSynchronize(s));
    long m0 = drm_kib();
    for (int i = 0; i < iters; ++i) CK(hipGraphLaunch(ge, s));
    CK(hipStreamSynchronize(s));
    printf("%d launches without updates: DRM memory %+ld KiB\n", iters, drm_kib() - m0);

    // argv[2] = R > 0: destroy and re-instantiate the exec (after a stream sync) every R updates, then re-apply the
    // current arguments to the fresh exec (it starts from the captured ones)
    int reinst = argc > 2 ? atoi(argv[2]) : 0;
    m0 = drm_kib();
    for (int i = 1; i <= iters; ++i)
    {
        v = i;
        if (reinst && i % reinst == 0)
        {
            CK(hipStreamSynchronize(s));
            CK(hipGraphExecDestroy(ge));
            CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        }
        CK(hipGraphExecKernelNodeSetParams(ge, node, &kp));
        CK(hipGraphLaunch(ge, s));
        if (i % (iters / 4) == 0)
        {
            CK(hipStreamSynchronize(s));
            long dm = drm_kib() - m0;
            printf("after %6d updates: DRM memory %+ld KiB (%.0f bytes per update)\n", i, dm, dm * 1024.0 / i);
            if (dm > 96 * 1024) { printf("stopping, growth over 96 MiB\n"); break; }
        }
    }
    int h;
    CK(hipMemcpy(&h, d, 4, hipMemcpyDeviceToHost));
    printf("last value %d (expected %d)\n", h, iters);
    CK(hipGraphExecDestroy(ge));
    CK(hipDeviceSynchronize());
    printf("after hipGraphExecDestroy: DRM memory %+ld KiB\n", drm_kib() - m0);
    return 0;
}
