// bulles.cu — étape 0 de l'étude « mégakernel » (voir MEGAKERNEL.md) : micro-banc autonome, sans llama.cpp.
//
// Question : sur une P40 (sm_61), que coûte l'enchaînement de noyaux séparés par rapport à UN noyau persistant
// qui enchaîne les mêmes étapes ? Chaque étape est un produit matrice-vecteur sur des poids 8 bits (__dp4a),
// de la taille de nos matrices. Deux modes, mêmes calculs, mêmes résultats :
//   A : une étape = un lancement de noyau (comme llama.cpp aujourd'hui) ;
//   B : un seul lancement ; les warps restent en place, se partagent les lignes de chaque étape par un compteur
//       atomique en mémoire globale et franchissent une barrière (compteur atomique) entre deux étapes.
// Sortie : tableau markdown (µs par étape et Go/s dans chaque mode).
//
// Compilation : nvcc -O3 -arch=sm_61 -o bulles bulles.cu
// Usage       : ./bulles [blocs_par_SM=16] [lignes_par_prise=8] [repetitions=15]

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call)                                                                              \
    do {                                                                                         \
        cudaError_t err_ = (call);                                                               \
        if (err_ != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA error %s at line %d: %s\n", #call, __LINE__, cudaGetErrorString(err_)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

struct Step {
    long long w_off;    // offset of the weights, in ints
    int       rows;
    int       row_ints; // 4 weights per int
    int       y_off;
    int       ncols;    // vecteurs d'entrée par ligne : 1 = génération, 3 = vérification MTP (les poids sont lus une fois)
};

// x contient 3 vecteurs de x_stride ints à la suite ; la somme des ncols produits est renvoyée
static __device__ __forceinline__ int row_dot(const int * w, const int * x, const int row_ints, const int lane,
                                              const int ncols, const int x_stride) {
    int acc = 0;
    if (ncols == 1) {
        for (int i = lane; i < row_ints; i += 32) {
            acc = __dp4a(w[i], x[i], acc);
        }
    } else {
        int acc1 = 0;
        int acc2 = 0;
        for (int i = lane; i < row_ints; i += 32) {
            const int wi = w[i];
            acc  = __dp4a(wi, x[i], acc);
            acc1 = __dp4a(wi, x[i + x_stride], acc1);
            acc2 = __dp4a(wi, x[i + 2 * x_stride], acc2);
        }
        acc += acc1 + acc2;
    }
    return acc;
}

static __device__ __forceinline__ int warp_sum(int v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// mode A : un warp par bloc, deux lignes par bloc
__global__ void step_kernel(const int * w, const int * x, float * y, const int rows, const int row_ints,
                            const int ncols, const int x_stride) {
    const int lane = threadIdx.x;
    const int r0   = 2 * blockIdx.x;
    for (int k = 0; k < 2; ++k) {
        const int r = r0 + k;
        if (r >= rows) {
            break;
        }
        const int acc = warp_sum(row_dot(w + (long long) r * row_ints, x, row_ints, lane, ncols, x_stride));
        if (lane == 0) {
            y[r] = (float) acc;
        }
    }
}

// mode B : noyau persistant, blocs de 2 warps, chaque warp travaille seul
__global__ void persistent_kernel(const int * w_all, const int * x, float * y_all, const Step * steps, const int n_steps,
                                  int * next, int * done, const int n_warps, const int chunk, int * error, const int x_stride) {
    const int lane = threadIdx.x & 31;
    for (int s = 0; s < n_steps; ++s) {
        const Step    st = steps[s];
        const int *   w  = w_all + st.w_off;
        float *       y  = y_all + st.y_off;

        while (true) {
            int r0 = 0;
            if (lane == 0) {
                r0 = atomicAdd(&next[s], chunk);
            }
            r0 = __shfl_sync(0xffffffff, r0, 0);
            if (r0 >= st.rows) {
                break;
            }
            const int r1 = r0 + chunk < st.rows ? r0 + chunk : st.rows;
            for (int r = r0; r < r1; ++r) {
                const int acc = warp_sum(row_dot(w + (long long) r * st.row_ints, x, st.row_ints, lane, st.ncols, x_stride));
                if (lane == 0) {
                    y[r] = (float) acc;
                }
            }
        }

        // barrière : l'étape suivante ne commence que lorsque tous les warps ont fini celle-ci
        if (lane == 0) {
            atomicAdd(&done[s], 1);
        }
        int spins = 0;
        while (true) {
            int d = 0;
            if (lane == 0) {
                d = *((volatile int *) &done[s]);
            }
            d = __shfl_sync(0xffffffff, d, 0);
            if (d >= n_warps) {
                break;
            }
            if (++spins > 20000000) { // tous les blocs ne sont pas résidents : on abandonne au lieu de bloquer
                if (lane == 0) {
                    *error = 1;
                }
                return;
            }
        }
    }
}

// mode C : noyau persistant à gros blocs (1024 threads = 32 warps). Partage statique des lignes (aucun atomique pour
// prendre du travail) ; barrière = un seul atomique par bloc et par étape, les warps d'un bloc se synchronisent
// entre eux par __syncthreads.
__global__ void persistent_block_kernel(const int * w_all, const int * x, float * y_all, const Step * steps, const int n_steps,
                                        int * done, const int n_blocks, int * error, const int x_stride) {
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int n_warps = n_blocks * (blockDim.x >> 5);
    const int gw      = blockIdx.x * (blockDim.x >> 5) + warp;
    __shared__ int go;

    for (int s = 0; s < n_steps; ++s) {
        const Step    st = steps[s];
        const int *   w  = w_all + st.w_off;
        float *       y  = y_all + st.y_off;

        // lignes contiguës pour ce warp
        const int r0 = (int) ((long long) st.rows * gw / n_warps);
        const int r1 = (int) ((long long) st.rows * (gw + 1) / n_warps);
        for (int r = r0; r < r1; ++r) {
            const int acc = warp_sum(row_dot(w + (long long) r * st.row_ints, x, st.row_ints, lane, st.ncols, x_stride));
            if (lane == 0) {
                y[r] = (float) acc;
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(&done[s], 1);
        }
        int spins = 0;
        while (true) {
            if (threadIdx.x == 0) {
                go = *((volatile int *) &done[s]) >= n_blocks ? 1 : (++spins > 20000000 ? 2 : 0);
            }
            __syncthreads();
            const int g = go;
            __syncthreads();
            if (g == 1) {
                break;
            }
            if (g == 2) {
                if (threadIdx.x == 0) {
                    *error = 1;
                }
                return;
            }
        }
    }
}

struct HostStep {
    int rows;
    int row_bytes;
    int ncols = 1;
};

struct Result {
    double us_a = 0.0;
    double us_b = 0.0;
    double us_c = 0.0;
    bool   same = false;
    bool   ok_b = false;
};

static int g_blocks_c = 30; // mode C : nombre de blocs de 1024 threads

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static Result run(const std::vector<HostStep> & hsteps, const int n_blocks, const int chunk, const int reps) {
    const int n_steps = (int) hsteps.size();
    std::vector<Step> steps(n_steps);
    long long total_ints = 0;
    int       total_rows = 0;
    int       max_row_ints = 0;
    for (int s = 0; s < n_steps; ++s) {
        steps[s].w_off    = total_ints;
        steps[s].rows     = hsteps[s].rows;
        steps[s].row_ints = hsteps[s].row_bytes / 4;
        steps[s].y_off    = total_rows;
        steps[s].ncols    = hsteps[s].ncols;
        total_ints += (long long) steps[s].rows * steps[s].row_ints;
        total_rows += steps[s].rows;
        max_row_ints = std::max(max_row_ints, steps[s].row_ints);
    }

    int *   w = nullptr;
    int *   x = nullptr;
    float * y_a = nullptr;
    float * y_b = nullptr;
    Step *  d_steps = nullptr;
    int *   next = nullptr;
    int *   done = nullptr;
    int *   error = nullptr;
    CHECK(cudaMalloc(&w, total_ints * sizeof(int)));
    CHECK(cudaMalloc(&x, 3 * (size_t) max_row_ints * sizeof(int)));
    CHECK(cudaMalloc(&y_a, total_rows * sizeof(float)));
    CHECK(cudaMalloc(&y_b, total_rows * sizeof(float)));
    CHECK(cudaMalloc(&d_steps, n_steps * sizeof(Step)));
    CHECK(cudaMalloc(&next, n_steps * sizeof(int)));
    CHECK(cudaMalloc(&done, n_steps * sizeof(int)));
    CHECK(cudaMalloc(&error, sizeof(int)));
    CHECK(cudaMemset(w, 3, total_ints * sizeof(int)));
    CHECK(cudaMemset(x, 1, 3 * (size_t) max_row_ints * sizeof(int)));
    CHECK(cudaMemset(y_a, 0, total_rows * sizeof(float)));
    CHECK(cudaMemset(y_b, 0, total_rows * sizeof(float)));
    CHECK(cudaMemcpy(d_steps, steps.data(), n_steps * sizeof(Step), cudaMemcpyHostToDevice));

    const int n_warps = 2 * n_blocks;
    std::vector<double> t_a, t_b, t_c;
    float * y_c = nullptr;
    CHECK(cudaMalloc(&y_c, total_rows * sizeof(float)));
    CHECK(cudaMemset(y_c, 0, total_rows * sizeof(float)));
    Result res;
    res.ok_b = true;

    for (int rep = 0; rep < reps + 2; ++rep) {
        // A : un lancement par étape
        CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < n_steps; ++s) {
            step_kernel<<<(steps[s].rows + 1) / 2, 32>>>(w + steps[s].w_off, x, y_a + steps[s].y_off, steps[s].rows, steps[s].row_ints,
                                                          steps[s].ncols, max_row_ints);
        }
        CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();

        // B : un seul lancement
        CHECK(cudaMemset(next, 0, n_steps * sizeof(int)));
        CHECK(cudaMemset(done, 0, n_steps * sizeof(int)));
        CHECK(cudaMemset(error, 0, sizeof(int)));
        CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        persistent_kernel<<<n_blocks, 64>>>(w, x, y_b, d_steps, n_steps, next, done, n_warps, chunk, error, max_row_ints);
        CHECK(cudaDeviceSynchronize());
        auto t3 = std::chrono::steady_clock::now();

        int err = 0;
        CHECK(cudaMemcpy(&err, error, sizeof(int), cudaMemcpyDeviceToHost));
        if (err != 0) {
            res.ok_b = false;
        }
        // C : un seul lancement, gros blocs
        CHECK(cudaMemset(done, 0, n_steps * sizeof(int)));
        CHECK(cudaMemset(error, 0, sizeof(int)));
        CHECK(cudaDeviceSynchronize());
        auto t4 = std::chrono::steady_clock::now();
        persistent_block_kernel<<<g_blocks_c, 1024>>>(w, x, y_c, d_steps, n_steps, done, g_blocks_c, error, max_row_ints);
        CHECK(cudaDeviceSynchronize());
        auto t5 = std::chrono::steady_clock::now();
        CHECK(cudaMemcpy(&err, error, sizeof(int), cudaMemcpyDeviceToHost));
        if (err != 0) {
            res.ok_b = false;
        }

        if (rep >= 2) { // les deux premiers passages chauffent
            t_a.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            t_b.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
            t_c.push_back(std::chrono::duration<double, std::micro>(t5 - t4).count());
        }
    }

    std::vector<float> h_a(total_rows), h_b(total_rows);
    CHECK(cudaMemcpy(h_a.data(), y_a, total_rows * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b.data(), y_b, total_rows * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> h_c(total_rows);
    CHECK(cudaMemcpy(h_c.data(), y_c, total_rows * sizeof(float), cudaMemcpyDeviceToHost));
    res.same = memcmp(h_a.data(), h_b.data(), total_rows * sizeof(float)) == 0 &&
               memcmp(h_a.data(), h_c.data(), total_rows * sizeof(float)) == 0;
    res.us_a = median(t_a);
    res.us_b = median(t_b);
    res.us_c = median(t_c);
    cudaFree(y_c);

    cudaFree(w); cudaFree(x); cudaFree(y_a); cudaFree(y_b); cudaFree(d_steps); cudaFree(next); cudaFree(done); cudaFree(error);
    return res;
}

static double total_bytes(const std::vector<HostStep> & steps) {
    double b = 0.0;
    for (const HostStep & s : steps) {
        b += (double) s.rows * s.row_bytes;
    }
    return b;
}

static void report(const char * name, const std::vector<HostStep> & steps, const int n_blocks, const int chunk, const int reps) {
    const Result r     = run(steps, n_blocks, chunk, reps);
    const double bytes = total_bytes(steps);
    const double n     = (double) steps.size();
    printf("| %s | %d | %.1f | %.1f | %.0f | %.1f | %.0f | %.2f | %.1f | %.0f | %.2f | %s |\n",
            name, (int) steps.size(), bytes / 1e6,
            r.us_a / n, bytes / r.us_a / 1e3,
            r.us_b / n, bytes / r.us_b / 1e3,
            r.us_a / r.us_b,
            r.us_c / n, bytes / r.us_c / 1e3,
            r.us_a / r.us_c,
            !r.ok_b ? "B a échoué (barrière)" : (r.same ? "identiques" : "DIFFÉRENTS"));
    fflush(stdout);
}

int main(int argc, char ** argv) {
    const int blocks_per_sm = argc > 1 ? atoi(argv[1]) : 16;
    const int chunk         = argc > 2 ? atoi(argv[2]) : 8;
    const int reps          = argc > 3 ? atoi(argv[3]) : 15;
    const int blocks_c_sm   = argc > 4 ? atoi(argv[4]) : 1;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
    const int n_blocks = prop.multiProcessorCount * blocks_per_sm;
    g_blocks_c = prop.multiProcessorCount * blocks_c_sm;

    printf("- GPU : %s, sm_%d%d, %d SM, lancement coopératif : %s\n", prop.name, prop.major, prop.minor,
            prop.multiProcessorCount, coop ? "oui" : "non");
    printf("- mode B : %d blocs de 64 threads (%d par SM), %d lignes par prise ; médiane de %d passages\n\n",
            n_blocks, blocks_per_sm, chunk, reps);
    printf("| scénario | étapes | Mo lus | A : µs/étape | A : Go/s | B : µs/étape | B : Go/s | A/B | C : µs/étape | C : Go/s | A/C | résultats |\n");
    printf("- mode C : %d blocs de 1024 threads (%d par SM), partage statique, un atomique par bloc et par étape\n\n", g_blocks_c, blocks_c_sm);
    printf("|---|---|---|---|---|---|---|---|---|---|---|---\n");

    // 1. étapes identiques, à la taille de nos matrices (lignes x octets par ligne)
    struct Uniform { const char * name; int rows; int row_bytes; int count; };
    const Uniform uniforms[] = {
        { "qkv Q4_0 8192x2048 (9,4 Mo) x40",          8192,   1152, 40 },
        { "z Q4_0 4096x2048 (4,7 Mo) x40",            4096,   1152, 40 },
        { "descente experts 16384x512 (5,2 Mo) x40",  16384,  320,  40 },
        { "routeur F32 256x2048 (2,1 Mo) x40",        256,    8192, 40 },
        { "petit 512x2048 Q4_0 (0,6 Mo) x40",         512,    1152, 40 },
        { "portes 64x2048 F32 (0,5 Mo) x40",          64,     8192, 40 },
        { "minuscule 2 lignes (8 Ko) x200",           2,      4096, 200 },
        { "tête Q6_K 248320x2048 (417 Mo) x2",        248320, 1680, 2 },
    };
    for (const Uniform & u : uniforms) {
        std::vector<HostStep> steps(u.count, HostStep{ u.rows, u.row_bytes, 1 });
        report(u.name, steps, n_blocks, chunk, reps);
    }
    {
        // les mêmes poids lus une fois pour 3 vecteurs d'entrée (vérification MTP)
        std::vector<HostStep> steps(40, HostStep{ 8192, 1152, 3 });
        report("qkv Q4_0 8192x2048 x40, 3 colonnes", steps, n_blocks, chunk, reps);
    }

    // 2. « token synthétique » : la suite d'étapes d'un token (40 couches + tête), petits noyaux compris
    const HostStep tiny = { 2, 4096, 1 };
    std::vector<HostStep> layer;
    for (int i = 0; i < 3; ++i) layer.push_back(tiny);           // norme, lecture et concaténation de l'état de convolution
    layer.push_back({ 8192, 1152, 1 });                           // qkv
    for (int i = 0; i < 3; ++i) layer.push_back(tiny);           // copie d'état, convolution, norme q/k
    layer.push_back({ 64, 8192 });                                // portes alpha/beta
    layer.push_back({ 4096, 512 });                               // delta-net : 2 Mo d'état
    layer.push_back(tiny);                                        // norme
    layer.push_back({ 4096, 1152 });                              // z
    layer.push_back(tiny);                                        // silu x mul
    layer.push_back({ 2048, 4352 });                              // sortie Q8_0 4096 -> 2048
    layer.push_back(tiny);                                        // norme
    layer.push_back({ 256, 8192 });                               // routeur
    layer.push_back(tiny);                                        // softmax + top-k
    layer.push_back({ 8192, 1152 });                              // 8 experts : montée + porte
    layer.push_back({ 1024, 2176 });                              // expert partagé : montée + porte (Q8_0)
    layer.push_back({ 16384, 320 });                              // 8 experts : descente
    layer.push_back(tiny);                                        // somme pondérée
    layer.push_back({ 2048, 544 });                               // expert partagé : descente
    for (int i = 0; i < 3; ++i) layer.push_back(tiny);           // porte de l'expert partagé, additions
    std::vector<HostStep> token;
    for (int l = 0; l < 40; ++l) {
        token.insert(token.end(), layer.begin(), layer.end());
    }
    token.push_back(tiny);
    token.push_back({ 248320, 1680 });                            // tête de sortie
    report("token synthétique (40 couches + tête)", token, n_blocks, chunk, reps);

    std::vector<HostStep> token_no_head(token.begin(), token.end() - 1);
    report("token synthétique sans la tête", token_no_head, n_blocks, chunk, reps);

    // 3. pas de vérification MTP à 3 colonnes : tout est lu une fois pour 3 vecteurs, sauf les experts routés,
    //    différents pour chaque token (étapes « montée + porte » et « descente » répétées 3 fois)
    std::vector<HostStep> layer3;
    for (const HostStep & st : layer) {
        const bool experts = (st.rows == 8192 && st.row_bytes == 1152 && &st != &layer[3]) || (st.rows == 16384 && st.row_bytes == 320);
        if (experts) {
            for (int j = 0; j < 3; ++j) {
                layer3.push_back(st);
            }
        } else {
            HostStep st3 = st;
            st3.ncols = st.rows > 2 ? 3 : 1;
            layer3.push_back(st3);
        }
    }
    std::vector<HostStep> verif;
    for (int l = 0; l < 40; ++l) {
        verif.insert(verif.end(), layer3.begin(), layer3.end());
    }
    verif.push_back(tiny);
    verif.push_back({ 248320, 1680, 3 });
    report("vérification MTP synthétique, 3 colonnes", verif, n_blocks, chunk, reps);

    printf("\nLecture : A/B > 1 = le noyau persistant est plus rapide. « Go/s » = octets de poids lus / temps total.\n");
    return 0;
}
