/*
 * Supplementary diagnostic (NOT part of the controlled comparison).
 * Same i-j-k loop and contiguous row-major layout as src/c/matmul.c, but only
 * the first `rows` rows of C are computed, so the cost per multiply-add of the
 * column-wise access to B can be measured at sizes too large for a full run.
 *
 * Usage: probe_c <n> <rows> <warmups> <reps>
 * Output: one line per measured run: c_flat,<n>,<rows>,<rep>,<ns>,<checksum>
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static long long now_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (long long)((double)t.QuadPart * 1e9 / (double)freq.QuadPart);
}

static uint64_t state = 2026;
static double next_value(void) {           /* xorshift64*, uniform in [-1, 1) */
    state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
    return (double)((state * 2685821657736338717ULL) >> 11) * 0x1.0p-52 - 1.0;
}

__attribute__((noinline))
static void multiply_rows(const double *a, const double *b, double *c, int n, int rows) {
    for (int i = 0; i < rows; i++) {
        const double *ai = a + (size_t)i * n;
        double *ci = c + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += ai[k] * b[(size_t)k * n + j];
            }
            ci[j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: probe_c <n> <rows> <warmups> <reps>\n"); return 1; }
    int n = atoi(argv[1]), rows = atoi(argv[2]), warmups = atoi(argv[3]), reps = atoi(argv[4]);
    double *a = malloc(sizeof(double) * (size_t)rows * n);
    double *b = malloc(sizeof(double) * (size_t)n * n);
    double *c = calloc((size_t)rows * n, sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "out of memory\n"); return 3; }
    for (size_t i = 0; i < (size_t)rows * n; i++) a[i] = next_value();
    for (size_t i = 0; i < (size_t)n * n; i++) b[i] = next_value();
    for (int r = 0; r < warmups + reps; r++) {
        long long t0 = now_ns();
        multiply_rows(a, b, c, n, rows);
        long long t1 = now_ns();
        double s = 0.0;
        for (size_t i = 0; i < (size_t)rows * n; i++) s += c[i];
        if (r >= warmups) printf("c_flat,%d,%d,%d,%lld,%.17g\n", n, rows, r - warmups, t1 - t0, s);
    }
    free(a); free(b); free(c);
    return 0;
}
