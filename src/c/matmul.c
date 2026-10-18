/*
 * Basic dense matrix multiplication, C = A * B, in C.
 *
 * Storage: one contiguous row-major block of n*n doubles per matrix.
 * Algorithm: textbook i-j-k triple loop, scalar accumulator, O(n^3).
 *
 * Usage:
 *   matmul_c --selftest
 *   matmul_c <n> <A.bin> <B.bin> <warmups> <reps> <out.bin|-> [--hold]
 *
 * Output (stdout), one line per run:  W|M,<index>,<nanoseconds>,<checksum>
 * then "DONE". With --hold the process waits for a line on stdin before
 * exiting, so the benchmark driver can read its peak working set.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static long long now_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (long long)((double)t.QuadPart * 1e9 / (double)freq.QuadPart);
}
#else
#include <time.h>
static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
#endif

__attribute__((noinline))
static void multiply(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
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

static double checksum(const double *c, int n) {
    double s = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) s += c[i];
    return s;
}

static double *alloc_matrix(int n) {
    double *m = calloc((size_t)n * n, sizeof(double));
    if (!m) { fprintf(stderr, "out of memory for n=%d\n", n); exit(3); }
    return m;
}

static double *read_matrix(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    double *m = alloc_matrix(n);
    if (fread(m, sizeof(double), (size_t)n * n, f) != (size_t)n * n) {
        fprintf(stderr, "short read in %s\n", path); exit(2);
    }
    fclose(f);
    return m;
}

/* ---------- self-test: known answers, exact comparison ---------- */

static int expect_equal(const char *name, const double *got, const double *want, int n) {
    for (int i = 0; i < n * n; i++) {
        if (got[i] != want[i]) {
            printf("FAIL %s at %d: got %.17g want %.17g\n", name, i, got[i], want[i]);
            return 1;
        }
    }
    printf("PASS %s\n", name);
    return 0;
}

static int selftest(void) {
    int fails = 0;
    double a2[] = {1, 2, 3, 4}, b2[] = {5, 6, 7, 8}, w2[] = {19, 22, 43, 50}, c2[4];
    multiply(a2, b2, c2, 2);
    fails += expect_equal("2x2_known", c2, w2, 2);

    double a3[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double b3[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    double w3[] = {30, 24, 18, 84, 69, 54, 138, 114, 90}, c3[9];
    multiply(a3, b3, c3, 3);
    fails += expect_equal("3x3_known", c3, w3, 3);

    int n = 5;
    double a[25], id[25], zero[25] = {0}, c[25];
    for (int i = 0; i < 25; i++) { a[i] = (double)(i * 7 % 11) - 5.0; id[i] = (i / n == i % n) ? 1.0 : 0.0; }
    multiply(a, id, c, n);
    fails += expect_equal("A_times_identity", c, a, n);
    multiply(id, a, c, n);
    fails += expect_equal("identity_times_A", c, a, n);
    multiply(a, zero, c, n);
    fails += expect_equal("A_times_zero", c, zero, n);
    printf(fails ? "SELFTEST FAILED\n" : "SELFTEST OK\n");
    return fails ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 7) {
        fprintf(stderr, "usage: %s <n> <A.bin> <B.bin> <warmups> <reps> <out.bin|-> [--hold]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int warmups = atoi(argv[4]);
    int reps = atoi(argv[5]);
    const char *out_path = argv[6];
    int hold = argc >= 8 && strcmp(argv[7], "--hold") == 0;

    double *a = read_matrix(argv[2], n);
    double *b = read_matrix(argv[3], n);
    double *c = alloc_matrix(n);

    for (int r = 0; r < warmups + reps; r++) {
        long long t0 = now_ns();
        multiply(a, b, c, n);
        long long t1 = now_ns();
        printf("%c,%d,%lld,%.17g\n", r < warmups ? 'W' : 'M', r < warmups ? r : r - warmups, t1 - t0, checksum(c, n));
        fflush(stdout);
    }

    if (strcmp(out_path, "-") != 0) {
        FILE *f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
        fwrite(c, sizeof(double), (size_t)n * n, f);
        fclose(f);
    }
    printf("DONE\n");
    fflush(stdout);
    if (hold) {
        int ch;
        while ((ch = getchar()) != EOF && ch != '\n') {}
    }
    free(a); free(b); free(c);
    return 0;
}
