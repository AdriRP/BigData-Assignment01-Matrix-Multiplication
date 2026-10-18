/**
 * Supplementary diagnostic (NOT part of the controlled comparison).
 * Same i-j-k loop as src/java/MatMul.java, but only the first `rows` rows of C
 * are computed, so the cost per multiply-add of the column-wise access to B can
 * be measured at sizes too large for a full run. Two layouts of B:
 *   java_2d   : double[n][n] (as in MatMul.java)
 *   java_flat : one double[n*n] array, row-major, indexed k*n + j (as in C)
 *
 * Usage: java Probe <2d|flat> <n> <rows> <warmups> <reps>
 * Output: one line per measured run: java_<layout>,<n>,<rows>,<rep>,<ns>,<checksum>
 */
public class Probe {
    static long state = 2026;

    static double nextValue() {               // xorshift64*, uniform in [-1, 1), same as probe.c
        state ^= state >>> 12; state ^= state << 25; state ^= state >>> 27;
        return (double) ((state * 2685821657736338717L) >>> 11) * 0x1.0p-52 - 1.0;
    }

    static void multiply2d(double[][] a, double[][] b, double[][] c, int n, int rows) {
        for (int i = 0; i < rows; i++) {
            double[] ai = a[i];
            double[] ci = c[i];
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += ai[k] * b[k][j];
                }
                ci[j] = sum;
            }
        }
    }

    static void multiplyFlat(double[] a, double[] b, double[] c, int n, int rows) {
        for (int i = 0; i < rows; i++) {
            int ai = i * n;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[ai + k] * b[k * n + j];
                }
                c[ai + j] = sum;
            }
        }
    }

    public static void main(String[] args) {
        String layout = args[0];
        int n = Integer.parseInt(args[1]), rows = Integer.parseInt(args[2]);
        int warmups = Integer.parseInt(args[3]), reps = Integer.parseInt(args[4]);
        boolean flat = layout.equals("flat");
        double[][] a2 = null, b2 = null, c2 = null;
        double[] a1 = null, b1 = null, c1 = null;
        if (flat) {
            a1 = new double[rows * n]; b1 = new double[n * n]; c1 = new double[rows * n];
            for (int i = 0; i < rows * n; i++) a1[i] = nextValue();
            for (int i = 0; i < n * n; i++) b1[i] = nextValue();
        } else {
            a2 = new double[rows][n]; b2 = new double[n][n]; c2 = new double[rows][n];
            for (double[] r : a2) for (int j = 0; j < n; j++) r[j] = nextValue();
            for (double[] r : b2) for (int j = 0; j < n; j++) r[j] = nextValue();
        }
        for (int r = 0; r < warmups + reps; r++) {
            long t0 = System.nanoTime();
            if (flat) multiplyFlat(a1, b1, c1, n, rows); else multiply2d(a2, b2, c2, n, rows);
            long t1 = System.nanoTime();
            double s = 0.0;
            if (flat) { for (double v : c1) s += v; } else { for (double[] row : c2) for (double v : row) s += v; }
            if (r >= warmups) {
                System.out.println("java_" + layout + "," + n + "," + rows + "," + (r - warmups) + "," + (t1 - t0) + ","
                        + String.format(java.util.Locale.ROOT, "%.17g", s));
            }
        }
    }
}
