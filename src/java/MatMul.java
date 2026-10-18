import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Basic dense matrix multiplication, C = A * B, in Java.
 *
 * Storage: double[n][n], i.e. an array of n separate row arrays (row-major
 * inside each row, rows not guaranteed to be contiguous with each other).
 * Algorithm: textbook i-j-k triple loop, scalar accumulator, O(n^3).
 *
 * Usage:
 *   java MatMul --selftest
 *   java MatMul <n> <A.bin> <B.bin> <warmups> <reps> <out.bin|-> [--hold]
 *
 * Output (stdout), one line per run:  W|M,<index>,<nanoseconds>,<checksum>
 * then "DONE". With --hold the process waits for a line on stdin before
 * exiting, so the benchmark driver can read its peak working set.
 */
public class MatMul {

    static void multiply(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
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

    static double checksum(double[][] c) {
        double s = 0.0;
        for (double[] row : c) {
            for (double v : row) {
                s += v;
            }
        }
        return s;
    }

    static double[][] readMatrix(String path, int n) throws IOException {
        double[][] m = new double[n][n];
        try (FileChannel ch = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                buf.clear();
                while (buf.hasRemaining()) {
                    if (ch.read(buf) < 0) {
                        throw new IOException("short read in " + path);
                    }
                }
                buf.flip();
                buf.asDoubleBuffer().get(m[i]);
            }
        }
        return m;
    }

    static void writeMatrix(String path, double[][] m, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
        try (OutputStream out = new FileOutputStream(path)) {
            for (double[] row : m) {
                buf.clear();
                buf.asDoubleBuffer().put(row);
                out.write(buf.array());
            }
        }
    }

    // ---------- self-test: known answers, exact comparison ----------

    static int expectEqual(String name, double[][] got, double[][] want) {
        for (int i = 0; i < want.length; i++) {
            for (int j = 0; j < want.length; j++) {
                if (got[i][j] != want[i][j]) {
                    System.out.printf("FAIL %s at (%d,%d): got %s want %s%n", name, i, j, got[i][j], want[i][j]);
                    return 1;
                }
            }
        }
        System.out.println("PASS " + name);
        return 0;
    }

    static int selftest() {
        int fails = 0;
        double[][] c2 = new double[2][2];
        multiply(new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}, c2, 2);
        fails += expectEqual("2x2_known", c2, new double[][]{{19, 22}, {43, 50}});

        double[][] c3 = new double[3][3];
        multiply(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                 new double[][]{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}, c3, 3);
        fails += expectEqual("3x3_known", c3, new double[][]{{30, 24, 18}, {84, 69, 54}, {138, 114, 90}});

        int n = 5;
        double[][] a = new double[n][n], id = new double[n][n], zero = new double[n][n], c = new double[n][n];
        for (int i = 0; i < n * n; i++) {
            a[i / n][i % n] = (double) (i * 7 % 11) - 5.0;
            id[i / n][i % n] = (i / n == i % n) ? 1.0 : 0.0;
        }
        multiply(a, id, c, n);
        fails += expectEqual("A_times_identity", c, a);
        multiply(id, a, c, n);
        fails += expectEqual("identity_times_A", c, a);
        multiply(a, zero, c, n);
        fails += expectEqual("A_times_zero", c, zero);
        System.out.println(fails == 0 ? "SELFTEST OK" : "SELFTEST FAILED");
        return fails == 0 ? 0 : 1;
    }

    public static void main(String[] args) throws IOException {
        if (args.length >= 1 && args[0].equals("--selftest")) {
            System.exit(selftest());
        }
        if (args.length < 6) {
            System.err.println("usage: java MatMul <n> <A.bin> <B.bin> <warmups> <reps> <out.bin|-> [--hold]");
            System.exit(1);
        }
        int n = Integer.parseInt(args[0]);
        int warmups = Integer.parseInt(args[3]);
        int reps = Integer.parseInt(args[4]);
        String outPath = args[5];
        boolean hold = args.length >= 7 && args[6].equals("--hold");

        double[][] a = readMatrix(args[1], n);
        double[][] b = readMatrix(args[2], n);
        double[][] c = new double[n][n];

        for (int r = 0; r < warmups + reps; r++) {
            long t0 = System.nanoTime();
            multiply(a, b, c, n);
            long t1 = System.nanoTime();
            boolean warm = r < warmups;
            System.out.println((warm ? "W," : "M,") + (warm ? r : r - warmups) + "," + (t1 - t0) + ","
                    + String.format(java.util.Locale.ROOT, "%.17g", checksum(c)));
            System.out.flush();
        }

        if (!outPath.equals("-")) {
            writeMatrix(outPath, c, n);
        }
        System.out.println("DONE");
        System.out.flush();
        if (hold) {
            System.in.read();
        }
    }
}
