import com.neurotec.biometrics.NBiometricStatus;
import com.neurotec.biometrics.NEAttributes;
import com.neurotec.biometrics.NIris;
import com.neurotec.biometrics.NSubject;
import com.neurotec.biometrics.NTemplateSize;
import com.neurotec.biometrics.client.NBiometricClient;
import com.neurotec.licensing.NLicense;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Runs Neurotechnology iris segmentation over every image under a dataset
 * folder and writes the pupil (inner) and limbus (outer) boundary points to a
 * JSON Lines file, one record per image, for use by the Python pipeline
 * (neurotec_boundaries.py / iris_encoder.image_to_iris_code_from_boundaries).
 *
 * Built from the single-image TestSegmentation program: same license, same
 * client settings, same API calls -- only batching, JSON output and resume
 * support are added. The license is obtained and the client created ONCE,
 * not per image.
 *
 * Usage:
 *   java ExportBoundaries <dataset-root> <output.jsonl>
 *
 * Re-running with the same output file skips images already recorded, so an
 * interrupted export can simply be restarted.
 *
 * Output (one line per image; "path" is relative to <dataset-root>, forward
 * slashes, and is always the first key):
 *   {"path":"012left/03.bmp","status":"OK","objects":1,"inner":[[235,76],...],"outer":[[289,76],...]}
 *   {"path":"012left/04.bmp","status":"TARGET_NOT_FOUND"}
 */
public class ExportBoundaries {

    private static final String[] EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"};
    private static final Pattern NUM = Pattern.compile("-?\\d+(?:\\.\\d+)?");
    private static final Pattern DONE = Pattern.compile("^\\{\"path\":\"([^\"]*)\"");

    public static void main(String[] args) throws Exception {
        if (System.getProperty("jna.library.path") == null) {
            System.setProperty(
                "jna.library.path",
                "C:\\Program Files\\Neurotechnology\\Neurotec_Biometric_2025_2_SDK\\Bin\\Win64_x64"
            );
        }

        if (args.length != 2) {
            System.out.println("Usage: java ExportBoundaries <dataset-root> <output.jsonl>");
            System.exit(1);
        }

        Path root = Paths.get(args[0]).toAbsolutePath().normalize();
        Path out = Paths.get(args[1]);

        List<Path> images;
        try (Stream<Path> s = Files.walk(root)) {
            images = s.filter(Files::isRegularFile)
                      .filter(ExportBoundaries::isImage)
                      .sorted()
                      .collect(Collectors.toList());
        }

        // Resume support: skip images that already have a complete record.
        Set<String> done = new HashSet<>();
        boolean outExists = Files.exists(out) && Files.size(out) > 0;
        if (outExists) {
            for (String line : Files.readAllLines(out, StandardCharsets.UTF_8)) {
                Matcher m = DONE.matcher(line);
                if (m.find() && line.endsWith("}")) {
                    done.add(m.group(1));
                }
            }
        }

        System.out.println("Found " + images.size() + " images under " + root
                           + " (" + done.size() + " already exported).");

        NBiometricClient client = null;
        try (BufferedWriter w = Files.newBufferedWriter(
                out, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {

            if (!NLicense.obtain("/local", 5000, "IrisExtractor")) {
                System.err.println("Could not obtain IrisExtractor license.");
                System.exit(-1);
            }

            client = new NBiometricClient();
            client.setIrisesTemplateSize(NTemplateSize.LARGE);

            // If a previous run was interrupted mid-line, start on a fresh line.
            if (outExists) {
                w.write("\n");
            }

            int processed = 0, ok = 0;
            long t0 = System.currentTimeMillis();

            for (Path img : images) {
                String rel = root.relativize(img.toAbsolutePath().normalize())
                                 .toString().replace('\\', '/');
                if (done.contains(rel)) {
                    continue;
                }

                String record = processOne(client, img, rel);
                w.write(record);
                w.write("\n");
                w.flush();

                processed++;
                if (record.contains("\"status\":\"OK\"")) {
                    ok++;
                }
                if (processed % 25 == 0) {
                    double secs = (System.currentTimeMillis() - t0) / 1000.0;
                    System.out.println("Processed " + processed + " (" + ok + " OK) in "
                                       + String.format(Locale.ROOT, "%.0f", secs) + " s");
                }
            }

            System.out.println("Done. Processed " + processed + " new images, " + ok
                               + " segmented OK. Output: " + out.toAbsolutePath());

        } finally {
            if (client != null) {
                client.dispose();
            }
        }
    }

    private static boolean isImage(Path p) {
        String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String e : EXTS) {
            if (n.endsWith(e)) {
                return true;
            }
        }
        return false;
    }

    /** Segments one image and returns its JSON record (never throws). */
    private static String processOne(NBiometricClient client, Path img, String rel) {
        NSubject subject = null;
        NIris iris = null;
        try {
            subject = new NSubject();
            iris = new NIris();
            iris.setFileName(img.toString());
            subject.getIrises().add(iris);

            NBiometricStatus status = client.createTemplate(subject);
            if (status != NBiometricStatus.OK) {
                return record(rel, String.valueOf(status), 0, null, null);
            }

            int count = iris.getObjects().size();
            if (count < 1) {
                return record(rel, "NO_IRIS_OBJECT", 0, null, null);
            }

            NEAttributes attr = iris.getObjects().get(0);
            if (!attr.isInnerBoundaryAvailable() || !attr.isOuterBoundaryAvailable()) {
                return record(rel, "BOUNDARY_UNAVAILABLE", count, null, null);
            }

            StringBuilder inner = new StringBuilder("[");
            int ni = attr.getInnerBoundaryPoints().size();
            for (int j = 0; j < ni; j++) {
                if (j > 0) inner.append(',');
                appendPoint(inner, attr.getInnerBoundaryPoints().get(j));
            }
            inner.append(']');

            StringBuilder outer = new StringBuilder("[");
            int no = attr.getOuterBoundaryPoints().size();
            for (int j = 0; j < no; j++) {
                if (j > 0) outer.append(',');
                appendPoint(outer, attr.getOuterBoundaryPoints().get(j));
            }
            outer.append(']');

            return record(rel, "OK", count, inner.toString(), outer.toString());

        } catch (Throwable e) {
            return record(rel, "ERROR_" + e.getClass().getSimpleName(), 0, null, null);
        } finally {
            if (iris != null) iris.dispose();
            if (subject != null) subject.dispose();
        }
    }

    /**
     * Appends "[x,y]" for a point. Coordinates are read from the point's text
     * form (the SDK prints points as "NPoint(235, 76)"), which avoids relying
     * on field/accessor names.
     */
    private static void appendPoint(StringBuilder sb, Object point) {
        Matcher m = NUM.matcher(String.valueOf(point));
        if (!m.find()) {
            throw new IllegalStateException("Cannot parse point: " + point);
        }
        String x = m.group();
        if (!m.find()) {
            throw new IllegalStateException("Cannot parse point: " + point);
        }
        String y = m.group();
        sb.append('[').append(x).append(',').append(y).append(']');
    }

    private static String record(String rel, String status, int objects, String inner, String outer) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"path\":\"").append(esc(rel)).append("\",\"status\":\"").append(esc(status)).append('"');
        if (inner != null && outer != null) {
            sb.append(",\"objects\":").append(objects);
            sb.append(",\"inner\":").append(inner);
            sb.append(",\"outer\":").append(outer);
        }
        sb.append('}');
        return sb.toString();
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
