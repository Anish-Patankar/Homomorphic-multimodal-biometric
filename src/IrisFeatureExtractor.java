import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import com.neurotec.biometrics.NBiometricStatus;
import com.neurotec.biometrics.NEAttributes;
import com.neurotec.biometrics.NERecord;
import com.neurotec.biometrics.NETemplate;
import com.neurotec.biometrics.NIris;
import com.neurotec.biometrics.NSubject;
import com.neurotec.biometrics.NTemplate;
import com.neurotec.biometrics.NTemplateSize;
import com.neurotec.biometrics.client.NBiometricClient;
import com.neurotec.geometry.NPoint;
import com.neurotec.io.NBuffer;
import com.neurotec.licensing.NLicense;

/**
 * IrisFeatureExtractor
 * ====================
 * Extracts iris feature vectors from images and writes JSON output
 * containing template metadata, segmentation boundaries, quality scores,
 * and the raw feature vector (Base64-encoded).
 *
 * Designed for FAR/FRR analysis pipelines.
 *
 * Modes of Operation:
 * -------------------
 * 1. Single Image Mode:
 *    java IrisFeatureExtractor <image> <output.json> <subject_id> <sample_id> [dataset] [subdataset]
 *
 * 2. Ultra-Fast In-Memory Batch Mode:
 *    java IrisFeatureExtractor --batch <images_root> <output_root> [dataset] [--skip-existing]
 *
 * Exit codes:
 *   0 = success
 *   1 = bad arguments / file not found
 *   2 = license error
 *   3 = extraction failed
 *   99 = unexpected exception
 */
public class IrisFeatureExtractor {

    private static final Set<String> SUPPORTED_EXTS = new HashSet<>(
            Arrays.asList(".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".wsq")
    );

    private static final Pattern PATTERN_A = Pattern.compile("^(\\d+)(left|right)$", Pattern.CASE_INSENSITIVE);
    private static final Pattern PATTERN_B = Pattern.compile("^Subject_(\\d+)$", Pattern.CASE_INSENSITIVE);

    // -----------------------------------------------------------------------
    // JSON helpers
    // -----------------------------------------------------------------------

    private static String jsonEscape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static void appendKV(StringBuilder sb, String key, String value, boolean last) {
        sb.append(String.format("  \"%s\": \"%s\"%s\n", key, jsonEscape(value), last ? "" : ","));
    }

    private static void appendKV(StringBuilder sb, String key, int value, boolean last) {
        sb.append(String.format("  \"%s\": %d%s\n", key, value, last ? "" : ","));
    }

    private static void appendKV(StringBuilder sb, String key, double value, boolean last) {
        sb.append(String.format("  \"%s\": %.4f%s\n", key, value, last ? "" : ","));
    }

    private static void appendKVNull(StringBuilder sb, String key, boolean last) {
        sb.append(String.format("  \"%s\": null%s\n", key, last ? "" : ","));
    }

    // -----------------------------------------------------------------------
    // Build JSON
    // -----------------------------------------------------------------------

    private static String buildJson(
            String dataset, String subdataset,
            String subjectId, String sampleId,
            String imagePath,
            NERecord record,
            NEAttributes attr,
            byte[] featureVectorBytes) {

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");

        // Metadata
        appendKV(sb, "dataset", dataset, false);
        appendKV(sb, "subdataset", subdataset, false);
        appendKV(sb, "subject", subjectId, false);
        appendKV(sb, "sample", sampleId, false);
        appendKV(sb, "extractor", "verieye_sdk", false);
        appendKV(sb, "source_image", imagePath, false);

        // Template record properties
        appendKV(sb, "position", record.getPosition().toString(), false);
        appendKV(sb, "quality", record.getQuality() & 0xFF, false);
        appendKV(sb, "image_width", record.getWidth(), false);
        appendKV(sb, "image_height", record.getHeight(), false);
        appendKV(sb, "cbeff_product_type", record.getCBEFFProductType(), false);
        appendKV(sb, "record_size_bytes", record.getSize(), false);

        // Segmentation attributes
        if (attr != null) {
            appendKV(sb, "detected_position", attr.getDetectedPosition().toString(), false);
            appendKV(sb, "iris_radius", attr.getIrisRadius() & 0xFF, false);
            appendKV(sb, "rotation_angle", attr.getRotationAngle(), false);
            appendKV(sb, "detection_confidence", attr.getDetectionConfidence() & 0xFF, false);

            // Inner boundary (pupil boundary)
            if (attr.isInnerBoundaryAvailable() && attr.getInnerBoundaryPoints().size() > 0) {
                sb.append("  \"inner_boundary_points\": [\n");
                int innerCount = attr.getInnerBoundaryPoints().size();
                for (int i = 0; i < innerCount; i++) {
                    NPoint p = attr.getInnerBoundaryPoints().get(i);
                    sb.append(String.format("    [%d, %d]%s\n", p.x, p.y,
                            (i < innerCount - 1) ? "," : ""));
                }
                sb.append("  ],\n");
            } else {
                sb.append("  \"inner_boundary_points\": [],\n");
            }

            // Outer boundary (iris/sclera boundary)
            if (attr.isOuterBoundaryAvailable() && attr.getOuterBoundaryPoints().size() > 0) {
                sb.append("  \"outer_boundary_points\": [\n");
                int outerCount = attr.getOuterBoundaryPoints().size();
                for (int i = 0; i < outerCount; i++) {
                    NPoint p = attr.getOuterBoundaryPoints().get(i);
                    sb.append(String.format("    [%d, %d]%s\n", p.x, p.y,
                            (i < outerCount - 1) ? "," : ""));
                }
                sb.append("  ],\n");
            } else {
                sb.append("  \"outer_boundary_points\": [],\n");
            }
        } else {
            appendKVNull(sb, "detected_position", false);
            appendKVNull(sb, "iris_radius", false);
            appendKVNull(sb, "rotation_angle", false);
            appendKVNull(sb, "detection_confidence", false);
            sb.append("  \"inner_boundary_points\": [],\n");
            sb.append("  \"outer_boundary_points\": [],\n");
        }

        // Feature vector — Binary bitstring (0s and 1s) and Base64 encoded
        appendKV(sb, "feature_vector_size_bytes", featureVectorBytes.length, false);

        StringBuilder binSb = new StringBuilder(featureVectorBytes.length * 8);
        for (byte b : featureVectorBytes) {
            for (int i = 7; i >= 0; i--) {
                binSb.append((b >> i) & 1);
            }
        }
        appendKV(sb, "feature_vector_binary", binSb.toString(), false);

        String b64 = Base64.getEncoder().encodeToString(featureVectorBytes);
        appendKV(sb, "feature_vector_base64", b64, true);

        sb.append("}");
        return sb.toString();
    }

    // -----------------------------------------------------------------------
    // Core extraction method
    // -----------------------------------------------------------------------

    public static boolean extractOne(
            NBiometricClient client,
            File imageFile,
            File outputFile,
            String subjectId,
            String sampleId,
            String dataset,
            String subdataset) {

        NSubject subject = null;
        NIris iris = null;
        try {
            subject = new NSubject();
            iris = new NIris();
            iris.setFileName(imageFile.getAbsolutePath());
            subject.getIrises().add(iris);

            NBiometricStatus status = client.createTemplate(subject);
            if (status != NBiometricStatus.OK) {
                System.err.println("FAIL (" + status + "): " + imageFile.getPath());
                return false;
            }

            NTemplate ntemplate = subject.getTemplate();
            NETemplate irisesTemplate = ntemplate != null ? ntemplate.getIrises() : null;
            if (irisesTemplate == null || irisesTemplate.getRecords().isEmpty()) {
                System.err.println("FAIL (no iris record in template): " + imageFile.getPath());
                return false;
            }

            NERecord record = irisesTemplate.getRecords().get(0);
            NEAttributes attr = (iris.getObjects() != null && !iris.getObjects().isEmpty())
                    ? iris.getObjects().get(0) : null;

            NBuffer recordBuffer = record.save();
            byte[] featureVectorBytes = recordBuffer.toByteArray();

            String json = buildJson(
                    dataset, subdataset,
                    subjectId, sampleId,
                    imageFile.getPath(),
                    record, attr,
                    featureVectorBytes
            );

            File parentDir = outputFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }

            // Write JSON output file
            try (Writer writer = new OutputStreamWriter(
                    new FileOutputStream(outputFile), StandardCharsets.UTF_8)) {
                writer.write(json);
            }

            // Write companion raw binary file (.bin) alongside .json
            String jsonPath = outputFile.getAbsolutePath();
            String binPath = jsonPath.endsWith(".json")
                    ? jsonPath.substring(0, jsonPath.length() - 5) + ".bin"
                    : jsonPath + ".bin";
            File binFile = new File(binPath);
            try (FileOutputStream fos = new FileOutputStream(binFile)) {
                fos.write(featureVectorBytes);
            }

            return true;

        } catch (Throwable e) {
            System.err.println("EXCEPTION on " + imageFile.getPath() + ": " + e.getMessage());
            return false;
        } finally {
            if (iris != null) iris.dispose();
            if (subject != null) subject.dispose();
        }
    }

    // -----------------------------------------------------------------------
    // Batch helper class & scanner
    // -----------------------------------------------------------------------

    static class BatchItem {
        File imageFile;
        File outputFile;
        String subjectId;
        String sampleId;
        String subdataset;

        BatchItem(File imageFile, File outputFile, String subjectId, String sampleId, String subdataset) {
            this.imageFile = imageFile;
            this.outputFile = outputFile;
            this.subjectId = subjectId;
            this.sampleId = sampleId;
            this.subdataset = subdataset;
        }
    }

    private static List<BatchItem> discoverBatchItems(File imagesRoot, File outputRoot) {
        List<BatchItem> items = new ArrayList<>();
        File[] folders = imagesRoot.listFiles(File::isDirectory);
        if (folders == null) return items;
        Arrays.sort(folders);

        for (File folder : folders) {
            String folderName = folder.getName();
            Matcher ma = PATTERN_A.matcher(folderName);
            Matcher mb = PATTERN_B.matcher(folderName);

            String subjectId;
            String side;

            if (ma.matches()) {
                subjectId = ma.group(1);
                side = ma.group(2).toLowerCase();
            } else if (mb.matches()) {
                subjectId = mb.group(1);
                side = "";
            } else {
                continue;
            }

            File[] files = folder.listFiles(File::isFile);
            if (files == null) continue;
            Arrays.sort(files);

            for (File file : files) {
                String name = file.getName();
                int dotIdx = name.lastIndexOf('.');
                if (dotIdx < 0) continue;
                String ext = name.substring(dotIdx).toLowerCase();
                if (!SUPPORTED_EXTS.contains(ext)) continue;

                String sampleId = name.substring(0, dotIdx);
                String inferredSide = side;
                if (inferredSide.isEmpty()) {
                    String u = sampleId.toUpperCase();
                    if (u.contains("L")) inferredSide = "left";
                    else if (u.contains("R")) inferredSide = "right";
                    else inferredSide = "default";
                }

                String outSubdir = !side.isEmpty() ? folderName : ("Subject_" + subjectId);
                File outDir = new File(outputRoot, outSubdir);
                File outFile = new File(outDir, sampleId + ".json");
                String subdataset = inferredSide.substring(0, 1).toUpperCase() + inferredSide.substring(1);

                items.add(new BatchItem(file, outFile, subjectId, sampleId, subdataset));
            }
        }
        return items;
    }

    // -----------------------------------------------------------------------
    // License helper
    // -----------------------------------------------------------------------

    private static boolean obtainLicense() {
        String[] licenseNames = {"IrisExtractor", "IrisClient"};
        for (String lic : licenseNames) {
            try {
                if (NLicense.obtain("/local", 5000, lic)) {
                    return true;
                }
            } catch (Throwable ignored) {}
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    public static void main(String[] args) {
        System.setProperty(
            "jna.library.path",
            "C:\\Program Files\\Neurotechnology\\Neurotec_Biometric_2025_2_SDK\\Bin\\Win64_x64"
        );

        if (args.length == 0) {
            System.err.println("Usage:");
            System.err.println("  Single image: java IrisFeatureExtractor <image> <output.json> <subject_id> <sample_id> [dataset] [subdataset]");
            System.err.println("  Batch mode:   java IrisFeatureExtractor --batch <images_root> <output_root> [dataset] [--skip-existing]");
            System.exit(1);
        }

        // ------------------ Batch Mode ------------------
        if (args[0].equalsIgnoreCase("--batch")) {
            if (args.length < 3) {
                System.err.println("Usage: java IrisFeatureExtractor --batch <images_root> <output_root> [dataset] [--skip-existing]");
                System.exit(1);
            }

            File imagesRoot = new File(args[1]);
            File outputRoot = new File(args[2]);
            String dataset = (args.length > 3 && !args[3].startsWith("--")) ? args[3] : "IrisDataset";
            boolean skipExisting = false;
            for (String arg : args) {
                if (arg.equalsIgnoreCase("--skip-existing")) {
                    skipExisting = true;
                }
            }

            if (!imagesRoot.exists() || !imagesRoot.isDirectory()) {
                System.err.println("ERROR: Images root directory not found: " + imagesRoot.getPath());
                System.exit(1);
            }

            System.out.println("Scanning directory: " + imagesRoot.getAbsolutePath());
            List<BatchItem> items = discoverBatchItems(imagesRoot, outputRoot);
            System.out.println("Discovered " + items.size() + " iris images.");

            if (items.isEmpty()) {
                System.out.println("No matching images found.");
                return;
            }

            if (!obtainLicense()) {
                System.err.println("ERROR: Could not obtain IrisExtractor or IrisClient license.");
                System.exit(2);
            }

            int threads = Math.min(8, Runtime.getRuntime().availableProcessors());
            for (int i = 0; i < args.length; i++) {
                if (args[i].equalsIgnoreCase("--threads") && i + 1 < args.length) {
                    try { threads = Integer.parseInt(args[i + 1]); } catch (Exception ignored) {}
                }
            }

            ExecutorService pool = Executors.newFixedThreadPool(threads);
            List<NBiometricClient> createdClients = Collections.synchronizedList(new ArrayList<>());
            ThreadLocal<NBiometricClient> clientHolder = ThreadLocal.withInitial(() -> {
                NBiometricClient c = new NBiometricClient();
                c.setIrisesTemplateSize(NTemplateSize.LARGE);
                createdClients.add(c);
                return c;
            });

            long startTime = System.currentTimeMillis();
            AtomicInteger ok = new AtomicInteger(0);
            AtomicInteger skipped = new AtomicInteger(0);
            AtomicInteger failed = new AtomicInteger(0);
            AtomicInteger completed = new AtomicInteger(0);
            int total = items.size();

            System.out.printf("Starting in-memory batch extraction with %d worker threads...%n", threads);

            final boolean doSkip = skipExisting;
            for (BatchItem item : items) {
                final String currentDataset = dataset;
                pool.submit(() -> {
                    try {
                        if (doSkip && item.outputFile.exists()) {
                            skipped.incrementAndGet();
                            int c = completed.incrementAndGet();
                            if (c % 100 == 0 || c == total) {
                                System.out.printf("[%4d/%4d] SKIP (already exists): %s%n", c, total, item.outputFile.getName());
                            }
                            return;
                        }

                        NBiometricClient client = clientHolder.get();
                        boolean success = extractOne(
                                client,
                                item.imageFile,
                                item.outputFile,
                                item.subjectId,
                                item.sampleId,
                                currentDataset,
                                item.subdataset
                        );

                        int c = completed.incrementAndGet();
                        if (success) {
                            ok.incrementAndGet();
                            if (c % 25 == 0 || c == total) {
                                System.out.printf("[%4d/%4d] OK   %s%n", c, total, item.imageFile.getName());
                            }
                        } else {
                            failed.incrementAndGet();
                            System.out.printf("[%4d/%4d] FAIL %s%n", c, total, item.imageFile.getPath());
                        }
                    } catch (Throwable t) {
                        failed.incrementAndGet();
                        completed.incrementAndGet();
                        System.err.println("Worker error on " + item.imageFile.getName() + ": " + t.getMessage());
                    }
                });
            }

            pool.shutdown();
            try {
                pool.awaitTermination(2, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                System.err.println("Batch extraction interrupted!");
            }

            for (NBiometricClient c : createdClients) {
                try { c.dispose(); } catch (Throwable ignored) {}
            }

            long elapsedMs = System.currentTimeMillis() - startTime;
            double elapsedSec = elapsedMs / 1000.0;
            System.out.println();
            System.out.printf("Batch extraction finished in %.2f seconds!%n", elapsedSec);
            System.out.printf("  OK:      %d%n", ok.get());
            System.out.printf("  SKIPPED: %d%n", skipped.get());
            System.out.printf("  FAILED:  %d%n", failed.get());
            System.out.printf("  TOTAL:   %d%n", total);
            System.out.println("Output written to: " + outputRoot.getAbsolutePath());
            return;
        }

        // ------------------ Single Image Mode ------------------
        if (args.length < 4) {
            System.err.println("Usage: java IrisFeatureExtractor <image> <output.json> <subject_id> <sample_id> [dataset] [subdataset]");
            System.exit(1);
        }

        String imagePath  = args[0];
        String outputPath = args[1];
        String subjectId  = args[2];
        String sampleId   = args[3];
        String dataset    = args.length > 4 ? args[4] : "IrisDataset";
        String subdataset = args.length > 5 ? args[5] : "Default";

        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            System.err.println("Image file not found: " + imagePath);
            System.exit(1);
        }

        if (!obtainLicense()) {
            System.err.println("ERROR: Could not obtain IrisExtractor or IrisClient license.");
            System.exit(2);
        }

        NBiometricClient client = null;
        try {
            client = new NBiometricClient();
            client.setIrisesTemplateSize(NTemplateSize.LARGE);

            boolean success = extractOne(
                    client,
                    imageFile,
                    new File(outputPath),
                    subjectId,
                    sampleId,
                    dataset,
                    subdataset
            );

            if (success) {
                System.out.println("OK: " + outputPath);
            } else {
                System.exit(3);
            }
        } finally {
            if (client != null) client.dispose();
        }
    }
}
