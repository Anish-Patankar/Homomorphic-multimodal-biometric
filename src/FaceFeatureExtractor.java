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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import com.neurotec.biometrics.NBiometricStatus;
import com.neurotec.biometrics.NFace;
import com.neurotec.biometrics.NLAttributes;
import com.neurotec.biometrics.NLFeaturePoint;
import com.neurotec.biometrics.NLRecord;
import com.neurotec.biometrics.NLTemplate;
import com.neurotec.biometrics.NSubject;
import com.neurotec.biometrics.NTemplate;
import com.neurotec.biometrics.NTemplateSize;
import com.neurotec.biometrics.client.NBiometricClient;
import com.neurotec.geometry.NRect;
import com.neurotec.io.NBuffer;
import com.neurotec.licensing.NLicense;

/**
 * FaceFeatureExtractor
 * ====================
 * Extracts facial feature vectors from images using Neurotechnology VeriLook SDK.
 * Outputs both:
 *   1. JSON metadata file with detection bounding box, pose (yaw, pitch, roll),
 *      facial landmarks (68 points), quality score, binary bitstring, and Base64.
 *   2. Companion raw binary file (.bin) containing the exact serialized feature vector bytes.
 *
 * Designed for datasets such as CelebA, LFW, or custom biometric databases.
 *
 * Usage:
 *   Single image:
 *     java FaceFeatureExtractor <image> <output.json> <subject_id> <sample_id> [dataset_name]
 *
 *   Batch mode:
 *     java FaceFeatureExtractor --batch <images_dir> <output_dir> [dataset] [--max N] [--threads T] [--skip-existing]
 */
public class FaceFeatureExtractor {

    private static final Set<String> SUPPORTED_EXTS = new HashSet<>(
            Arrays.asList(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".wsq")
    );

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
        sb.append(String.format("  \"%s\": %.2f%s\n", key, value, last ? "" : ","));
    }

    private static void appendKVNull(StringBuilder sb, String key, boolean last) {
        sb.append(String.format("  \"%s\": null%s\n", key, last ? "" : ","));
    }

    // -----------------------------------------------------------------------
    // Build JSON
    // -----------------------------------------------------------------------

    private static String buildJson(
            String dataset,
            String subjectId,
            String sampleId,
            String imagePath,
            NLRecord record,
            NLAttributes attr,
            byte[] featureVectorBytes) {

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");

        // Metadata
        appendKV(sb, "dataset", dataset, false);
        appendKV(sb, "subject", subjectId, false);
        appendKV(sb, "sample", sampleId, false);
        appendKV(sb, "extractor", "verilook_sdk", false);
        appendKV(sb, "source_image", imagePath, false);

        // Record properties
        int quality = (record != null) ? (record.getQuality() & 0xFF) : 0;
        appendKV(sb, "quality", quality, false);
        appendKV(sb, "cbeff_product_type", record != null ? record.getCBEFFProductType() : 0, false);
        appendKV(sb, "record_size_bytes", featureVectorBytes.length, false);

        // Bounding box & Pose
        if (attr != null) {
            NRect rect = attr.getBoundingRect();
            sb.append("  \"bounding_box\": {\n");
            sb.append(String.format("    \"x\": %d,\n", rect.left));
            sb.append(String.format("    \"y\": %d,\n", rect.top));
            sb.append(String.format("    \"width\": %d,\n", rect.width()));
            sb.append(String.format("    \"height\": %d\n", rect.height()));
            sb.append("  },\n");

            appendKV(sb, "yaw", attr.getYaw(), false);
            appendKV(sb, "pitch", attr.getPitch(), false);
            appendKV(sb, "roll", attr.getRoll(), false);

            if (attr.getGender() != null) {
                appendKV(sb, "gender", attr.getGender().toString(), false);
                appendKV(sb, "gender_confidence", attr.getGenderConfidence() & 0xFF, false);
            } else {
                appendKVNull(sb, "gender", false);
                appendKVNull(sb, "gender_confidence", false);
            }

            // Key landmark centers
            sb.append("  \"landmarks\": {\n");
            NLFeaturePoint re = attr.getRightEyeCenter();
            NLFeaturePoint le = attr.getLeftEyeCenter();
            NLFeaturePoint nt = attr.getNoseTip();
            NLFeaturePoint mc = attr.getMouthCenter();

            if (re != null && re.confidence > 0) {
                sb.append(String.format("    \"right_eye\": [%d, %d],\n", re.x, re.y));
            } else {
                sb.append("    \"right_eye\": null,\n");
            }
            if (le != null && le.confidence > 0) {
                sb.append(String.format("    \"left_eye\": [%d, %d],\n", le.x, le.y));
            } else {
                sb.append("    \"left_eye\": null,\n");
            }
            if (nt != null && nt.confidence > 0) {
                sb.append(String.format("    \"nose_tip\": [%d, %d],\n", nt.x, nt.y));
            } else {
                sb.append("    \"nose_tip\": null,\n");
            }
            if (mc != null && mc.confidence > 0) {
                sb.append(String.format("    \"mouth_center\": [%d, %d]\n", mc.x, mc.y));
            } else {
                sb.append("    \"mouth_center\": null\n");
            }
            sb.append("  },\n");

            // All detected 68 feature points
            int ptCount = (attr.getFeaturePoints() != null) ? attr.getFeaturePoints().size() : 0;
            appendKV(sb, "feature_points_count", ptCount, false);
            sb.append("  \"feature_points\": [\n");
            for (int i = 0; i < ptCount; i++) {
                NLFeaturePoint pt = attr.getFeaturePoints().get(i);
                sb.append(String.format("    {\"code\": %d, \"x\": %d, \"y\": %d, \"confidence\": %d}%s\n",
                        pt.code, pt.x, pt.y, pt.confidence & 0xFF,
                        (i < ptCount - 1) ? "," : ""));
            }
            sb.append("  ],\n");

        } else {
            appendKVNull(sb, "bounding_box", false);
            appendKVNull(sb, "yaw", false);
            appendKVNull(sb, "pitch", false);
            appendKVNull(sb, "roll", false);
            appendKVNull(sb, "gender", false);
            appendKVNull(sb, "gender_confidence", false);
            appendKVNull(sb, "landmarks", false);
            appendKV(sb, "feature_points_count", 0, false);
            sb.append("  \"feature_points\": [],\n");
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
            String dataset) {

        NSubject subject = null;
        NFace face = null;
        try {
            subject = new NSubject();
            face = new NFace();
            face.setFileName(imageFile.getAbsolutePath());
            subject.getFaces().add(face);

            NBiometricStatus status = client.createTemplate(subject);
            if (status != NBiometricStatus.OK) {
                System.err.println("FAIL (" + status + "): " + imageFile.getPath());
                return false;
            }

            NTemplate ntemplate = subject.getTemplate();
            NLTemplate facesTemplate = (ntemplate != null) ? ntemplate.getFaces() : null;
            if (facesTemplate == null || facesTemplate.getRecords().isEmpty()) {
                System.err.println("FAIL (no face record in template): " + imageFile.getPath());
                return false;
            }

            NLRecord record = facesTemplate.getRecords().get(0);
            NLAttributes attr = (face.getObjects() != null && !face.getObjects().isEmpty())
                    ? face.getObjects().get(0) : null;

            NBuffer recordBuffer = record.save();
            byte[] featureVectorBytes = recordBuffer.toByteArray();

            String json = buildJson(
                    dataset,
                    subjectId, sampleId,
                    imageFile.getPath(),
                    record, attr,
                    featureVectorBytes
            );

            File parentDir = outputFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }

            // 1. Write JSON output file
            try (Writer writer = new OutputStreamWriter(
                    new FileOutputStream(outputFile), StandardCharsets.UTF_8)) {
                writer.write(json);
            }

            // 2. Write companion raw binary file (.bin)
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
            if (face != null) face.dispose();
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

        BatchItem(File imageFile, File outputFile, String subjectId, String sampleId) {
            this.imageFile = imageFile;
            this.outputFile = outputFile;
            this.subjectId = subjectId;
            this.sampleId = sampleId;
        }
    }

    private static List<BatchItem> discoverBatchItems(File imagesRoot, File outputRoot, int maxItems) {
        List<BatchItem> items = new ArrayList<>();

        // Check if root has subdirectories or direct images
        File[] subdirs = imagesRoot.listFiles(File::isDirectory);
        boolean hasSubdirs = subdirs != null && subdirs.length > 0;

        if (hasSubdirs) {
            Arrays.sort(subdirs);
            for (File dir : subdirs) {
                File[] files = dir.listFiles(File::isFile);
                if (files == null) continue;
                Arrays.sort(files);

                String subjectId = dir.getName();
                for (File file : files) {
                    String name = file.getName();
                    int dot = name.lastIndexOf('.');
                    if (dot < 0) continue;
                    String ext = name.substring(dot).toLowerCase();
                    if (!SUPPORTED_EXTS.contains(ext)) continue;

                    String sampleId = name.substring(0, dot);
                    File outDir = new File(outputRoot, subjectId);
                    File outFile = new File(outDir, sampleId + ".json");
                    items.add(new BatchItem(file, outFile, subjectId, sampleId));

                    if (maxItems > 0 && items.size() >= maxItems) {
                        return items;
                    }
                }
            }
        } else {
            // Flat directory of images (like img_align_celeba/000001.jpg)
            File[] files = imagesRoot.listFiles(File::isFile);
            if (files != null) {
                Arrays.sort(files);
                for (File file : files) {
                    String name = file.getName();
                    int dot = name.lastIndexOf('.');
                    if (dot < 0) continue;
                    String ext = name.substring(dot).toLowerCase();
                    if (!SUPPORTED_EXTS.contains(ext)) continue;

                    String sampleId = name.substring(0, dot);
                    String subjectId = sampleId; // For CelebA default to filename stem
                    File outFile = new File(outputRoot, sampleId + ".json");
                    items.add(new BatchItem(file, outFile, subjectId, sampleId));

                    if (maxItems > 0 && items.size() >= maxItems) {
                        return items;
                    }
                }
            }
        }
        return items;
    }

    // -----------------------------------------------------------------------
    // License helper
    // -----------------------------------------------------------------------

    private static boolean obtainLicense() {
        String[] licenseNames = {"FaceExtractor", "FaceClient"};
        for (String lic : licenseNames) {
            try {
                if (NLicense.obtain("/local", 5000, lic)) {
                    System.out.println("Obtained license: " + lic);
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
            System.err.println("  Single image: java FaceFeatureExtractor <image> <output.json> <subject_id> <sample_id> [dataset]");
            System.err.println("  Batch mode:   java FaceFeatureExtractor --batch <images_dir> <output_dir> [dataset] [--max N] [--threads T] [--skip-existing]");
            System.exit(1);
        }

        // ------------------ Batch Mode ------------------
        if (args[0].equalsIgnoreCase("--batch")) {
            if (args.length < 3) {
                System.err.println("Usage: java FaceFeatureExtractor --batch <images_dir> <output_dir> [dataset] [--max N] [--threads T] [--skip-existing]");
                System.exit(1);
            }

            File imagesRoot = new File(args[1]);
            File outputRoot = new File(args[2]);
            String dataset = (args.length > 3 && !args[3].startsWith("--")) ? args[3] : "CelebA";

            int maxItems = 0;
            int threads = Math.min(8, Runtime.getRuntime().availableProcessors());
            boolean skipExisting = false;

            for (int i = 0; i < args.length; i++) {
                if (args[i].equalsIgnoreCase("--max") && i + 1 < args.length) {
                    try { maxItems = Integer.parseInt(args[i + 1]); } catch (Exception ignored) {}
                }
                if (args[i].equalsIgnoreCase("--threads") && i + 1 < args.length) {
                    try { threads = Integer.parseInt(args[i + 1]); } catch (Exception ignored) {}
                }
                if (args[i].equalsIgnoreCase("--skip-existing")) {
                    skipExisting = true;
                }
            }

            if (!imagesRoot.exists() || !imagesRoot.isDirectory()) {
                System.err.println("ERROR: Images root directory not found: " + imagesRoot.getPath());
                System.exit(1);
            }

            System.out.println("Scanning directory: " + imagesRoot.getAbsolutePath());
            List<BatchItem> items = discoverBatchItems(imagesRoot, outputRoot, maxItems);
            System.out.printf("Discovered %d face image(s)%s.%n",
                    items.size(), (maxItems > 0 ? " (limited by --max " + maxItems + ")" : ""));

            if (items.isEmpty()) {
                System.out.println("No matching images found.");
                return;
            }

            if (!obtainLicense()) {
                System.err.println("ERROR: Could not obtain FaceExtractor or FaceClient license.");
                System.exit(2);
            }

            ExecutorService pool = Executors.newFixedThreadPool(threads);
            List<NBiometricClient> createdClients = Collections.synchronizedList(new ArrayList<>());
            ThreadLocal<NBiometricClient> clientHolder = ThreadLocal.withInitial(() -> {
                NBiometricClient c = new NBiometricClient();
                c.setFacesTemplateSize(NTemplateSize.LARGE);
                c.setFacesDetectAllFeaturePoints(true);
                createdClients.add(c);
                return c;
            });

            long startTime = System.currentTimeMillis();
            AtomicInteger ok = new AtomicInteger(0);
            AtomicInteger skipped = new AtomicInteger(0);
            AtomicInteger failed = new AtomicInteger(0);
            AtomicInteger completed = new AtomicInteger(0);
            int total = items.size();

            System.out.printf("Starting in-memory face batch extraction with %d worker threads...%n", threads);

            final boolean doSkip = skipExisting;
            for (BatchItem item : items) {
                final String currentDataset = dataset;
                pool.submit(() -> {
                    try {
                        if (doSkip && item.outputFile.exists()) {
                            skipped.incrementAndGet();
                            int c = completed.incrementAndGet();
                            if (c % 100 == 0 || c == total) {
                                System.out.printf("[%5d/%5d] SKIP (already exists): %s%n", c, total, item.outputFile.getName());
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
                                currentDataset
                        );

                        int c = completed.incrementAndGet();
                        if (success) {
                            ok.incrementAndGet();
                            if (c % 25 == 0 || c == total) {
                                System.out.printf("[%5d/%5d] OK   %s%n", c, total, item.imageFile.getName());
                            }
                        } else {
                            failed.incrementAndGet();
                            System.out.printf("[%5d/%5d] FAIL %s%n", c, total, item.imageFile.getPath());
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
                pool.awaitTermination(24, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                System.err.println("Batch extraction interrupted!");
            }

            for (NBiometricClient c : createdClients) {
                try { c.dispose(); } catch (Throwable ignored) {}
            }

            long elapsedMs = System.currentTimeMillis() - startTime;
            double elapsedSec = elapsedMs / 1000.0;
            System.out.println();
            System.out.printf("Face batch extraction finished in %.2f seconds!%n", elapsedSec);
            System.out.printf("  OK:      %d%n", ok.get());
            System.out.printf("  SKIPPED: %d%n", skipped.get());
            System.out.printf("  FAILED:  %d%n", failed.get());
            System.out.printf("  TOTAL:   %d%n", total);
            System.out.println("Output written to: " + outputRoot.getAbsolutePath());
            return;
        }

        // ------------------ Single Image Mode ------------------
        if (args.length < 4) {
            System.err.println("Usage: java FaceFeatureExtractor <image> <output.json> <subject_id> <sample_id> [dataset]");
            System.exit(1);
        }

        String imagePath  = args[0];
        String outputPath = args[1];
        String subjectId  = args[2];
        String sampleId   = args[3];
        String dataset    = args.length > 4 ? args[4] : "CelebA";

        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            System.err.println("Image file not found: " + imagePath);
            System.exit(1);
        }

        if (!obtainLicense()) {
            System.err.println("ERROR: Could not obtain FaceExtractor or FaceClient license.");
            System.exit(2);
        }

        NBiometricClient client = null;
        try {
            client = new NBiometricClient();
            client.setFacesTemplateSize(NTemplateSize.LARGE);
            client.setFacesDetectAllFeaturePoints(true);

            boolean success = extractOne(
                    client,
                    imageFile,
                    new File(outputPath),
                    subjectId,
                    sampleId,
                    dataset
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
