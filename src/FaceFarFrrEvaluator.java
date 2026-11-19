import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import com.neurotec.biometrics.NBiometricStatus;
import com.neurotec.biometrics.NMatchingResult;
import com.neurotec.biometrics.NSubject;
import com.neurotec.biometrics.client.NBiometricClient;
import com.neurotec.io.NBuffer;
import com.neurotec.io.NFile;
import com.neurotec.licensing.NLicense;

/**
 * FaceFarFrrEvaluator
 * ====================
 * Evaluates False Acceptance Rate (FAR) and False Rejection Rate (FRR)
 * for face biometrics using the Neurotechnology VeriLook matching engine
 * (FaceMatcher / FaceClient license).
 *
 * It loads all face templates (.bin) from lfw_features (or any similarly
 * structured directory where each subfolder = one person / identity), then:
 *   1. Genuine pairs  - intra-class: different images of the same person
 *   2. Impostor pairs - inter-class: first sample of every pair of different persons
 *   3. FAR / FRR curves across all integer threshold values
 *   4. EER and standard operating points (FAR = 1%, 0.1%, 0.01%, ZeroFAR, ZeroFRR)
 *
 * Usage:
 *   java FaceFarFrrEvaluator [features_directory] [output_report_directory]
 *
 * Example:
 *   java FaceFarFrrEvaluator lfw_features face_evaluation_results
 */
public class FaceFarFrrEvaluator {

    // -------------------------------------------------------------------------
    // Data structures
    // -------------------------------------------------------------------------

    static class TemplateInfo {
        String  personId;
        String  sampleId;
        File    binFile;
        NBuffer buffer;

        TemplateInfo(String personId, String sampleId, File binFile, NBuffer buffer) {
            this.personId = personId;
            this.sampleId = sampleId;
            this.binFile  = binFile;
            this.buffer   = buffer;
        }
    }

    static class MatchPair {
        TemplateInfo probe;
        TemplateInfo gallery;
        boolean      isGenuine;
        int          score;

        MatchPair(TemplateInfo probe, TemplateInfo gallery, boolean isGenuine) {
            this.probe     = probe;
            this.gallery   = gallery;
            this.isGenuine = isGenuine;
            this.score     = 0;
        }
    }

    static class RocPoint {
        int    threshold;
        double far;
        double frr;

        RocPoint(int threshold, double far, double frr) {
            this.threshold = threshold;
            this.far       = far;
            this.frr       = frr;
        }
    }

    // -------------------------------------------------------------------------
    // License acquisition
    // -------------------------------------------------------------------------

    private static boolean obtainLicense() {
        String[] candidates = { "FaceMatcher", "FaceClient" };
        for (String lic : candidates) {
            try {
                System.out.println("Trying license: " + lic);
                boolean ok = NLicense.obtain("/local", 5000, lic);
                System.out.println(lic + " => " + ok);
                if (ok) {
                    System.out.println("Obtained license: " + lic);
                    return true;
                }
            } catch (Throwable e) {
                System.err.println("License attempt failed: " + lic);
                e.printStackTrace();
            }
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        System.setProperty(
            "jna.library.path",
            "C:\\Program Files\\Neurotechnology\\Neurotec_Biometric_2025_2_SDK\\Bin\\Win64_x64"
        );

        String featuresDir = args.length > 0 ? args[0] : "lfw_features";
        String reportDir   = args.length > 1 ? args[1] : "face_evaluation_results";

        File inDir  = new File(featuresDir);
        if (!inDir.exists() || !inDir.isDirectory()) {
            System.err.println("ERROR: Features directory not found: " + inDir.getAbsolutePath());
            System.exit(1);
        }

        File outDir = new File(reportDir);
        outDir.mkdirs();

        if (!obtainLicense()) {
            System.err.println("ERROR: Could not obtain FaceMatcher or FaceClient license.");
            System.exit(2);
        }

        try {
            // -----------------------------------------------------------------
            // 1. Discover and load all .bin face templates
            // -----------------------------------------------------------------
            System.out.println("Scanning face templates in: " + inDir.getAbsolutePath());

            File[] personFolders = inDir.listFiles(File::isDirectory);
            if (personFolders == null || personFolders.length == 0) {
                System.err.println("No person subfolders found!");
                System.exit(1);
            }
            Arrays.sort(personFolders);

            Map<String, List<TemplateInfo>> personTemplates = new HashMap<>();
            int totalTemplates = 0;

            for (File folder : personFolders) {
                String personId = folder.getName();
                File[] binFiles = folder.listFiles((d, name) -> name.toLowerCase().endsWith(".bin"));
                if (binFiles == null || binFiles.length == 0) continue;
                Arrays.sort(binFiles);

                List<TemplateInfo> list = new ArrayList<>();
                for (File f : binFiles) {
                    try {
                        NBuffer buf     = NFile.readAllBytes(f.getAbsolutePath());
                        String sampleId = f.getName().substring(0, f.getName().lastIndexOf('.'));
                        list.add(new TemplateInfo(personId, sampleId, f, buf));
                        totalTemplates++;
                    } catch (Throwable e) {
                        System.err.println("Warning: failed to read " + f.getAbsolutePath());
                    }
                }
                if (!list.isEmpty()) personTemplates.put(personId, list);
            }

            List<String> personIds = new ArrayList<>(personTemplates.keySet());
            Collections.sort(personIds);

            System.out.printf("Loaded %d templates across %d distinct persons.%n",
                    totalTemplates, personIds.size());

            if (personIds.size() < 2) {
                System.err.println("Need at least 2 distinct persons to compute FAR/FRR.");
                System.exit(1);
            }

            // -----------------------------------------------------------------
            // 2. Generate Genuine and Impostor Pairs
            // -----------------------------------------------------------------
            List<MatchPair> genuinePairs  = new ArrayList<>();
            List<MatchPair> impostorPairs = new ArrayList<>();

            // Genuine: all intra-class pairs (same person, different images)
            int personsWithMultiple = 0;
            for (String pid : personIds) {
                List<TemplateInfo> templates = personTemplates.get(pid);
                int n = templates.size();
                if (n < 2) continue;
                personsWithMultiple++;
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        genuinePairs.add(new MatchPair(templates.get(i), templates.get(j), true));
                    }
                }
            }

            // Impostor: first sample of every pair of different persons
            int nPersons = personIds.size();
            for (int i = 0; i < nPersons; i++) {
                TemplateInfo tA = personTemplates.get(personIds.get(i)).get(0);
                for (int j = i + 1; j < nPersons; j++) {
                    TemplateInfo tB = personTemplates.get(personIds.get(j)).get(0);
                    impostorPairs.add(new MatchPair(tA, tB, false));
                }
            }

            System.out.printf("Persons with >= 2 samples: %d%n", personsWithMultiple);
            System.out.printf("Formed %d GENUINE pairs and %d IMPOSTOR pairs.%n",
                    genuinePairs.size(), impostorPairs.size());

            if (genuinePairs.isEmpty()) {
                System.err.println("ERROR: No genuine pairs found. Each person needs at least 2 .bin files.");
                System.exit(1);
            }

            // -----------------------------------------------------------------
            // 3. Match pairs using multi-threaded VeriLook engine
            // -----------------------------------------------------------------
            int threads = Math.min(8, Runtime.getRuntime().availableProcessors());
            ExecutorService pool = Executors.newFixedThreadPool(threads);

            List<NBiometricClient> createdClients = Collections.synchronizedList(new ArrayList<>());
            ThreadLocal<NBiometricClient> clientHolder = ThreadLocal.withInitial(() -> {
                NBiometricClient c = new NBiometricClient();
                c.setMatchingThreshold(0);
                createdClients.add(c);
                return c;
            });

            System.out.printf("Matching pairs using %d worker threads...%n", threads);
            long matchStart = System.currentTimeMillis();

            System.out.println("Processing genuine matches...");
            AtomicInteger genCounter = new AtomicInteger(0);
            for (MatchPair pair : genuinePairs) {
                pool.submit(() -> {
                    pair.score = matchTemplates(clientHolder.get(), pair.probe, pair.gallery);
                    int c = genCounter.incrementAndGet();
                    if (c % 500 == 0 || c == genuinePairs.size()) {
                        System.out.printf("  Genuine progress: [%d/%d]%n", c, genuinePairs.size());
                    }
                });
            }

            System.out.println("Processing impostor matches...");
            AtomicInteger impCounter = new AtomicInteger(0);
            for (MatchPair pair : impostorPairs) {
                pool.submit(() -> {
                    pair.score = matchTemplates(clientHolder.get(), pair.probe, pair.gallery);
                    int c = impCounter.incrementAndGet();
                    if (c % 10000 == 0 || c == impostorPairs.size()) {
                        System.out.printf("  Impostor progress: [%d/%d]%n", c, impostorPairs.size());
                    }
                });
            }

            pool.shutdown();
            pool.awaitTermination(4, TimeUnit.HOURS);
            for (NBiometricClient c : createdClients) {
                try { c.dispose(); } catch (Throwable ignored) {}
            }

            long elapsed = System.currentTimeMillis() - matchStart;
            System.out.printf("All matching completed in %.2f seconds!%n%n", elapsed / 1000.0);

            // -----------------------------------------------------------------
            // 4. Compute FAR, FRR, EER, and Statistics
            // -----------------------------------------------------------------
            int[] genScores = new int[genuinePairs.size()];
            for (int i = 0; i < genuinePairs.size(); i++) genScores[i] = genuinePairs.get(i).score;

            int[] impScores = new int[impostorPairs.size()];
            for (int i = 0; i < impostorPairs.size(); i++) impScores[i] = impostorPairs.get(i).score;

            Arrays.sort(genScores);
            Arrays.sort(impScores);

            int maxScore = 0;
            if (genScores.length > 0) maxScore = Math.max(maxScore, genScores[genScores.length - 1]);
            if (impScores.length > 0) maxScore = Math.max(maxScore, impScores[impScores.length - 1]);

            double genMean   = average(genScores);
            double genStd    = stdDev(genScores, genMean);
            double genMedian = median(genScores);

            double impMean   = average(impScores);
            double impStd    = stdDev(impScores, impMean);
            double impMedian = median(impScores);

            List<RocPoint> rocCurve = new ArrayList<>();
            double minDiff  = Double.MAX_VALUE;
            double eerValue = 0.0;
            int    eerThresh = 0;

            for (int t = 0; t <= maxScore + 1; t++) {
                int falseRejects = countLessThan(genScores, t);
                double frr = (double) falseRejects / genScores.length;

                int falseAccepts = countGreaterOrEqual(impScores, t);
                double far = (double) falseAccepts / impScores.length;

                rocCurve.add(new RocPoint(t, far, frr));

                double diff = Math.abs(far - frr);
                if (diff < minDiff) {
                    minDiff   = diff;
                    eerThresh = t;
                    eerValue  = (far + frr) / 2.0;
                }
            }

            RocPoint opFar1Pct   = findOperatingPointForFar(rocCurve, 0.01);
            RocPoint opFar01Pct  = findOperatingPointForFar(rocCurve, 0.001);
            RocPoint opFar001Pct = findOperatingPointForFar(rocCurve, 0.0001);
            RocPoint opZeroFar   = findZeroFar(rocCurve);
            RocPoint opZeroFrr   = findZeroFrr(rocCurve);

            // -----------------------------------------------------------------
            // 5. Write output files
            // -----------------------------------------------------------------
            File reportFile = new File(outDir, "face_far_frr_report.txt");
            File rocCsvFile = new File(outDir, "face_roc_curve.csv");
            File genCsvFile = new File(outDir, "face_genuine_scores.csv");
            File impCsvFile = new File(outDir, "face_impostor_scores.csv");

            try (PrintWriter pw = new PrintWriter(new FileWriter(rocCsvFile))) {
                pw.println("Threshold,FAR,FRR,FAR_Percent,FRR_Percent,GAR_Percent");
                for (RocPoint p : rocCurve) {
                    pw.printf("%d,%.6f,%.6f,%.4f,%.4f,%.4f%n",
                            p.threshold, p.far, p.frr,
                            p.far * 100, p.frr * 100, (1.0 - p.frr) * 100);
                }
            }

            try (PrintWriter pw = new PrintWriter(new FileWriter(genCsvFile))) {
                pw.println("Person,ProbeSample,GallerySample,Score");
                for (MatchPair p : genuinePairs) {
                    pw.printf("%s,%s,%s,%d%n",
                            p.probe.personId, p.probe.sampleId, p.gallery.sampleId, p.score);
                }
            }

            try (PrintWriter pw = new PrintWriter(new FileWriter(impCsvFile))) {
                pw.println("ProbePerson,ProbeSample,GalleryPerson,GallerySample,Score");
                for (MatchPair p : impostorPairs) {
                    pw.printf("%s,%s,%s,%s,%d%n",
                            p.probe.personId, p.probe.sampleId,
                            p.gallery.personId, p.gallery.sampleId,
                            p.score);
                }
            }

            StringBuilder sb = new StringBuilder();
            sb.append("================================================================================\n");
            sb.append("        BIOMETRIC FACE VERIFICATION PERFORMANCE REPORT (FAR / FRR)\n");
            sb.append("================================================================================\n");
            sb.append(String.format("Engine:              Neurotechnology VeriLook SDK (FaceMatcher)\n"));
            sb.append(String.format("Dataset Evaluated:   %s\n", inDir.getAbsolutePath()));
            sb.append(String.format("Total Persons:       %d\n", personIds.size()));
            sb.append(String.format("Total Templates:     %d\n", totalTemplates));
            sb.append(String.format("Genuine Matches:     %d comparisons\n", genuinePairs.size()));
            sb.append(String.format("Impostor Matches:    %d comparisons\n", impostorPairs.size()));
            sb.append("--------------------------------------------------------------------------------\n");
            sb.append("SCORE DISTRIBUTIONS:\n");
            sb.append(String.format(
                "  Genuine Scores  : Min = %d, Max = %d, Mean = %.2f, Median = %.2f, StdDev = %.2f\n",
                genScores[0], genScores[genScores.length - 1], genMean, genMedian, genStd));
            sb.append(String.format(
                "  Impostor Scores : Min = %d, Max = %d, Mean = %.2f, Median = %.2f, StdDev = %.2f\n",
                impScores[0], impScores[impScores.length - 1], impMean, impMedian, impStd));
            sb.append("--------------------------------------------------------------------------------\n");
            sb.append("KEY ACCURACY METRICS:\n");
            sb.append(String.format(
                "  Equal Error Rate (EER):   %.4f %% (at Threshold = %d)\n",
                eerValue * 100, eerThresh));
            sb.append(String.format(
                "  ZeroFAR Threshold:        %d (FRR at ZeroFAR: %.4f %%)\n",
                opZeroFar.threshold, opZeroFar.frr * 100));
            sb.append(String.format(
                "  ZeroFRR Threshold:        %d (FAR at ZeroFRR: %.4f %%)\n",
                opZeroFrr.threshold, opZeroFrr.far * 100));
            sb.append("--------------------------------------------------------------------------------\n");
            sb.append("OPERATING POINTS (STANDARD FAR THRESHOLDS):\n");
            sb.append("  Target FAR       | Score Threshold | Actual FAR (%) | Actual FRR (%) | GAR / TAR (%)\n");
            sb.append("  -----------------+-----------------+----------------+----------------+--------------\n");
            appendOpRow(sb, "FAR = 1.0%      ", opFar1Pct);
            appendOpRow(sb, "FAR = 0.1%      ", opFar01Pct);
            appendOpRow(sb, "FAR = 0.01%     ", opFar001Pct);
            appendOpRow(sb, "EER Point       ", new RocPoint(eerThresh, eerValue, eerValue));
            appendOpRow(sb, "ZeroFAR (0%)    ", opZeroFar);
            appendOpRow(sb, "ZeroFRR (0%)    ", opZeroFrr);
            sb.append("================================================================================\n");
            sb.append("Output Files Generated:\n");
            sb.append("  - " + reportFile.getAbsolutePath() + "\n");
            sb.append("  - " + rocCsvFile.getAbsolutePath() + "\n");
            sb.append("  - " + genCsvFile.getAbsolutePath() + "\n");
            sb.append("  - " + impCsvFile.getAbsolutePath() + "\n");
            sb.append("================================================================================\n");

            try (PrintWriter pw = new PrintWriter(new FileWriter(reportFile))) {
                pw.print(sb.toString());
            }
            System.out.println(sb.toString());

        } catch (Throwable e) {
            System.err.println("EXCEPTION: " + e.getMessage());
            e.printStackTrace();
            System.exit(99);
        }
    }

    // -------------------------------------------------------------------------
    // VeriLook pair matching
    // -------------------------------------------------------------------------

    private static int matchTemplates(NBiometricClient client,
                                      TemplateInfo probe,
                                      TemplateInfo gallery) {
        NSubject sProbe   = null;
        NSubject sGallery = null;
        try {
            sProbe   = NSubject.fromMemory(probe.buffer);
            sGallery = NSubject.fromMemory(gallery.buffer);
            NBiometricStatus status = client.verify(sProbe, sGallery);
            if (status == NBiometricStatus.OK) {
                List<NMatchingResult> results = sProbe.getMatchingResults();
                if (results != null && !results.isEmpty()) {
                    return results.get(0).getScore();
                }
            }
            return 0;
        } catch (Throwable t) {
            return 0;
        } finally {
            if (sProbe   != null) sProbe.dispose();
            if (sGallery != null) sGallery.dispose();
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static void appendOpRow(StringBuilder sb, String label, RocPoint p) {
        sb.append(String.format("  %-16s | %-15d | %-14.4f | %-14.4f | %-12.4f\n",
                label, p.threshold, p.far * 100, p.frr * 100, (1.0 - p.frr) * 100));
    }

    private static int countLessThan(int[] sorted, int threshold) {
        int idx = Arrays.binarySearch(sorted, threshold);
        if (idx < 0) return -idx - 1;
        while (idx > 0 && sorted[idx - 1] == threshold) idx--;
        return idx;
    }

    private static int countGreaterOrEqual(int[] sorted, int threshold) {
        return sorted.length - countLessThan(sorted, threshold);
    }

    private static RocPoint findOperatingPointForFar(List<RocPoint> roc, double targetFar) {
        for (RocPoint p : roc) {
            if (p.far <= targetFar) return p;
        }
        return roc.get(roc.size() - 1);
    }

    private static RocPoint findZeroFar(List<RocPoint> roc) {
        for (RocPoint p : roc) {
            if (p.far == 0.0) return p;
        }
        return roc.get(roc.size() - 1);
    }

    private static RocPoint findZeroFrr(List<RocPoint> roc) {
        for (int i = roc.size() - 1; i >= 0; i--) {
            if (roc.get(i).frr == 0.0) return roc.get(i);
        }
        return roc.get(0);
    }

    private static double average(int[] arr) {
        if (arr.length == 0) return 0.0;
        double sum = 0.0;
        for (int v : arr) sum += v;
        return sum / arr.length;
    }

    private static double median(int[] sortedArr) {
        if (sortedArr.length == 0) return 0.0;
        int mid = sortedArr.length / 2;
        if (sortedArr.length % 2 == 1) return sortedArr[mid];
        return (sortedArr[mid - 1] + sortedArr[mid]) / 2.0;
    }

    private static double stdDev(int[] arr, double mean) {
        if (arr.length <= 1) return 0.0;
        double sum = 0.0;
        for (int v : arr) {
            double d = v - mean;
            sum += d * d;
        }
        return Math.sqrt(sum / (arr.length - 1));
    }
}
