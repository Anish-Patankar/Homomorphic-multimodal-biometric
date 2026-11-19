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
 * IrisFarFrrEvaluator
 * ====================
 * Evaluates False Acceptance Rate (FAR) and False Rejection Rate (FRR)
 * using the Neurotechnology VeriEye matching engine (IrisMatcher).
 *
 * It loads all iris templates (.bin / .json) from the features directory,
 * computes:
 *   1. Genuine match scores (intra-class: same eye, different samples)
 *   2. Impostor match scores (inter-class: different eyes)
 *   3. FAR and FRR curves across all threshold values
 *   4. Equal Error Rate (EER) and operating points (FAR=1%, 0.1%, 0.01%, 0%)
 *
 * Usage:
 *   java IrisFarFrrEvaluator [features_directory] [output_report_directory]
 *
 * Example:
 *   java IrisFarFrrEvaluator "IIT iris iriscodes" "evaluation_results"
 */
public class IrisFarFrrEvaluator {

    static class TemplateInfo {
        String eyeId;      // e.g. "001left"
        String sampleId;   // e.g. "01"
        File binFile;
        NBuffer buffer;

        TemplateInfo(String eyeId, String sampleId, File binFile, NBuffer buffer) {
            this.eyeId = eyeId;
            this.sampleId = sampleId;
            this.binFile = binFile;
            this.buffer = buffer;
        }
    }

    static class MatchPair {
        TemplateInfo probe;
        TemplateInfo gallery;
        boolean isGenuine;
        int score;

        MatchPair(TemplateInfo probe, TemplateInfo gallery, boolean isGenuine) {
            this.probe = probe;
            this.gallery = gallery;
            this.isGenuine = isGenuine;
            this.score = 0;
        }
    }

    private static boolean obtainLicense() {
        String[] licenseNames = {
            "IrisMatcher",
            "IrisClient"
        };

        for (String lic : licenseNames) {
            try {
                System.out.println("Trying license: " + lic);

                boolean obtained = NLicense.obtain("/local", 5000, lic);

                System.out.println(lic + " => " + obtained);

                if (obtained) {
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

    public static void main(String[] args) {
        System.setProperty(
            "jna.library.path",
            "C:\\Program Files\\Neurotechnology\\Neurotec_Biometric_2025_2_SDK\\Bin\\Win64_x64"
        );

        String featuresDir = args.length > 0 ? args[0] : "IIT iris iriscodes";
        String reportDir   = args.length > 1 ? args[1] : "evaluation_results";

        File inDir = new File(featuresDir);
        if (!inDir.exists() || !inDir.isDirectory()) {
            System.err.println("ERROR: Features directory not found: " + inDir.getAbsolutePath());
            System.exit(1);
        }

        File outDir = new File(reportDir);
        outDir.mkdirs();

        if (!obtainLicense()) {
            System.err.println("ERROR: Could not obtain IrisMatcher or IrisClient license.");
            System.exit(2);
        }

        try {
            // ---------------------------------------------------------------
            // 1. Discover and load all templates
            // ---------------------------------------------------------------
            System.out.println("Scanning feature templates in: " + inDir.getAbsolutePath());
            File[] subfolders = inDir.listFiles(File::isDirectory);
            if (subfolders == null || subfolders.length == 0) {
                System.err.println("No subfolders found in features directory!");
                System.exit(1);
            }
            Arrays.sort(subfolders);

            Map<String, List<TemplateInfo>> eyeTemplates = new HashMap<>();
            int totalTemplates = 0;

            for (File folder : subfolders) {
                String eyeId = folder.getName();
                File[] files = folder.listFiles((d, name) -> name.toLowerCase().endsWith(".bin"));
                if (files == null || files.length == 0) continue;
                Arrays.sort(files);

                List<TemplateInfo> list = new ArrayList<>();
                for (File f : files) {
                    try {
                        NBuffer buf = NFile.readAllBytes(f.getAbsolutePath());
                        String sampleId = f.getName().substring(0, f.getName().lastIndexOf('.'));
                        list.add(new TemplateInfo(eyeId, sampleId, f, buf));
                        totalTemplates++;
                    } catch (Throwable e) {
                        System.err.println("Warning: failed to read template " + f.getName());
                    }
                }
                if (!list.isEmpty()) {
                    eyeTemplates.put(eyeId, list);
                }
            }

            List<String> eyeIds = new ArrayList<>(eyeTemplates.keySet());
            Collections.sort(eyeIds);

            System.out.printf("Loaded %d templates across %d distinct eyes.%n", totalTemplates, eyeIds.size());
            if (eyeIds.size() < 2) {
                System.err.println("Need at least 2 distinct eyes to compute FAR/FRR.");
                System.exit(1);
            }

            // ---------------------------------------------------------------
            // 2. Generate Genuine and Impostor Pairs
            // ---------------------------------------------------------------
            List<MatchPair> genuinePairs = new ArrayList<>();
            List<MatchPair> impostorPairs = new ArrayList<>();

            // A) Genuine pairs: All pairs within the same eye (intra-class)
            for (String eyeId : eyeIds) {
                List<TemplateInfo> templates = eyeTemplates.get(eyeId);
                int count = templates.size();
                for (int i = 0; i < count; i++) {
                    for (int j = i + 1; j < count; j++) {
                        genuinePairs.add(new MatchPair(templates.get(i), templates.get(j), true));
                    }
                }
            }

            // B) Impostor pairs: Comparisons between different eyes (inter-class)
            // Compare sample 1 of every eye with sample 1 of every other eye (n * (n-1) / 2)
            int nEyes = eyeIds.size();
            for (int i = 0; i < nEyes; i++) {
                TemplateInfo tA = eyeTemplates.get(eyeIds.get(i)).get(0);
                for (int j = i + 1; j < nEyes; j++) {
                    TemplateInfo tB = eyeTemplates.get(eyeIds.get(j)).get(0);
                    impostorPairs.add(new MatchPair(tA, tB, false));
                }
            }

            System.out.printf("Formed %d GENUINE pairs and %d IMPOSTOR pairs.%n",
                    genuinePairs.size(), impostorPairs.size());

            // ---------------------------------------------------------------
            // 3. Match pairs using multi-threaded VeriEye engine
            // ---------------------------------------------------------------
            int threads = Math.min(8, Runtime.getRuntime().availableProcessors());
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            List<NBiometricClient> createdClients = Collections.synchronizedList(new ArrayList<>());
            ThreadLocal<NBiometricClient> clientHolder = ThreadLocal.withInitial(() -> {
                NBiometricClient c = new NBiometricClient();
                c.setMatchingThreshold(0); // return all match scores
                createdClients.add(c);
                return c;
            });

            System.out.printf("Matching pairs using %d worker threads...%n", threads);
            long matchStart = System.currentTimeMillis();

            // Run genuine matches
            System.out.println("Processing genuine matches...");
            AtomicInteger genCounter = new AtomicInteger(0);
            for (MatchPair pair : genuinePairs) {
                pool.submit(() -> {
                    NBiometricClient client = clientHolder.get();
                    pair.score = matchTemplates(client, pair.probe, pair.gallery);
                    int c = genCounter.incrementAndGet();
                    if (c % 1000 == 0 || c == genuinePairs.size()) {
                        System.out.printf("  Genuine progress: [%d/%d]%n", c, genuinePairs.size());
                    }
                });
            }

            // Run impostor matches
            System.out.println("Processing impostor matches...");
            AtomicInteger impCounter = new AtomicInteger(0);
            for (MatchPair pair : impostorPairs) {
                pool.submit(() -> {
                    NBiometricClient client = clientHolder.get();
                    pair.score = matchTemplates(client, pair.probe, pair.gallery);
                    int c = impCounter.incrementAndGet();
                    if (c % 10000 == 0 || c == impostorPairs.size()) {
                        System.out.printf("  Impostor progress: [%d/%d]%n", c, impostorPairs.size());
                    }
                });
            }

            pool.shutdown();
            pool.awaitTermination(2, TimeUnit.HOURS);
            for (NBiometricClient c : createdClients) {
                try { c.dispose(); } catch (Throwable ignored) {}
            }

            long matchElapsed = System.currentTimeMillis() - matchStart;
            System.out.printf("All matching completed in %.2f seconds!%n%n", matchElapsed / 1000.0);

            // ---------------------------------------------------------------
            // 4. Compute FAR, FRR, EER, and Statistics
            // ---------------------------------------------------------------
            int[] genScores = new int[genuinePairs.size()];
            for (int i = 0; i < genuinePairs.size(); i++) genScores[i] = genuinePairs.get(i).score;

            int[] impScores = new int[impostorPairs.size()];
            for (int i = 0; i < impostorPairs.size(); i++) impScores[i] = impostorPairs.get(i).score;

            Arrays.sort(genScores);
            Arrays.sort(impScores);

            int maxScore = 0;
            if (genScores.length > 0) maxScore = Math.max(maxScore, genScores[genScores.length - 1]);
            if (impScores.length > 0) maxScore = Math.max(maxScore, impScores[impScores.length - 1]);

            // Summary stats
            double genMean = average(genScores);
            double genStd  = stdDev(genScores, genMean);
            double genMedian = median(genScores);

            double impMean = average(impScores);
            double impStd  = stdDev(impScores, impMean);
            double impMedian = median(impScores);

            // Calculate FAR & FRR for every threshold T in [0, maxScore + 1]
            List<RocPoint> rocCurve = new ArrayList<>();
            double minDiff = Double.MAX_VALUE;
            double eerValue = 0.0;
            int eerThreshold = 0;

            for (int t = 0; t <= maxScore + 1; t++) {
                // FRR = count(gen < t) / totalGen
                int falseRejects = countLessThan(genScores, t);
                double frr = (double) falseRejects / genScores.length;

                // FAR = count(imp >= t) / totalImp
                int falseAccepts = countGreaterOrEqual(impScores, t);
                double far = (double) falseAccepts / impScores.length;

                rocCurve.add(new RocPoint(t, far, frr));

                double diff = Math.abs(far - frr);
                if (diff < minDiff) {
                    minDiff = diff;
                    eerThreshold = t;
                    eerValue = (far + frr) / 2.0;
                }
            }

            // Find operating points (FAR = 1%, 0.1%, 0.01%, 0.001%, ZeroFAR, ZeroFRR)
            RocPoint opFar1Pct   = findOperatingPointForFar(rocCurve, 0.01);
            RocPoint opFar01Pct  = findOperatingPointForFar(rocCurve, 0.001);
            RocPoint opFar001Pct = findOperatingPointForFar(rocCurve, 0.0001);
            RocPoint opZeroFar   = findZeroFar(rocCurve);
            RocPoint opZeroFrr   = findZeroFrr(rocCurve);

            // ---------------------------------------------------------------
            // 5. Generate Reports and Output Files
            // ---------------------------------------------------------------
            File reportFile = new File(outDir, "far_frr_report.txt");
            File rocCsvFile = new File(outDir, "roc_curve.csv");
            File genCsvFile = new File(outDir, "genuine_scores.csv");
            File impCsvFile = new File(outDir, "impostor_scores.csv");

            // Write CSV: ROC curve
            try (PrintWriter pw = new PrintWriter(new FileWriter(rocCsvFile))) {
                pw.println("Threshold,FAR,FRR,FAR_Percent,FRR_Percent,GAR_Percent");
                for (RocPoint p : rocCurve) {
                    pw.printf("%d,%.6f,%.6f,%.4f,%.4f,%.4f%n",
                            p.threshold, p.far, p.frr, p.far * 100, p.frr * 100, (1.0 - p.frr) * 100);
                }
            }

            // Write CSV: Genuine scores
            try (PrintWriter pw = new PrintWriter(new FileWriter(genCsvFile))) {
                pw.println("Eye,ProbeSample,GallerySample,Score");
                for (MatchPair p : genuinePairs) {
                    pw.printf("%s,%s,%s,%d%n", p.probe.eyeId, p.probe.sampleId, p.gallery.sampleId, p.score);
                }
            }

            // Write CSV: Impostor scores
            try (PrintWriter pw = new PrintWriter(new FileWriter(impCsvFile))) {
                pw.println("ProbeEye,ProbeSample,GalleryEye,GallerySample,Score");
                for (MatchPair p : impostorPairs) {
                    pw.printf("%s,%s,%s,%s,%d%n", p.probe.eyeId, p.probe.sampleId, p.gallery.eyeId, p.gallery.sampleId, p.score);
                }
            }

            // Write Full Text Report
            StringBuilder sb = new StringBuilder();
            sb.append("================================================================================\n");
            sb.append("           BIOMETRIC IRIS VERIFICATION PERFORMANCE REPORT (FAR / FRR)\n");
            sb.append("================================================================================\n");
            sb.append(String.format("Engine:              Neurotechnology VeriEye SDK (IrisMatcher)\n"));
            sb.append(String.format("Dataset Evaluated:   %s\n", inDir.getAbsolutePath()));
            sb.append(String.format("Total Eyes (Classes):%d\n", eyeIds.size()));
            sb.append(String.format("Total Templates:     %d\n", totalTemplates));
            sb.append(String.format("Genuine Matches:     %d comparisons\n", genuinePairs.size()));
            sb.append(String.format("Impostor Matches:    %d comparisons\n", impostorPairs.size()));
            sb.append("--------------------------------------------------------------------------------\n");
            sb.append("SCORE DISTRIBUTIONS:\n");
            sb.append(String.format("  Genuine Scores  : Min = %d, Max = %d, Mean = %.2f, Median = %.2f, StdDev = %.2f\n",
                    genScores[0], genScores[genScores.length - 1], genMean, genMedian, genStd));
            sb.append(String.format("  Impostor Scores : Min = %d, Max = %d, Mean = %.2f, Median = %.2f, StdDev = %.2f\n",
                    impScores[0], impScores[impScores.length - 1], impMean, impMedian, impStd));
            sb.append("--------------------------------------------------------------------------------\n");
            sb.append("KEY ACCURACY METRICS:\n");
            sb.append(String.format("  Equal Error Rate (EER):   %.4f %% (at Threshold = %d)\n", eerValue * 100, eerThreshold));
            sb.append(String.format("  ZeroFAR Threshold:        %d (FRR at ZeroFAR: %.4f %%)\n",
                    opZeroFar.threshold, opZeroFar.frr * 100));
            sb.append(String.format("  ZeroFRR Threshold:        %d (FAR at ZeroFRR: %.4f %%)\n",
                    opZeroFrr.threshold, opZeroFrr.far * 100));
            sb.append("--------------------------------------------------------------------------------\n");
            sb.append("OPERATING POINTS (STANDARD FAR THRESHOLDS):\n");
            sb.append("  Target FAR       | Score Threshold | Actual FAR (%) | Actual FRR (%) | GAR / TAR (%)\n");
            sb.append("  -----------------+-----------------+----------------+----------------+--------------\n");
            appendOpRow(sb, "FAR = 1.0%      ", opFar1Pct);
            appendOpRow(sb, "FAR = 0.1% (T36)", opFar01Pct);
            appendOpRow(sb, "FAR = 0.01%(T48)", opFar001Pct);
            appendOpRow(sb, "EER Point       ", new RocPoint(eerThreshold, eerValue, eerValue));
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

            // Print report to standard output
            System.out.println(sb.toString());

        } catch (Throwable e) {
            System.err.println("EXCEPTION: " + e.getMessage());
            e.printStackTrace();
            System.exit(99);
        }
    }

    private static int matchTemplates(NBiometricClient client, TemplateInfo probe, TemplateInfo gallery) {
        NSubject sProbe = null;
        NSubject sGallery = null;
        try {
            sProbe = NSubject.fromMemory(probe.buffer);
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
            if (sProbe != null) sProbe.dispose();
            if (sGallery != null) sGallery.dispose();
        }
    }

    static class RocPoint {
        int threshold;
        double far;
        double frr;

        RocPoint(int threshold, double far, double frr) {
            this.threshold = threshold;
            this.far = far;
            this.frr = frr;
        }
    }

    private static void appendOpRow(StringBuilder sb, String label, RocPoint p) {
        sb.append(String.format("  %-16s | %-15d | %-14.4f | %-14.4f | %-12.4f\n",
                label, p.threshold, p.far * 100, p.frr * 100, (1.0 - p.frr) * 100));
    }

    private static int countLessThan(int[] sortedArray, int threshold) {
        int idx = Arrays.binarySearch(sortedArray, threshold);
        if (idx < 0) {
            return -idx - 1;
        } else {
            // Find first occurrence
            while (idx > 0 && sortedArray[idx - 1] == threshold) {
                idx--;
            }
            return idx;
        }
    }

    private static int countGreaterOrEqual(int[] sortedArray, int threshold) {
        return sortedArray.length - countLessThan(sortedArray, threshold);
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
