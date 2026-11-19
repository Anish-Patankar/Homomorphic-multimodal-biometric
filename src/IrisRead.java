import com.neurotec.biometrics.NERecord;
import com.neurotec.biometrics.NETemplate;
import com.neurotec.biometrics.NTemplate;
import com.neurotec.io.NBuffer;
import com.neurotec.io.NFile;

public class IrisRead {

    public static void main(String[] args) {
        // Configure native library path for 64-bit Windows Neurotec SDK
        System.setProperty(
            "jna.library.path",
            "C:\\Program Files\\Neurotechnology\\Neurotec_Biometric_2025_2_SDK\\Bin\\Win64_x64"
        );

        // Default to templates/iris.txt if no arguments provided
        String templatePath = "templates/iris.txt";
        String exportPath = null;

        if (args.length >= 1) {
            templatePath = args[0];
        }
        if (args.length >= 2) {
            exportPath = args[1];
        }

        System.out.println("Reading template from: " + templatePath);

        NBuffer templateBuffer = null;
        NTemplate template = null;

        try {
            // Step 1: Read template file bytes into an NBuffer
            templateBuffer = NFile.readAllBytes(templatePath);
            System.out.println("File loaded successfully (" + templateBuffer.size() + " bytes).");

            // Step 2: Parse into an NTemplate container
            template = new NTemplate(templateBuffer);

            // Step 3: Retrieve the iris template (NETemplate)
            NETemplate irisesTemplate = template.getIrises();
            if (irisesTemplate == null || irisesTemplate.getRecords().isEmpty()) {
                System.out.println("No iris records found in the template.");
                return;
            }

            int recordCount = irisesTemplate.getRecords().size();
            System.out.println("\nFound " + recordCount + " iris record(s) / feature vector(s):");
            System.out.println("==================================================");

            int index = 1;
            for (NERecord record : irisesTemplate.getRecords()) {
                System.out.println("Iris Record #" + index + ":");
                System.out.println("  Position:            " + record.getPosition());
                System.out.println("  Quality:             " + (record.getQuality() & 0xFF));
                System.out.println("  Source Image Width:  " + record.getWidth() + " px");
                System.out.println("  Source Image Height: " + record.getHeight() + " px");
                System.out.println("  CBEFF Product Type:  " + record.getCBEFFProductType());
                System.out.println("  Record Size:         " + record.getSize() + " bytes");

                // Step 4: Extract the raw feature vector bytes
                // In Neurotec SDK, record.save() serializes the feature vector (NERecord)
                NBuffer recordBuffer = record.save();
                byte[] featureVectorBytes = recordBuffer.toByteArray();

                System.out.println("  Feature Vector Size: " + featureVectorBytes.length + " bytes");

                // Preview the first 32 bytes in hexadecimal
                StringBuilder hexPreview = new StringBuilder();
                int previewLength = Math.min(32, featureVectorBytes.length);
                for (int i = 0; i < previewLength; i++) {
                    hexPreview.append(String.format("%02X ", featureVectorBytes[i]));
                }
                if (featureVectorBytes.length > previewLength) {
                    hexPreview.append("...");
                }
                System.out.println("  Hex Preview (first 32 bytes): " + hexPreview.toString());

                // Optional: Export raw feature vector to a binary file
                if (exportPath != null) {
                    String outFileName = (recordCount > 1) 
                        ? exportPath + "_record" + index + ".bin" 
                        : exportPath;
                    NFile.writeAllBytes(outFileName, recordBuffer);
                    System.out.println("  Exported feature vector to: " + outFileName);
                }

                System.out.println("==================================================");
                index++;
            }

        } catch (Throwable e) {
            System.err.println("Error reading the template file: " + e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (template != null) {
                template.dispose();
            }
            if (templateBuffer != null) {
                templateBuffer.dispose();
            }
        }
    }
}
