// import java.nio.file.Files;
// import java.nio.file.Path;

import com.neurotec.biometrics.NBiometricStatus;
import com.neurotec.biometrics.NIris;
import com.neurotec.biometrics.NSubject;
import com.neurotec.biometrics.NTemplateSize;
import com.neurotec.biometrics.client.NBiometricClient;
import com.neurotec.io.NFile;
import com.neurotec.licensing.NLicense;
// import com.neurotec.licensing.NLicenseManager;

public class IrisEnroll {

    public static void main(String[] args) {
        System.setProperty(
            "jna.library.path",
            "C:\\Program Files\\Neurotechnology\\Neurotec_Biometric_2025_2_SDK\\Bin\\Win64_x64"
        );

        if (args.length != 2) {
            System.out.println(
                "Usage: java IrisEnroll <input-image> <output-template>"
            );
            System.exit(1);
        }

        String imagePath = args[0];
        String templatePath = args[1];

        NBiometricClient biometricClient = null;
        NSubject subject = null;
        NIris iris = null;

        try {
            // Enable trial mode.
            // Remove this when using a purchased license.
            // NLicenseManager.setTrialMode(true);

            // Obtain the Iris Extractor license.
            if (!NLicense.obtain("/local", 5000, "IrisExtractor")) {
                System.err.println(
                    "Could not obtain IrisExtractor license."
                );
                System.exit(-1);
            }

            // Create Neurotechnology objects.
            biometricClient = new NBiometricClient();
            subject = new NSubject();
            iris = new NIris();

            // Give VeriEye the iris image.
            iris.setFileName(imagePath);

            // Add the iris to the subject.
            subject.getIrises().add(iris);

            // Use a large iris template.
            biometricClient.setIrisesTemplateSize(
                NTemplateSize.LARGE
            );

            // Extract the biometric template.
            NBiometricStatus status =
                biometricClient.createTemplate(subject);

            if (status == NBiometricStatus.OK) {

                System.out.println("Template extracted successfully.");

                // Serialize the Neurotechnology template.
                NFile.writeAllBytes(
                    templatePath,
                    subject.getTemplate().save()
                );

                System.out.println(
                    "Template saved: " + templatePath
                );

            } else {
                System.err.println(
                    "Template extraction failed: " + status
                );
                System.exit(-1);
            }

        } catch (Throwable e) {
            e.printStackTrace();
            System.exit(-1);

        } finally {
            if (iris != null) {
                iris.dispose();
            }

            if (subject != null) {
                subject.dispose();
            }

            if (biometricClient != null) {
                biometricClient.dispose();
            }
        }
    }
}




// C:\Users\shivam\Documents\Iris\src>java -cp ".;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\neurotec-core.jar;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\neurotec-biometrics.jar;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\neurotec-biometrics-client.jar;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\neurotec-licensing.jar;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\neurotec-devices.jar;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\neurotec-media.jar;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\neurotec-media-processing.jar;C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java\jna.jar" IrisEnroll "C:\Users\shivam\Documents\Iris\images\02.bmp" "C:\Users\shivam\Documents\Iris\templates\iris.dat"
// WARNING: A restricted method in java.lang.System has been called
// WARNING: java.lang.System::load has been called by com.sun.jna.Native in an unnamed module (file:/C:/Program%20Files/Neurotechnology/Neurotec_Biometric_2025_2_SDK/Bin/Java/jna.jar)
// WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
// WARNING: Restricted methods will be blocked in a future release unless native access is enabled

// Template extracted successfully.
// Template saved: C:\Users\shivam\Documents\Iris\templates\iris.dat