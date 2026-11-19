import com.neurotec.biometrics.NBiometricStatus;
import com.neurotec.biometrics.NEAttributes;
import com.neurotec.biometrics.NIris;
import com.neurotec.biometrics.NSubject;
import com.neurotec.biometrics.NTemplateSize;
import com.neurotec.biometrics.client.NBiometricClient;
import com.neurotec.licensing.NLicense;

public class TestSegmentation {

    public static void main(String[] args) {

        System.setProperty(
            "jna.library.path",
            "C:\\Program Files\\Neurotechnology\\Neurotec_Biometric_2025_2_SDK\\Bin\\Win64_x64"
        );

        if (args.length != 1) {
            System.out.println("Usage: java TestSegmentation <iris-image>");
            System.exit(1);
        }

        String imagePath = args[0];

        NBiometricClient client = null;
        NSubject subject = null;
        NIris iris = null;

        try {
            if (!NLicense.obtain("/local", 5000, "IrisExtractor")) {
                System.err.println("Could not obtain IrisExtractor license.");
                System.exit(-1);
            }

            client = new NBiometricClient();
            subject = new NSubject();
            iris = new NIris();

            iris.setFileName(imagePath);
            subject.getIrises().add(iris);

            client.setIrisesTemplateSize(NTemplateSize.LARGE);

            NBiometricStatus status = client.createTemplate(subject);

            System.out.println("Status: " + status);

            if (status != NBiometricStatus.OK) {
                return;
            }

            System.out.println(
                "Number of iris objects: " +
                iris.getObjects().size()
            );

            for (int i = 0; i < iris.getObjects().size(); i++) {

                NEAttributes attr = iris.getObjects().get(i);

                System.out.println("\n=== Iris Object " + i + " ===");

                System.out.println(
                    "Position: " + attr.getPosition()
                );

                System.out.println(
                    "Detected position: " +
                    attr.getDetectedPosition()
                );

                System.out.println(
                    "Inner boundary available: " +
                    attr.isInnerBoundaryAvailable()
                );

                System.out.println(
                    "Outer boundary available: " +
                    attr.isOuterBoundaryAvailable()
                );

                System.out.println(
                    "Iris radius: " +
                    attr.getIrisRadius()
                );

                System.out.println(
                    "Rotation angle: " +
                    attr.getRotationAngle()
                );

                System.out.println(
                    "Inner boundary point count: " +
                    attr.getInnerBoundaryPoints().size()
                );

                System.out.println(
                    "Outer boundary point count: " +
                    attr.getOuterBoundaryPoints().size()
                );

                int innerCount =
                    attr.getInnerBoundaryPoints().size();

                System.out.println("\nFirst inner boundary points:");

                for (int j = 0; j < innerCount; j++) {
                    System.out.println(
                        attr.getInnerBoundaryPoints().get(j)
                    );
                }

                int outerCount =
                    attr.getOuterBoundaryPoints().size();

                System.out.println("\nFirst outer boundary points:");

                for (int j = 0; j < outerCount; j++) {
                    System.out.println(
                        attr.getOuterBoundaryPoints().get(j)
                    );
                }
            }

        } catch (Throwable e) {
            e.printStackTrace();

        } finally {
            if (iris != null) iris.dispose();
            if (subject != null) subject.dispose();
            if (client != null) client.dispose();
        }
    }
}