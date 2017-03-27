import java.io.FileReader;
import java.util.List;
import java.util.Arrays;
import com.opencsv.CSVReader;


public class LogisticRegressionTest {
    public static void main (String args[]) {
        try {
            // Parse CSV file
            CSVReader reader = new CSVReader(new FileReader("./data/data.csv"));
            String[] header = reader.readNext();
            List<String[]> lines = reader.readAll();
            reader.close();

            // Get X and y
            double[][] inputMatrix = M.parseDoubleMatrix(lines, 0, M.searchHeader(header, "target")+1);
            double[][] X = M.subMatrix(inputMatrix, 0, M.searchHeader(header, "target"));
            double[] y = M.getColVal(inputMatrix, M.searchHeader(header, "target"));

            // Split
            double[][] X_train = Arrays.copyOfRange(X, 0, 70);
            double[][] X_test = Arrays.copyOfRange(X, 70, 100);
            double[] y_train = Arrays.copyOfRange(y, 0, 70);
            double[] y_test = Arrays.copyOfRange(y, 70, 100);
            System.out.println("X_train: (" + X_train.length + ", " + X_train[0].length + ")");
            System.out.println("X_test: (" + X_test.length + ", " + X_test[0].length + ")");
            System.out.println("y_train: (" + y_train.length + ")");
            System.out.println("y_test: (" + y_test.length + ")");
            
            // Apply machine learning
            //LogisticRegression classifier = new LogisticRegression();
            //classifier.fit(X, y);
            //double[] prediction = classifier.predict(X);
        } catch (Exception e) {
            System.out.println(e);
        } // end try catch
    } // end main
} // end class
