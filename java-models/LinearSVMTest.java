import java.io.FileReader;
import java.util.List;
import com.opencsv.CSVReader;


public class LinearSVMTest {
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
            double[] y_ = M.getColVal(inputMatrix, M.searchHeader(header, "target"));
            for (int i=0; i<y_.length; i++) {
                if (y_[i] == 0.0)
                    y_[i] = -1.0;
            }
            
            // Apply machine learning
            LinearSVM classifier = new LinearSVM();
            classifier.fit(X, y_);
            double[] y_pred = classifier.predict(X);
        } catch (Exception e) {
            System.out.println(e);
        } // end try catch
    } // end main
} // end class
