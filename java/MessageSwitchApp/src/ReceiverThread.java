import java.io.*;
import java.net.*;

public class ReceiverThread extends Thread {
    private Socket receiverSocket;
    private BufferedReader in;
    
    public ReceiverThread (Socket socket) {
        receiverSocket = socket;
    } // end constructor ReceiverThread
    
    public void run() {
        try {
            in = new BufferedReader(new InputStreamReader(receiverSocket.getInputStream()));
            // receive and print out the input streams
            String userInput;
            while ((userInput = in.readLine()) != null) {
                if (userInput.equalsIgnoreCase("quit"))
                    break;
                System.out.println("Received: " + userInput);
            } // end while
            // close all streams when user types quit
            in.close();
            receiverSocket.close();
            System.out.println("Goodbye!");
            System.exit(0);
        } catch (IOException ioexc) {
            System.err.println("Exception caught in run method of ReceiverThread");
            System.exit(1);
        }
    } // end method run
} // end class ReceiverThread
