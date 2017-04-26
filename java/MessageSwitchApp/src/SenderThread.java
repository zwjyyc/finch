import java.io.*;
import java.net.*;

public class SenderThread extends Thread {
    private Socket senderSocket;
    private PrintWriter out;
    
    public SenderThread (Socket socket) {
        senderSocket = socket;
    } // end constructor SenderThread
    
    public void run() {
        try {
            out = new PrintWriter(senderSocket.getOutputStream(), true);
            // use while true to always listen to keyboard input
            while (true) {
                System.out.println("Please input message: ");
                BufferedReader stdIn = new BufferedReader(new InputStreamReader(System.in));
                // send and print out the output streams
                String userInput;
                while ((userInput = stdIn.readLine()) != null) {
                    out.println(userInput);
                    System.out.println("Sent: " + userInput);
                    if (userInput.equalsIgnoreCase("quit"))
                        break;
                } // end while
                // close all streams when user types quit
                out.close();
                stdIn.close();
                senderSocket.close();
                System.out.println("Goodbye!");
                System.exit(0);
            } // end while
        } catch (IOException ioexc) {
            System.err.println("Exception caught in run method of SenderThread");
            System.exit(1);
        } // end try catch
    } // end method run
} // end class SenderThread
