import java.io.*;
import java.net.*;

public class MessageSwitchThread extends Thread {
    private Socket inputSocket;
    private Socket outputSocket;
    private BufferedReader input;
    private PrintWriter output;
    
    public MessageSwitchThread (Socket senderSocket, Socket receiverSocket) throws IOException {
        inputSocket = senderSocket;
        outputSocket = receiverSocket;
        input = new BufferedReader(new InputStreamReader(inputSocket.getInputStream()));
        output = new PrintWriter(outputSocket.getOutputStream(), true);
    } // end constructor MessageSwitchThread
    
    public void run() {
        try {
            // forward message from input stream to output stream
            String inputLine;
            while ((inputLine = input.readLine()) != null) {
                System.out.println("Received: " + inputLine);
                if (inputLine.equalsIgnoreCase("quit"))
                    break;
                output.println(inputLine);
                System.out.println("Sent: " + inputLine);
            } // end while
            // close all streams only when user types quit
            input.close();
            output.close();
            inputSocket.close();
            outputSocket.close();
            System.out.println("Goodbye!");
            System.exit(0);
        } catch (IOException ioexc) {
            System.err.println("I/O Exception caught in method run of MessageSwitchThread");
            System.exit(1);
        } // end try catch
    } // end method run
} // end class MessageSwitchThread
