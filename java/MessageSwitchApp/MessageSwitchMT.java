import java.io.*;
import java.net.*;

public class MessageSwitchMT {
    private ServerSocket serverSocket;
    private static Socket clientSocket1;
    private static Socket clientSocket2;
    
    public MessageSwitchMT (String host) throws IOException {
        final int portNumber = 8888;
        
        // set up server
        try {
            serverSocket = new ServerSocket (portNumber);
        } catch (IOException ioexc) {
            System.err.println("Exception caught when trying to listen on port " + portNumber);
            System.exit(1);
        } // end try catch
        System.out.println("Server listening on port: " + portNumber);
        
        // accept client 1 connection
        try {
            clientSocket1 = serverSocket.accept();
            System.out.println("Successfully connected to client 1: " + clientSocket1.getInetAddress().getHostName());
        } catch (IOException ioexc) {
            System.err.println("Failed to accept receiver socket for port: " + portNumber);
            System.exit(1);
        } // end try catch
        
        // accept client 2 connection
        try {
            clientSocket2 = serverSocket.accept();
            System.out.println("Successfully connected to client 2: " + clientSocket2.getInetAddress().getHostName());
        } catch (IOException ioexc) {
            System.err.println("Failed to accept sender socket for port: " + portNumber);
            System.exit(1);
        } // end try catch
    } // end constructor MessageSwitchMT
    
    public static void main (String args[]) {
        MessageSwitchMT messageSwitchMT;
        try {
            if (args.length == 0)
                messageSwitchMT = new MessageSwitchMT("127.0.0.1");
            else
                messageSwitchMT = new MessageSwitchMT(args[0]);
            // two threads to handle bidirectional input
            MessageSwitchThread thread1 = new MessageSwitchThread(clientSocket1,clientSocket2);
            MessageSwitchThread thread2 = new MessageSwitchThread(clientSocket2,clientSocket1);
            thread1.start();
            thread2.start();
        } catch (IOException ioexc) {
            System.err.println("I/O Exception caught in main method");
        } // end try catch
    } // end method main
} // end class MessageSwitchMT
