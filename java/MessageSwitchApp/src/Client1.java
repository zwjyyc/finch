import java.io.*;
import java.net.*;

public class Client1 {
    private static Socket clientSocket;
    
    public Client1 (String host) throws IOException {
        final int portNumber = 8888;
        try {
            // connected to server
            clientSocket = new Socket (InetAddress.getByName(host),portNumber);
        } catch (UnknownHostException uhe) {
            System.err.println("Don't know about host " + host);
            System.exit(1);
        } catch (IOException ioexc) {
            System.err.println("Couldn't get I/O for the connection to " + host);
            System.exit(1);
        } // end try...catch...catch
        System.out.println("Client 1 successfully connected to host " + host);
    } // end constructor Client1
    
    public static void main (String args[]) {
        Client1 client1;
        try {
            if (args.length == 0)
                client1 = new Client1 ("127.0.0.1");
            else
                client1 = new Client1 (args[0]);
            // two threads enable receiving and sending simultaneously
            ReceiverThread receiverThread = new ReceiverThread(clientSocket);
            SenderThread senderThread = new SenderThread(clientSocket);
            receiverThread.start();
            senderThread.start();
        } catch (IOException ioexc) {
            System.err.println("I/O Exception caught in main method of Client 1");
        }
    } // end method main
}// end class Client1
