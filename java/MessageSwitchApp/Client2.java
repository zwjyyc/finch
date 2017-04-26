import java.io.*;
import java.net.*;

public class Client2 {
    private static Socket clientSocket;
    
    public Client2 (String host) throws IOException {
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
        System.out.println("Client 2 successfully connected to host " + host);
    } // end constructor Client2
    
    public static void main (String args[]) {
        Client2 client2;
        try {
            if (args.length == 0)
                client2 = new Client2 ("127.0.0.1");
            else
                client2 = new Client2 (args[0]);
            // two threads enable sending and receiving simultaneously
            SenderThread senderThread = new SenderThread(clientSocket);
            ReceiverThread receiverThread = new ReceiverThread(clientSocket);
            senderThread.start();
            receiverThread.start();
        } catch (IOException ioexc) {
            System.err.println("I/O Exception caught in main method of Client 2");
        }
    } // end method main
}// end class Client2
