## What is this project?
[PDF](https://github.com/zhedongzheng/finch/blob/master/java/MessageSwitchApp/problem-description.pdf)

## Run this project

Step 1: run `MessageSwitchMT.java` in console 1, you will see:  
> Server listening on port: 8888  

Step 2: run `Client1.java` in console 2, you will see:
> Client 1 successfully connected to host 127.0.0.1  
> Please input message:  

Step 3: run `Client2.java` in console 3, you will see:  
> Client 2 successfully connected to host 127.0.0.1  
> Please input message:  

Now, you can type a message in `client 1 (console 2)` or `client 2 (console 3)`, and receive that message in the other one 

`MessageSwitchMT` is the server which should not be terminated unless you want to exit
