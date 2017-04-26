### Problem Description
[PDF](https://github.com/zhedongzheng/finch/blob/master/java/MessageSwitchApp/problem-description.pdf)

### Command line Op
* compile all .java
  ```
  javac *.java
  ```
* console 1 types:
  ```
  java MessageSwitchMT
  ```
  displays:  
  > Server listening on port: 8888
  
* console 2 types:
  ```
  java Client1
  ```
  displays:
  > Client 1 successfully connected to host 127.0.0.1  
  > Please input message: 
  
* console 3 types
  ```
  java Client2
  ```
  displays:
  > Client 2 successfully connected to host 127.0.0.1  
  > Please input message:  
  
* Now, you can type a message in `console 2` or `console 3`, and receive the message in the other console 
* `MessageSwitchMT` is the server which should not be terminated unless you want to exit the program
