This project is based on [Kyubyong's](https://github.com/Kyubyong/transformer) excellent work, thanks for his first attempt!

* Based on that, we have:
  * implemented the model under the architecture of ```tf.estimator.Estimator``` API
  * added an option to share the weights between encoder embedding and decoder embedding
  * added an option to share the weights between decoder embedding and output projection
  * added the learning rate variation according to the formula in paper, and also expotential decay
  * added more activation choices (leaky relu / elu) for easier gradient propagation
  * generated the key and query masks before positional encoding
    * because after positional encoding, the paddings in key and query will move from zeros to other values

* Small Task: learn sorting characters
    * ``` python train_letters.py```
    * greedy sampling after 3128 steps
        ```
        apple -> aelpp<end><end><end>
        common -> cmmnoo<end><end>
        zhedong -> deghnoz<end>
        ```
* Small Task: learn chinese chatting
    * ``` python train_letters.py```
    * greedy sampling after 2000 steps
        ```
        你是谁 -> 我是小通<end><end><end><end><end><end>
        你喜欢我吗 -> 我喜欢你喜欢你<end><end><end>
        给我唱一首歌 -> =。=<end>=<end>=<end>=<end>
        我帅吗 -> =。=<end>=<end>=<end>=<end>
        ```

* I found an image on internet (a kind of) illustrating how an army of attentions work ([Reference](https://techcrunch.com/2017/08/31/googles-transformer-solves-a-tricky-problem-in-machine-translation/)):
![alt text](https://github.com/zhedongzheng/finch/blob/master/assets/transform20fps.gif)
