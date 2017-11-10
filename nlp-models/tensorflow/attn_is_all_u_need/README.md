This project is based on [Kyubyong's](https://github.com/Kyubyong/transformer) excellent work, thanks for his first attempt!

* Based on that, we have:
  * implemented the model under the architecture of ```tf.estimator.Estimator``` API
  * added an option to share the weights between encoder embedding and decoder embedding
  * added an option to share the weights between decoder embedding and output projection
  * added the learning rate variation according to the formula in paper
  * added more activation choices (leaky relu / elu) for easier gradient propagation
  * generated the key and query masks before positional encoding
    * because after positional encoding, the paddings in key and query will move from zero to other values

* Small Task: learn sorting characters
    * ``` python train_letters.py --hidden_units=128 --num_block=1 --tied_embedding=1 --lr_decay=exp ```

    * greedy sampling after 3128 steps
        ```
        apple -> aelpp<end><end><end>
        common -> cmmnoo<end><end>
        zhedong -> deghnoz<end>
        ```

* I found an image on internet (a kind of) illustrating how an army of attentions work ([Reference](https://techcrunch.com/2017/08/31/googles-transformer-solves-a-tricky-problem-in-machine-translation/)):
![alt text](https://github.com/zhedongzheng/finch/blob/master/assets/transform20fps.gif)
