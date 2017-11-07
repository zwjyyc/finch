This project is based on [Kyubyong's](https://github.com/Kyubyong/transformer) excellent work, thanks for his first attempt!

* Based on that, we have:
  * implemented the model under the architecture of ```tf.estimator.Estimator``` API
  * added an option to share the weights between encoder embedding and decoder embedding
  * added an option to share the weights between decoder embedding and output projection
  * added the learning rate variation according to the paper
  * added more activation choices (leaky relu / elu) for for easier gradient propagation

* Example running (the task is to learn sorting letters):
  ```  python train_letters.py --tied_embedding=1 ```
* Example running (the task is to learn chinese chatting):
  ```  python train_dialog.py ```
I found an image on internet (a kind of) illustrating how an army of attentions work ([Reference](https://techcrunch.com/2017/08/31/googles-transformer-solves-a-tricky-problem-in-machine-translation/)):
![alt text](https://github.com/zhedongzheng/finch/blob/master/assets/transform20fps.gif)
