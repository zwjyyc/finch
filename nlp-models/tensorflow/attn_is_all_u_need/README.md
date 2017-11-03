This project is based on [Kyubyong's](https://github.com/Kyubyong/transformer) excellent work, thanks for his first attempt!

Based on that, we have:
* implemented the model under the architecture of ```tf.estimator.Estimator``` API
* added an option to share the weights between encoder embedding and decoder embedding
* added an option to share the weights between decoder embedding and output projection

Example running:
>  python train.py --hidden_units=128 --num_blocks=3 --tied_proj_weight --tied_embedding --label_smoothing

I found an image on internet (a kind of) illustrating how an army of attentions work ([Reference](https://techcrunch.com/2017/08/31/googles-transformer-solves-a-tricky-problem-in-machine-translation/)):
![alt text](https://github.com/zhedongzheng/finch/blob/master/assets/transform20fps.gif)
