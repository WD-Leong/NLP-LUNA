# NLP-LUNA
This repository implements the decoder-only [LUNA](https://arxiv.org/abs/2106.01540) architecture in Tensorflow v2. Please note that work is still on-going.

## Processing the Data
To train the chatbot based on the [Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset, run
```
python process_movie_dialogue_subword.py
```
to process the corpus into sub-word tokens. The processing code follows this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely.

# Training and Inference
To train the model, run
```
python train_movie_dialogue_sw_tf_ver2_gpt_luna.py
```
followed by
```
python infer_movie_dialogue_sw_tf_ver2_gpt_luna.py
```
to perform inference using the trained model.

## Extending to Long Sequences
To allow the model to be trained on long sequences with modest compute resources, `tf_ver2_gpt_luna_v1.py` was introduced. Using ideas from [Transformers are RNNs](https://arxiv.org/abs/2006.16236), the long sequences is divided into windows, each of size `window_len`, and the state of the prefix sums of the preceding window is fed into the next window. This maintains the training parallelism of Transformers across windows while making the different windows auto-regressive during training. 

Training then proceeds by accumulating the gradients of the loss function on each window across all windows. Inference is made more efficient by making the inference auto-regressive based on its preceding time step. Please note that the formulation has not been thoroughly checked to ensure that it is correct.

To train the model, run
```
python train_movie_dialogue_sw_tf_ver2_gpt_luna_v1.py
```
followed by
```
python infer_movie_dialogue_sw_tf_ver2_gpt_luna_v1.py
```
to perform inference using the trained model.
