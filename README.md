# NLP-LUNA
This repository implements the decoder-only [LUNA](https://arxiv.org/abs/2106.01540) architecture in Tensorflow v2. Please note that work is still on-going.

# Processing the Data
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

