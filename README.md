# Video to Language Challenge (MSR-VTT Challenge 2016)



Team Fudan-ILC (Fudan University & Intel Labs China) solution for the MSR-VTT Challenge (http://ms-multimedia-challenge.com/2016/challenge).

This repository contains the code for the original implementation in the challenge and our updated version.

The original version was ranked 4th in human evaluation and ranked 5th in the automatic evaluation metrics based [Leaderboard](http://ms-multimedia-challenge.com/2016/leaderboard) 
**(Note that we only used a single model (no ensemble) and did not use any category or audio information provided by the dataset in this version.)**


**The updated version is part of the [DenseVidCap](https://arxiv.org/abs/1704.01502) project that combines the audio features, category information and other techniques, which achieves higher scores.**

This code is based on NeuralTalk2 (https://github.com/karpathy/neuraltalk2).

## Table of Contents
1. [Usage](#usage)
2. [Train challenge models](#Train-original-model)
3. [Test challenge models](#Test-original-model)
4. [Challenge results](#Original-results)
5. [Train updated models](#Train-updated-model)
6. [Updated results](#Updated-results)
7. [Two improvements for NeuralTalk2](#improvements)
8. [Contact](#contact)

## Usage 
0. Clone our repository:

  ```shell
  git clone https://github.com/szq0214/MSR-VTT-Challenge.git
  ```
  
1. Our requirements are similar as [NeuralTalk2](https://github.com/karpathy/neuraltalk2). You can follow the instructions there or do the following instructions.
  
### Requirements (modified from [NeuralTalk2](https://github.com/karpathy/neuraltalk2))

This code is written in Lua and requires [Torch](http://torch.ch/). If you're on Ubuntu, installing Torch in your home directory may look something like: 

```bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
```

See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

```bash
$ luarocks install nn
$ luarocks install nngraph 
$ luarocks install image 
```

We're also going to need the [cjson](http://www.kyne.com.au/~mark/software/lua-cjson-manual.html) library so that we can load/save json files. Follow their [download link](http://www.kyne.com.au/~mark/software/lua-cjson.php) and then look under their section 2.4 for easy luarocks install.

If you'd like to run on an NVIDIA GPU using CUDA (which you really, really want to if you're training a model, since we're using a VGGNet), you'll of course need a GPU, and you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then get the `cutorch` and `cunn` packages:

```bash
$ luarocks install cutorch
$ luarocks install cunn
```

If you'd like to use the cudnn backend (the pretrained checkpoint does), you also have to install [cudnn](https://github.com/soumith/cudnn.torch). First follow the link to [NVIDIA website](https://developer.nvidia.com/cuDNN), register with them and download the cudnn library. Then make sure you adjust your `LD_LIBRARY_PATH` to point to the `lib64` folder that contains the library (e.g. `libcudnn.so.7.0.64`). Then git clone the `cudnn.torch` repo, `cd` inside and do `luarocks make cudnn-scm-1.rockspec` to build the Torch bindings.

Finally, you will also need to install [torch-hdf5](https://github.com/deepmind/torch-hdf5), and [h5py](http://www.h5py.org/), since we will be using hdf5 files to store the preprocessed data.

### Other preparations

3. Download training and validation [features](https://drive.google.com/open?id=0B4cvsEOB5eUCckFvOU8zb0RVWTg) (including the updated version) and put them into `root` directory of this project. You can also download our pre-trained [model](https://drive.google.com/open?id=0B4cvsEOB5eUCZU5UQ01vUnU4RUE) here which is used in the challenge.
4. Download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.
5. Download the MSR-VTT [annotations](https://drive.google.com/open?id=0B4cvsEOB5eUCM0Q2TFdvX0ZjLWc) into `coco-caption/annotations` directory.


## Train challenge models
1. Set training data, json file and options in `mean_pool_MSR_VTT_train.lua` file. Use `-input_h5 data_augmentation.h5`, `-input_json data_augmentation.json`, `-rnn_size 1024`, `-input_encoding_size 1024`,  `-update_iter 4`, `-learning_rate 4e-4`, `-beam_size 1`, `-train_split train`, `-eval_split val`, `-val_images_use 497`, `-save_checkpoint_every 250`, `-language_eval 1` option. options
2. Set feature dimension `-input_feature_dim 27648 `.

 >The feature we used includs: 1) ResNet (2048); 2) VGG19 (4096); 3) Places-VGG16 (4096); 4) Places-GoogLeNet (1024); 5) EventNet (4096); 6) C3D\_2 (4096); 7) C3D\_8 (4096); 8) C3D\_16 (4096). The total feature dimension is 27648.

 >Data augmentation (segment video clip):(1) mean-pooling over all frames from each clip (100%); (2) over first 25% frames; (3) over first 50% frames(4) over first 75% frames; (5) over last 50% frames.

3. Train a model using

  ```shell
  th mean_pool_MSR_VTT_train.lua -gpuid 0 | tee MSR_VTT_challenge.log
  ```

## Test challenge models
1. Set test data, json file and options in `eval_MSR_VTT.lua` file. Use 
`-input_feature_dim 27648 `, `-model model_id_MSR_VTT_challenge.t7`, `-num_images 2990`, `-language_eval 1`, `-input_h5 data_augmentation_test.h5`, `-input_json data_augmentation_test.json`, `-beam_size 2`, `-split val` options

2. Test the model using

  ```shell
  th eval_MSR_VTT.lua -gpuid 0
  ```


## Challenge results
The validation score curve during training (Score = BLEU@4 + METEOR + CIDEr + ROUGE-L):

<img src="https://cloud.githubusercontent.com/assets/3794909/20632164/7a02e716-b376-11e6-8aa5-30e793527820.png" width="480">


The tables below show the results of Fudan-ILC on MSR-VTT challenge.

M1 performance:

Team | BLEU@4 | METEOR | CIDEr | ROUGE-L 
-------|:--------:|:--------:|:--------:|:--------:|
Fudan-ILC (validation set) |39.0 |27.7 | 44.0|60.1
Fudan-ILC (test set)|38.7 |26.8 | 41.9|59.5

M2 performance:

Team | C1 | C2 | C3
-------|:--------:|:--------:|:--------:|
Fudan-ILC (test set)|3.185 |2.999 | 2.979


## Train updated models
1. Train language models with visual features (category-wise manner)

  * Train language models for category_X (replace X below with 0,1,...,19 to train 20 category-wise models)
  
     1) Set the training data, json file and options. Use `-input_feature_dim 11264`, `-input_h5 data_lexical.h5`, `-input_json data_lexical.json`, `-rnn_size 512`  `-update_iter 1`, `-learning_rate 2e-4`, `-beam_size 1`, `-train_split train_X`, `-eval_split val_X`, `-val_images_use XXX`, `-save_checkpoint_every 100`, `-checkpoint_path lexical`, `-language_eval 1`, `-id _lexical_X` options
    >If you want to evaluate the whole validation set, please make `-val_images_use` larger than the number of examples in each category. For convenience, you can set it with a large number like 1000 for all categories.
  
     >In the challenge model, we use two linear layers to embed the input features, while for efficiency, we apply single layer in the updated model. The embedding parameters are learned jointly with the language model. You can modify line 24~31 in `misc/net_utils.lua` to

       >cnn_part:add(nn.Linear(opt.input_feature_dim, encoding_size, true))
  
       >cnn_part:add(backend.ReLU(true))
  
       >cnn_part:add(nn.Dropout(p2))
  
     2) Train a model using
  
     ```shell
     th mean_pool_MSR_VTT_train.lua -gpuid 0 | tee lexical/lexical_X.log
     ```

  * Calculate the final results: Since each category has different numbers of videos, we can not simply average all best performance scores of all categories. We need to collect all generated best-sentences into a single file from lexical_X.log files (You can search 't7' to find out the best sentences (with highest scores) for collecting) like follows:

       >image video6791: a man is talking about something	
evaluating validation performance... 1/71 (6.718832)	
image video6935: a man is talking to a camera for a video game	
evaluating validation performance... 2/71 (4.730399)	
image video6697: a man in a black shirt is playing tennis	
evaluating validation performance... 3/71 (3.754927)	
image video6929: a man is running on the field	
evaluating validation performance... 4/71 (4.739570)	
image video6629: a person is playing a video game	
evaluating validation performance... 5/71 (3.456113)	
......

      Then you can evaluate the final results with
      
       ```shell
       python eval_category_wise.py --res_file your_res_file_path
       ```

2. Visual features + C3D_16:

  Set options `-input_feature_dim 15360`, `-input_h5 data_lexical_C3D_16.h5`, `-input_json data_lexical_C3D_16.json` and follow steps above.

3. Visual features + C3D\_16 + C3D\_2:

  Set options `-input_feature_dim 19456`, `-input_h5 data_lexical_C3D.h5`, `-input_json data_lexical_C3D.json` and follow steps above.

4. Visual features + C3D\_16 + C3D\_2 + BoAW:

  Set options `-input_feature_dim 19556`, `-input_h5 data_lexical_C3D_audio.h5`, `-input_json data_lexical_C3D_audio.json` and follow steps above.

## Updated results
Performance on the validation set:

Method | BLEU@4 | METEOR | CIDEr | ROUGE-L 
-------|:--------:|:--------:|:--------:|:--------:|
Category-wise |40.9 |28.2 | 44.7|61.8
+C3D_16|42.2 |28.7 | 46.8|61.9
+C3D_2|43.4 |**29.4** | 49.6|**62.8**
+audio (BoAW)|**44.2** |**29.4** | **50.5**|62.6

## Two improvements for NeuralTalk2
To further improve the language model performance, we modified the vanilla [NeuralTalk2](https://github.com/karpathy/neuraltalk2) with two aspects.

1) A trick to overcome the GPU memory constrain by accumulating gradients over two training iterations. Set option `-update_iter` to a larger number if necessary.

2) See issue [87](https://github.com/karpathy/neuraltalk2/issues/87). Following the explanation there, we also replaced the log\_probs (p) with log\_perplexity (ppl) in the beam search operation. This is more consistent with the optimization function during training, and could give higher BLEU, METEOR, CIDEr and ROUGE-L scores (we used in our updated models).

If you find this helps your research, please consider citing:

     @inproceedings{shen2017weakly,
          title={Weakly Supervised Dense Video Captioning},
          author={Shen, Zhiqiang and Li, Jianguo and Su, Zhou and Li, Minjun and Chen, Yurong and Jiang, Yu-Gang and Xue, Xiangyang},
          booktitle ={CVPR},
          year={2017}
           }

## Contact
Zhiqiang Shen (zhiqiangshen0214 at gmail.com) 

Any discussions and suggestions are welcome!






