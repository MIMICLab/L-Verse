# L-verse: Bidirectional Generation Between Image and Text

**Taehoon Kim, Gwangmo Song, Sihaeng Lee, Sangyun Kim, Yewon Seo, Soonyoung Lee, Seung Hwan Kim, Honglak Lee, Kyunghoon Bae [[Paper]](https://arxiv.org/abs/2111.11133.pdf)** 

**LG AI Research**

**CVPR 2022 (Oral)** 

<img src=assets/lverse.png width=1280>


## Abstract
Far beyond learning long-range interactions of natural language, transformers are becoming the de-facto standard for many vision tasks with their power and scalability. Especially with cross-modal tasks between image and text, vector quantized variational autoencoders (VQ-VAEs) are widely used to make a raw RGB image into a sequence of feature vectors. To better leverage the correlation between image and text, we propose L-Verse, a novel architecture consisting of feature-augmented variational autoencoder (AugVAE) and bidirectional auto-regressive transformer (BiART) for text-to-image and image-to-text generation. Our AugVAE shows the state-of-the-art reconstruction performance on ImageNet1K validation set, along with the robustness to unseen images in the wild. Unlike other models, BiART can distinguish between image (or text) as a conditional reference and a generation target. L-Verse can be directly used for image-to-text or text-to-image generation tasks without any finetuning or extra object detection framework. In quantitative and qualitative experiments, L-Verse shows impressive results against previous methods in both image-to-text and text-to-image generation on MS-COCO Captions.  We furthermore assess the scalability of L-Verse architecture on Conceptual Captions and present the initial results of bidirectional vision-language representation learning on general domain. 



## Preparation

### Requirements

```
pip install -r requirements.txt
```

### Dataset

 Place any image dataset with ImageNet-style directory structure (directory with at least 1 sub-directory) to fit the dataset into pytorch ImageFolder.
 Alternatively, you can also use [ImageDataset2](https://github.com/lgai-research/L-Verse/blob/973ea99ab3053158fb4b92757d52d72a3b70fad9/latent_verse/loader.py#L201) which doesn't require any sub-directroy. In this case, replace [ImageDataset](https://github.com/lgai-research/L-Verse/blob/973ea99ab3053158fb4b92757d52d72a3b70fad9/latent_verse/loader.py#L126) with ImageDataset2. Our code also supports [WebDataset](https://github.com/webdataset/webdataset). 


### Pretrained weights 

- We provide the AugVAE pretrained weights on ImageNet dataset. 

    AugVAE-ML: [Google Drive](https://drive.google.com/file/d/1muj3Z-gEPwFtuwKLqZXAGZCVLa4eBKhk/view?usp=sharing)

    AugVAE-SL: [Google Drive](https://drive.google.com/file/d/1N9NOL5nOffBYCwYT7yTNa4X0zRJreabp/view?usp=sharing)

## AugVAE
### Training
For faster training, we our training code supports multi-gpu. 
To enable multi-gpu training, add " --gpus " flag with number of gpus in your machine (default 1).


For training, provide config file and training dataset.
If you are training AugVAE-SL, you must also provide pretrained AugVAE-ML weight
Please refer to example config files in configs. 


```
python train_vae.py --configs [config_file] --train_dir [path_to_train_data] --val_dir [path_to_val_data]
```

You can also test functionality with randomly generated fake data.

```
python train_vae.py --fake_data --configs [config_file] 
```

### Evaluation 
For faster evaluation, we our evaluation code supports multi-gpu. 
To enable multi-gpu evaluation, add " --gpus " flag with number of gpus in your machine (default 1).

For evaluation, provide config file, pretrained AugVAE weight, and test dataset
Please refer to example config files in configs. 


```
python eval_vae.py --configs [config_file] --ckpt_path [weight_file] --test_dir [path_to_test_data] 
```

You can also test functionality with randomly generated fake data.
```
python eval_vae.py --fake_data --configs [config_file] --ckpt_path [weight_file]
```

## BiART
Among many open-sourced Transformer (GPT) repositories, we used Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) with extra embedding layer for Segment Embedding. 

Here's an example modification code to apply Segment Embedding to [minGPT](https://github.com/karpathy/minGPT).

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, ... )):    
        ...
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.seg_emb = nn.Embedding(2, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))

    def forward(self, idx, seg, ...:
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        segment_embeddings = self.seg_emb(seg)
        ...
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + segment_embeddings + position_embeddings)
        ...
```

There's also [Pytorch Lightning version](https://github.com/williamFalcon/minGPT) which fits well with our AugVAE implementation.

## License

This project is distributed under MIT license.

```
Copyright (c) 2022-present LG AI Research.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## How to cite
```
@misc{kim2021lverse,
      title={L-Verse: Bidirectional Generation Between Image and Text}, 
      author={Taehoon Kim and Gwangmo Song and Sihaeng Lee and Sangyun Kim and Yewon Seo and Soonyoung Lee and Seung Hwan Kim and Honglak Lee and Kyunghoon Bae},
      year={2021},
      eprint={2111.11133},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


