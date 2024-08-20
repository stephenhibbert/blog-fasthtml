---
title: 👀 How does a LLM see?
date: '2024-5-6'
tags:
- llm
- vlm
- vit
- theory
- cv
- transformers
published: true
description: Building a Vision Language Model (VLM) with an LLM + Vision Encoder.
image: /public/images/llm-vision/banner.png
twitter_image: /public/images/llm-vision/banner.png
---

In 2024, many foundation models have native vision capabilities.
Anthropic Claude 3 Opus, OpenAI GPT 4o and Google Gemini Pro 1.5 all have the ability to generate a recipe from a photo of ingredients, or create a functional website from a doodle.

How does this work?! At the time of writing the frontier models are black boxes and we know very few details about their architecture.
But happily the open source community is also making progress and recent publications have made it accessible to start building a picture of how LLM vision might work in these models.

In this post we start by reviewing fundamentals images and computer vision (feel free to skip this if you're already familiar).
Next we introduce the Vision Transformer (ViT) as a computer vision architecture.
Finally, we see how we can apply ViT to encoded images into in a modern vision capable LLM.

- [Part 1 - Computer Vision](#computer-vision)
- [Part 2 - Vision Transformer (ViT)](#vision-transformer-vit)
- [Part 3 - Vision Language Model (VLM)](#vision-language-model)

## Computer Vision

Let's start by reviewing how biological vision works.
The human eye takes an input of electromagnetic waves in the visible spectrum that happen to have scattered off an object.
This light passes through the cornea and pupil, is refracted by the lens and focussed onto the retina.
The **image** is the pattern of light that is formed on the retina. It's a 2D projection of the 3D world.

> <Image alt="human eye" src="/public/images/llm-vision/human-eye.png" width={500} height={350} />
> <cite>Human eye. Drawn on https://www.tldraw.com/</cite>

The retina is covered in photoreceptor cells which are sensitive to light.
The brain processes the electrical signal from the retina to create a mental model of the world around us.

Computer vision works very differently to human vision but there are some analogies.
Instead of the biological eye, we have digital cameras with artificial sensors that convert light into electrical signals.
Instead of the brain, we have computer vision models that process these signals to understand the world around us.

The input to a vision model starts with a digital image.
A digital image is an 2D array of pixels where each pixel at coordinates $(x, y)$ has a numerical value which is stored as bytes in a file.

Consider this image from the ImageNet dataset. It appears to be a picture of a lionfish:

> <Image alt="lionfish" src="/public/images/llm-vision/lionfish.jpg" width={346} height={346} />
> <cite>ImageNet dataset: https://image-net.org</cite>

This image has a width of $640$ pixels and a height of $480$ pixels.
$(0, 0)$ is the top left pixel.
The pixel in position $(0, 320)$ is in the first column and the middle row.
This pixel has the $(R, G, B)$ value of $(59, 142, 164)$. Each number is a byte with a value between $0$ and $2^8 = 255$ in decimal.

**Color**  
- R: ![#3B0000](/public/images/llm-vision/3B0000.png) `#3B0000`  
- G: ![#008E00](/public/images/llm-vision/008E00.png) `#008E00`  
- B: ![#0000A4](/public/images/llm-vision/0000A4.png) `#0000A4`  
- Total: ![#3B8EA4](/public/images/llm-vision/3B8EA4.png) `#3B8EA4`

**Decimal**  
- **R**: `59`  
- **G**: `142`  
- **B**: `164`  
- **Total**: `(59, 142, 164)`

**Binary**  
- **R**: `00111011`  
- **G**: `10001110`  
- **B**: `10100100`  
- **Total**: `00111011 10001110 10100100`

**Hex**  
- **R**: `0x3B`  
- **G**: `0x8E`  
- **B**: `0xA4`  
- **Total**: `#3B8EA4`

Notice the highest value is in the blue channel, followed by green with
only a small contribution from red. This pixel shows the blue-green (turquoise) of the ocean.

There are $w * h$ pixels which makes $640 * 480 = 307,200$ pixels for our lionfish image.
With the 3 channels, this means there are $307,200 * 3 = 921,600$ bytes (almost 1MB) required to store the raw image.
However, this image is actually encoded as a JPEG which uses a lossy compression algorithm to reduce the number of bytes
required to store on disk to just 49KB. But that's a topic for another time.

When feeding images to a computer vision model, we resize and 'flatten' it to map the pixel values into a 1D array.
A normalisation step ensures that the pixel values lie within a specific range, typically $[-1, 1]$.
Normalization for images, just like for other types of input data, ensures that the features (pixel values) are on the same scale, leading to more effective and stable learning.

It helps in achieving better performance, faster convergence, and improved generalization by standardizing the input range.

<Image alt="resize & flatten" src="/public/images/llm-vision/flatten.png" width={800} height={600} />

Now the transformed image is ready to be fed into the first layer of a neural network.
There are many architectures for computer vision tasks, but the most popular has been convolutional neural networks (CNNs) since [AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) kickstarted the deep learning revolution in 2012.
More recently, transformers have been applied to computer vision tasks, which we will explore in the next section.

<Image alt="nn" src="/public/images/llm-vision/nn.png" width={500} height={300} />

For an excellent introduction to deep neural networks, check out 3Blue1Brown's [Neural Networks series](https://www.youtube.com/watch?v=aircAruvnKk).
For a more rigorous treatment, dive into Simon Prince's new book, [Understanding Deep Learning](https://www.amazon.co.uk/dp/0262048647).

## Vision Transformer (ViT)

Transformers have been a game changer in natural language processing (NLP) since the release of the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper in 2017.
The transformer architecture has since been adapted to many other domains, including computer vision.
This approach was originally laid out in [An Image is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929).

In the ViT architecture, the image is split into fixed-size patches, which are then linearly embedded before being fed into a transformer block.
We'll work through this process step by step.

Loading the lionfish image using the PIL library we first resize to a standard size expected by the model - which is 224x224.
Let's do this manually before digging further under the hood.

```python
url = 'https://github.com/latenttime/blog/blob/main/public/public/images/llm-vision/lionfish.jpg?raw=true'
image = Image.open(requests.get(url, stream=True).raw)
image = image.resize((224, 224))
image
```

We can see the resulting squished and more pixelated image below:

<Image alt="lionfish resized" src="/public/images/llm-vision/lionfish_resized.png" width={224} height={224} />

A naive approach to processing the image with a transformer would be to flatten the entire image into a 1D array of pixels.
The (normalised) raw pixel values would be passed to the model as input.
Our flattened vector has length $224 * 224 * 3 = 150,528$ floating point numbers.

However, we know that the pixel values are not independent of each other.
The pixel, for example, at position $(20, 100)$ is likely to have a similar value to adjacent pixels pixel at positions $(20, 101)$ and $(19, 100)$.
Furthermore, there are sequences of pixels that form lines, shapes and objects in the image.
These spacial semantics can be learned from data by a deep neural network at training time.
The dimensionality of the input is much higher than the actual information content of the image for real world distributions of images.

On a more practical level, we know that the attention blocks in the transformer architectures scale quadratically with the input length.
Therefore, it would be very computationally expensive to process raw images in this way.

In practice, we split an image into fixed-size patches (e.g. 16x16 pixels), linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard transformer encoder.

We'll use a [Google vision transformer model](https://huggingface.co/google/vit-base-patch16-224) downloaded from Hugging Face to process our lionfish image and extract the features so we can build an intuition for what's happening.

```python
from transformers import ViTImageProcessor, ViTForImageClassification

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
```

The true patch size of `google/vit-base-patch16-224` is 16x16 pixels.
In reality there are $14*14=196$ patches for the 224x224 image.
To aid our visualisations we'll pretend the patch size is 64x64 pixels which makes 9 patches.
Bear in mind that the model is actually working with 4x smaller patches.

<Image alt="patch grid" src="/public/images/llm-vision/patch_grid.png" width={224} height={224} />

Let's choose a single patch and see how we go from RGB pixel integers to a 768 dimension embedding vector.
The first step is to flatten the patch. Similar to how we flatten the 2D array of patches in the original image, we take the pixels row by row and concatenate them into a 1D array.
We then normalise the integer RGB pixel values to a float between -1 and 1.
It's then common to use a linear projection which is implemented as a learnable linear layer.
This multiplication by a weight matrix is equivalent to a linear transformation of the pixel values.
The output is a 768 dimension latent vector that no longer directly represents the pixel values but instead a learned representation of the patch.

<Image alt="patch flat" src="/public/images/llm-vision/patch_flat.png" width={640} height={240} />

Note that the patch sequence order matters here. Just like with text, the meaning changes if the patches are shuffled.

<Image
  alt="patch reordered grid"
  src="/public/images/llm-vision/patch_grid_reordered.png"
  width={224}
  height={224}
/>

In order to represent the correct order [position embeddings are added](https://github.com/google-research/vision_transformer/blob/143bd26cf285f835a2d1954f85b14f33f7d3ea8e/vit_jax/models_vit.py#L37-L63) to the image patch embeddings.
A [class token (CLS) is optionally added](https://github.com/google-research/vision_transformer/blob/143bd26cf285f835a2d1954f85b14f33f7d3ea8e/vit_jax/models_vit.py#L279-L283) for classification tasks.
The encoded image embeddings are subsequently input to the transformer block in the LLM.

<Image alt="patch embedding" src="/public/images/llm-vision/patch_embedding.png" width={640} height={240} />

Finally, we have our image patch tokens, which are the input to the transformer.

We can extract the patch embeddings from the model's hidden states to visualise the information the model is working with.
With the transformers library we can do this by setting `output_hidden_states=True` in the forward pass.

```python
def get_patch_embeddings_for_layer(hidden_layer = 0):
    patch_embeddings = []
    # Iterate through the length of patches
    for idx in range(len(patches)):
        embeddings = outputs.hidden_states[hidden_layer][:, idx, :]  # Get the hidden state's embeddings
        patch_embeddings.append(embeddings.squeeze(0))  # Squeeze to remove batch dimension and append
    return patch_embeddings

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    patch_embeddings = get_patch_embeddings_for_layer(hidden_layer)
    embeddings = torch.stack([emb.squeeze() for emb in patch_embeddings])
```

Each embedding vector is treated similarly to a token embedding in NLP tasks, where each patch embedding represents part of the image information, analogous to how each word embedding would represent part of the semantic information in a text processing task.
We can visualise the 768 dimension embedding vectors in a 2D space using PCA to reduce the dimensionality of the embeddings.
Below we see the patch embeddings for the first layer of the transformer.
Note we switched back to the true 16x16 patch size for this visualisation.

<Image alt="patch embeddings layer 0" src="/public/images/llm-vision/layer_0.png" width={640} height={640} />

The transformer then processes these embeddings, considering both individual features and interactions between patches (through self-attention mechanisms), to understand and classify the image or perform other vision tasks.

The `google/vit-base-patch16-224` model has 12 hidden layers in total.
Below we can see how the PCA of the embeddings evolve through each layer in the transformer in this model which is trained on ImageNet.
The vector output from the final layer is used to predict the class of the image.
Note that in an vision capable LLM trained on a different task, the activations will be different but the principle remains the same.

<Image
  alt="patch embeddings gif"
  src="/public/images/llm-vision/patch_embeddings_pca.gif"
  width={640}
  height={640}
/>

We can see in the [model card](https://huggingface.co/google/vit-base-patch16-224) that this model has been trained for zero shot image classification on the ImageNet dataset.
There is a prediction processing head that takes the output of the transformer blocks to maps the final hidden state to the 1,000 ImageNet classes.

So our lionfish was actually in the training set, so hopefully the model can predict the correct class:

```python
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

```
Predicted class: lionfish
```

Indeed it does!

## Vision Language Model

Now that we have a basic understanding of how a vision transformer works, we can see how this architecture can be used in a vision capable LLM.
Building upon image classification, the focus here shifts to autoregressive text generation (similar to a language model), conditioned on a combination of interleaved images and text.
This opens up new possibilities for tasks such as general visual question answering, counting, captioning, text transcription, document understanding, chart and figure understanding, table understanding, visual reasoning, geometry, spotting differences between two images, or even converting a screenshot to functional code.

Remember the state of the art is developing rapidly and the details of the latest models are not always public.
Luckily for the open source community, the recent [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) model from Hugging Face comes with a recently released paper, [What matters when building vision-language models?](https://arxiv.org/pdf/2405.02246).
This will be our reference for the following section, but other models may have different architectures.

SOTA VLMs are commonly composed by gluing together pre-trained unimodal backbones, and initialising some new parameters to connect the modalities.
Idefics2 reuses [Mistral-7B](https://huggingface.co/google/siglip-so400m-patch14-384) for the language backbone and [SigLIP-SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) for the vision encoder.

When fusing the two modalities, the image and text embeddings are concatenated and passed through a transformer encoder.
The visual embeddings from the from the visual encoder are projected through a learned linear mapping into the input space of the language model.

<Image alt="vlm" src="/public/images/llm-vision/vlm.png" width={640} height={640} />

The new parameters first undergo a pre-training phase on a large dataset of image-text pairs.
The Hugging Face team also released the [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) dataset used for this pre-training.
Just like in the language only LLMs, the model is then fine-tuned on a downstream task in an instruction fine-tuning stage.
For Idefics2, the fine tuning dataset was [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), a collection of 50 vision-language datasets.

After training, we can run inference with a combined text and image input.
The multimodal representations are computed and passed to the LLM transformer blocks to process these embedding vectors to understand the context and autoregressively generate the output text.

Finally, we have a vision capable LLM! 👀

## Conclusion

We started by reviewing some fundamentals of computer vision.
Next we learned about transformers can be applied to computer vision with vision transformer (ViT) architectures.
Finally, we saw conceptually how to fuse LLM and VE backbones to create a vision capable LLM, a VLM.

You can find the code used to create the visualisations for this post in this [notebook](https://github.com/latenttime/blog/blob/main/notebooks/llm-vision.ipynb).

## References

https://arxiv.org/pdf/2010.11929

https://arxiv.org/pdf/2405.02246

https://arxiv.org/abs/2301.13823

https://huggingface.co/google/vit-base-patch16-224

https://huggingface.co/HuggingFaceM4/idefics2-8b

https://huggingface.co/google/siglip-so400m-patch14-384

https://huggingface.co/mistralai/Mistral-7B-v0.1
