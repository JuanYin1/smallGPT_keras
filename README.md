# result of translating from bigram.py to bigram_keras.py

## `bigram.py`
<img width="751" alt="截屏2024-05-09 19 35 51" src="https://github.com/JuanYin1/smallGPT_keras/assets/79886525/9cbe2100-d2f0-45c9-a296-278fda92d2ea">

## `bigram_keras.py`
<img width="1118" alt="截屏2024-05-09 19 36 07" src="https://github.com/JuanYin1/smallGPT_keras/assets/79886525/e4a85cb4-129d-482c-bba0-04c7230e5fb2">



# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT
