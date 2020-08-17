# fun-neural-style-transfer
I'm very bored! I don't know how long it would take to complete (I meant, my new model)! So, I suppose I need to have something to do! This project sounds fun and exciting 🤩, and a little bit needs some different configurations, which I'm going to show you how to do it! Stay tuned!

## How to use?
Open your terminal and follow the process step by step.

```bash
git clone https://github.com/m3hrdadfi/fun-neural-style-transfer.git app
cd app

python transfer.py \
    --language pt \
    --content_image_path SPECIFY_YOUR_CONTENT_IMAGE_PATH \
    --content_type d \
    --style_image_path SPECIFY_YOUR_STYLE_IMAGE_PATH \
    --style_type e \
    --log_every_step 500 \
    --steps 4000 \
    --learning_rate 1e-3 \
    --style_weights 0.2,0.2,0.2,0.2,0.2 \
    --alpha 8.0 \
    --beta 1e4 \
    --save_dir SPECIFY_YOUR_OUTPUT_DIR
```

For more information
```bash
python transfer.py --h
```

Output
```bash
usage: transfer.py [-h] --content_image_path CONTENT_IMAGE_PATH
                   [--content_type CONTENT_TYPE] --style_image_path
                   STYLE_IMAGE_PATH [--style_type STYLE_TYPE] --language
                   LANGUAGE [--style_weights STYLE_WEIGHTS] [--alpha ALPHA]
                   [--beta BETA] [--steps STEPS]
                   [--log_every_step LOG_EVERY_STEP]
                   [--learning_rate LEARNING_RATE] [--save_dir SAVE_DIR]
                   [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --content_image_path CONTENT_IMAGE_PATH
                        The content image path as p (picture)
  --content_type CONTENT_TYPE
                        The content type must be in the list: a, b, c, d, e
  --style_image_path STYLE_IMAGE_PATH
                        The style image path as a (artwork)
  --style_type STYLE_TYPE
                        The style type must be in the list: a, b, c, d, e
  --language LANGUAGE   Language name selected in the list: pt, tf
  --style_weights STYLE_WEIGHTS
                        Style weights must be declared as list of int for all
                        layers: conv1_1, conv2_1, conv3_1, conv4_1, conv4_1
  --alpha ALPHA
  --beta BETA
  --steps STEPS
  --log_every_step LOG_EVERY_STEP
  --learning_rate LEARNING_RATE
  --save_dir SAVE_DIR
  --seed SEED           random seed for initialization
```

[![asciicast](https://asciinema.org/a/353815.svg)](https://asciinema.org/a/353815)

Notebook Example regarding how to use `transfer.py` script

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/fun-neural-style-transfer/blob/master/notebooks/PyTorch_Neural_Style_Transfer_CMD.ipynb)

## Final Result
Neural Style Transfer (NST) depends on two entities, a content and a style. It accepts one pair of images, one as content and the other as a styler. Then, it produces a new target that preserves the content and tries to shift the style of primary into the styler. In summary:

- Content image used as a basis that provides objects and arrangements for the target.
- Style image used as a painter (styler) to transfer its style, colors, and textures to the target image.

**Content and Style**

![Content - Style](/assets/content-style.png)

**Target (as a fruit of NST)**

![Content - Style](/assets/target.jpg)



## Copyright

- Content: [Masuleh](https://unsplash.com/photos/I5oxikudcFo)
- Style: [Golestan Palace Springhouse](https://fa.wikipedia.org/wiki/%D8%AD%D9%88%D8%B6%E2%80%8C%D8%AE%D8%A7%D9%86%D9%87_%D8%B9%D9%85%D8%A7%D8%B1%D8%AA_%DA%AF%D9%84%D8%B3%D8%AA%D8%A7%D9%86)
  

## License
This project is entirely free and open-source and licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.