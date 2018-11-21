# Img Pre Processor

Image pre processor that runs with command line options.

Commands: 

```
engine.py -i IMAGE [-p PREPROCESS] [-b [BLUR]] [-r RESIZE]
          [-s SAVE]
```

Where:

```i```: the file path for the image;

```p```: preprocesses the img (binary threshold for gray scale);

```b```: blurs the image;

```r```: resizes the image for a given row == col argument;

```s```: saves the processed image on the OS. Otherwise, just displays the img.