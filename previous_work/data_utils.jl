using Images, TestImages, Statistics, Distributions
"""
Randomly apply function func to x with probability p.
"""
function random_apply(func, p, x)

    return rand() .< p ? func(x) : x
end

"""
Transform RGB image to grayscale image
If keep_channels=true, channel number will be tiled to 3.
"""
function to_grayscale(image; keep_channels=true)
    image = Gray.(image)
    if keep_channels
        image = repeat(image, outer = [1, 1, 3])
    end
    return image
end


"""
Adjust contrast of RGB or grayscale images.
This is a convenience method that converts RGB images to float
representation, adjusts their contrast, and then converts them back to the
original data type. If several adjustments are chained, it is advisable to
minimize the number of redundant conversions.
`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`
Contrast is adjusted independently for each channel of each image.
For each channel, this Op computes the mean of the image pixels in the
channel and then adjusts each component `x` of each pixel to
`(x - mean) * contrast_factor + mean`.
"""
function random_contrast(image, lower, upper)
    cont_f= rand(lower:0.00001:upper)
    #rand(Uniform(4,6))
    mean_f =  mean(image, dims=(1,2))
    f_image = (image .- mean_f)*cont_f .+ mean_f
    return f_image
end

"""
Adjust the brightness of RGB or Grayscale images.
 This is a convenience method that converts RGB images to float
 representation, adjusts their brightness, and then converts them back to the
 original data type. If several adjustments are chained, it is advisable to
 minimize the number of redundant conversions.
 The value `delta` is added to all components of the tensor `image`. `image` is
 converted to `float` and scaled appropriately if it is in fixed-point
 representation, and `delta` is converted to the same data type. For regular
 images, `delta` should be in the range `(-1,1)`, as it is added to the image
 in floating point representation, where pixel values are in the `[0,1)` range.
"""
function random_brightness(image, brightness)
    if brightness < 0
        raise("brightness must be non-negative.")
    else
        r_bright = rand(-brightness:0.00001:brightness)
        #image = Float64.(image)
        #b_image =  permutedims(channelview(image), (2, 3, 1)) .+ r_bright
        permutedims(channelview(image),(2, 3, 1))
        image =  image .+ r_bright
        image = RGB.(image)
    end
    return image
end


function random_saturation(image, lower, upper)
    hsv_img = HSV.(image)
    channels = channelview(float.(hsv_img))
    sat_factor = rand(lower:0.00001:upper)
    channels[2,:,:] *= sat_factor
    #channels = clamp(channels,0, 1)
    #image = RGB.(channels)
    image =colorview(RGB, channels)
    #image = convert(Array{RGB{Float64}}, image);
    return image
end



"""
Adjust saturation of RGB images.
This is a convenience method that converts RGB images to float
representation, converts them to HSV, adds an offset to the
saturation channel, converts back to RGB and then back to the original
data type. If several adjustments are chained it is advisable to minimize
the number of redundant conversions.
`image` is an RGB image or images.  The image saturation is adjusted by
converting the images to HSV and multiplying the saturation (S) channel by
`saturation_factor` and clipping. The images are then converted back to RGB.
"""
function random_saturation(image, lower, upper)
    hsv_img = HSV.(image)
    channels = channelview(float.(hsv_img))
    sat_factor = rand(lower:0.00001:upper)
    channels[2,:,:] *= sat_factor
    channels = clamp(channels,0, 1)
    image = RGB.(channels)
    return image
end


"""
Adjust hue of RGB images.
  This is a convenience method that converts an RGB image to float
  representation, converts it to HSV, adds an offset to the
  hue channel, converts back to RGB and then back to the original
  data type. If several adjustments are chained it is advisable to minimize
  the number of redundant conversions.
  `image` is an RGB image.  The image hue is adjusted by converting the
  image(s) to HSV and rotating the hue channel (H) by
  `delta`.  The image is then converted back to RGB.
  `delta` must be in the interval `[-1, 1]`.
"""
function random_hue(image, max_delta)
    if max_delta > 0.5
        error("max_delta must be <= 0.5.")
    end
    if max_delta < 0
        error("max_delta must be non-negative.")
    end
    hsv_img = HSV.(image)
    channels = channelview(float.(hsv_img))
    hue_factor = rand(-max_delta:0.00001:max_delta)
    channels[1,:,:] *= hue_factor
    #channels = clamp(channels,0, 1)
    #image = RGB.(channels)
    image =colorview(RGB, channels)
    #image = convert(Array{RGB{Float64}}, image);
    return image
end

"""Preprocesses the given image for training.

Args:
  image: `Tensor` representing an image of arbitrary size.
  height: Height of output image.
  width: Width of output image.
  color_distort: Whether to apply the color distortion.
  crop: Whether to crop the image.
  flip: Whether or not to flip left and right of an image.

Returns:
  A preprocessed image `Tensor`.
"""
function preprocess_for_train(image, height, width,
                         color_distort=True, crop=True, flip=True)
  if crop
      image = random_crop_with_resize(image, height, width)
  end
  if flip
      image = random_flip_left_right(image)
  end
  if color_distort
      image = random_color_jitter(image)
  end
  image = reshape(image, [height, width, 3])
  image = clip_by_value(image, 0., 1.)
  return image
end

  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.

  Returns:
    A preprocessed image `Tensor`.
  """
function preprocess_for_eval(image, height, width, crop=True)

  if crop
      image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  end
  image = reshape(image, [height, width, 3])
  image = clip_by_value(image, 0., 1.)
  return image
end

  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    is_training: `bool` for whether the preprocessing is for training.
    color_distort: whether to apply the color distortion.
    test_crop: whether or not to extract a central crop of the images
        (as for standard ImageNet evaluation) during the evaluation.

  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
function preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):

  image = convert_image_dtype(image, dtype=tf.float32)
  if is_training
    return preprocess_for_train(image, height, width, color_distort)
  else
    return preprocess_for_eval(image, height, width, test_crop)
end
