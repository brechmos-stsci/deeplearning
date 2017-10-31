from keras import applications

# build the VGG16 network
model = applications.VGG16(include_top=True,
                           weights='imagenet')


def layer_to_visualize(layer, img_to_visualize):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = 4
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions))[:4]:
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')


from scipy.ndimage import imread

input_img = imread('hubble.jpg')
hubble = input_img[:224,:224]
hubble = np.expand_dims(hubble, axis=0)
layer_to_visualize(model.layers[4], hubble)

