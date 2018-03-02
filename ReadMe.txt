digit_detection:

ex1.ipynb includes the models for recognizing digits of Mnist,
and also includes the code to predict my own images.

There is a .py file which also can predict own images.
To predict own images:  1) put images in the "images" folder ( There are some images there already )
			2) make sure you have "img_resized" folder ( will be fiiled by the program )
			3) make sure you got the "best_model" folder with the model in it
			4) run the .py file or the ipynb jupyter notebook ( the wanted part of the code )

The "Reports" folder holds 3 Word docx with the metadata about the models.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Animals_flowers_detection:

Ex2.ipynb includes the models for recognizing animals from Cifar, and also
includes the fine tuned model for recognizing flowers

There are 2 .py files:  1) recognize own images of animals
			2) recognize own images of flowers or not flowers

predict own images:
	animals: 1) put images in the "images" folder ( There are some images there already )
		 2) make sure you have "img_resized" folder ( will be fiiled by the program )
		 3) make sure you got the "best_model" folder with the model in it
		 4) run the .py file or the ipynb jupyter notebook ( the wanted part of the code )

	flowers: 1) put images in the "images_transfer" folder
		 2) make sure you have "img_resized_transfer" folder ( will be fiiled by the program )
		 3) make sure you got the "saved_model_transfer" folder with the model in it
		 4) run the .py file or the ipynb jupyter notebook ( the wanted part of the code )

The "Reports" folder holds 3 Word docx with the metadata about the models.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ImageGeneration:

color_and_resize_images.py includes all the code for coloring a 32x32 Grayscale flower img
and then upsacling it to 96X96 using super resolution.

*you need to make sure you got "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5" file
of the weights of inception resnet v2 for the coloring model.

*make sure you have a "saved models" folder with the coloring model Json file, coloring models weights and
the super resolution models weights

Link to one drive shared folder with the weights, Test example and our zipped project:
https://1drv.ms/f/s!AiNgY5UvQ308gYBc2XhF9CIrHjvcZw

copy the weights file which is named to the root folder, ImageGeneration:
inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5

color and resize own images: 1) put images in the directory you want
			     2) make sure you have directory: Test/color result and Test/final result
			     3) run color_and_resize_images.py and enter the directory with the images

The "Reports" folder holds a Word docx with the metadata about the models.
				