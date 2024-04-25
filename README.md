This is a step-by-step implementation of the "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" paper in PyTorch code.

Paper: https://arxiv.org/abs/2010.11929

The project notebook opens with a brief discussion of the Transformer architecture, its application to a Computer Vision task, and the challenges compared to Convolutional Neural Networks. I then go through the machine learning workflow from data preprocessing and training a model, to utilizing transfer learning and hyperparamter tuning. Along the way, the mathematical equations of the original ViT paper are turned into clean PyTorch modules. The pieces as well as the final ViT architecture can be found in the modules folder.

Throughout the project there were 2 main errors I found myself running into: incorrect/non-matching shape sizes, and device related errors. For this reason, you'll notice in the final workplace notebook I am consistently checking the shape of our image as it goes through our vit architecture, verifying input and output shapes match up. I am also specific wherever needed about the device to be run on.

For clarity in what's happening in the data-preprocessing steps, I make sure to visualize the journey a sample image takes to become a properly formatted input image: it goes from regular image, to a patchified image, to a convolutional feature map (2D embedding), and finally to a flattened feature map (1D embedding vector)

1. Import Libraries and Set Device
2. Get Data
  * I use a function stored in helper_funcs called get_data to download the data from my github
  * I use a function stored in data_setup called create_dataloaders to apply transforms to the train and test data and turn them into dataloaders
3. Equation 1: Split Data into Patches and Create the Class, Position and Patch Embedding -- The Patch Embedding Layer
  * Turn the following equation into a reusable PyTorch Module: Eq1_output = [class_token, patch1_embedding, patch2_embedding, ... patchN_embedding] + [position_token for each of the elements in that previous vector]
4. Equation 2: Multi-Head Self-Attention (MSA) Block Consisting of MSA layer and LayerNorm (LN) -- Part 1 of our Transformer Encoder
  * Turn the following equation into a reusable PyTorch Module: Eq_2 output = MSA_layer(LN_layers(Eq1_output)) + Eq1_output
5. Equation 3: MultiLayerPerceptron (MLP) Block consisting of MLP Layer and LN -- Part 2 of our Transformer Encoder
  * Turn the following equation into a reusable PyTorch Module: Eq3_output = MLP_layer(LN_layer(Eq2_output)) + Eq2_output
6. Put together Eq.2 and Eq.3 to form our Transformer Encoder
7. Create the full ViT Architecture
  * Equation 4 is added in here: Eq4_output = Linear_layer(LN_layer(Eq3_output))
8. Training our Model and Plot Results
  * Pick a Loss Func: its multi-class classification so we use torch.nn.CrossEntropyLoss
  * Optimizer: paper states they used Adam optimizer
9. What Went Wrong?
10. Transfer Learning: Fine-tuning a Feature Extractor from a Pre-trained ViT
11. Custom Image Prediction
