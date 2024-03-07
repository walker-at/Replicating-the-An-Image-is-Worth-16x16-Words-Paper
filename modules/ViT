class PatchEmbedding(nn.Module):
  def __init__(self,
               in_channels:int=3,
               patch_size:int=16,
               embedding_dim:int=768): # from table 1 for ViT
    super().__init__()

    self.patch_size = patch_size

    # layer to turn an image into embedded patches
    self.patchify = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)

    # layer to flatten feature map outputs of Conv2d
    self.flatten = nn.Flatten(start_dim=2, # only flatten height and width of feature map
                              end_dim=3) 

  # func for the forward computation steps
  def forward(self, x):
    # verify input shape
    image_resolution = x.shape[-1]
    assert image_resolution % patch_size == 0, f'input image shape: {image_resolution} must be divisible by patch size: {self.patch_size}'

    # forward pass
    x_patched = self.patchify(x)
    x_flattened = self.flatten(x_patched)
    # reshape to (batch_size, number_of_patches, embedding_dimension)
    x_flattened_reshaped = x_flattened.permute(0, 2, 1)

    batch_size, embedding_dimension = x_flattened_reshaped.shape[0], x_flattened_reshaped.shape[-1]
        
    # Create class token
    class_token = nn.Parameter(torch.randn(batch_size, 1, embedding_dimension), requires_grad=True)
    x_class_prepended = torch.cat((class_token, x_flattened_reshaped), dim=1)

    # Create position embedding
    height, width = x.shape[-2], x.shape[-1]
    number_of_patches = int((height * width) / (self.patch_size ** 2))
    position_embedding = nn.Parameter(torch.ones(1, number_of_patches + 1, embedding_dimension), requires_grad=True)

    x_position_embedded = x_class_prepended + position_embedding

    return x_position_embedded
