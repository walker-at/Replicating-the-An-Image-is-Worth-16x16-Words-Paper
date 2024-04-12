class PatchEmbedding(nn.Module):
      '''
      Takes a 2D input image and converts it into a 1D learnable embedding vector.

      Args:
      in_channels (int): Num of color channels
      patch_size (int): Size of patch to convert image to
      embedding_dim (int): Size of embedding to convert image into.
      '''
      def __init__(self,
               img_size:int=224, # training res from table 3
               in_channels:int=3,
               patch_size:int=16,
               embedding_dim:int=768): # from table 1 for ViT
      super().__init__()

      self.patch_size = patch_size

      self.num_patches = (img_size * img_size) // patch_size**2

      # layer to turn an image into embedded patches
      self.patchify = nn.Conv2d(in_channels=in_channels,
                                out_channels=embedding_dim,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0)

      # layer to flatten feature map outputs of Conv2d
      self.flatten = nn.Flatten(start_dim=2, # only flatten height and width of feature map
                                end_dim=3)

      # learnable class embedding
      self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                          requires_grad=True)

      # learnable position embedding
      self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                             requires_grad=True)

      # func for the forward computation steps
      def forward(self, x):

          # batch size
          batch_size = x.shape[0]

          # class token - expand the single learnable class token across the batch dimension
          class_token = self.class_embedding.expand(batch_size, -1, -1) # we need to infer the dimension

          # patch_embedding
          x = self.patchify(x)

          # flatten
          x = self.flatten(x)
      
          # reshape
          x = x.permute(0, 2, 1)
      
          # prepend class embedding
          x = torch.cat((class_token, x), dim=1)

          # position embedding
          x = self.position_embedding + x
      
          return x
      
      # func for the forward computation steps
      def forward(self, x):
        
          # verify input shape
          image_resolution = x.shape[-1]
          assert image_resolution % patch_size == 0, f'input image shape: {image_resolution} must be divisible by patch size: {self.patch_size}'
      
          # forward pass
          x = self.patchify(x)
      
          x = self.flatten(x)
      
          # reshape to (batch_size, number_of_patches, embedding_dimension)
          return x_flattened.permute(0, 2, 1)

class MultiHeadSelfAttentionBlock(nn.Module):
      '''
      Creates the MSA block.
      '''
      def __init__(self,
                  embedding_dim:int=768, # from table 1
                  num_heads:int=12, # from table 1
                  attn_dropout:int=0): # appendix B specifies dropout not used after qkv projections
          super().__init__()
      
          # LN is going to normalize our input values across the shape we choose
          self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
      
          # MSA layer
          self.mh_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads,
                                               dropout=attn_dropout,
                                               batch_first=True) # True: (batch, number_of_patches, embedding_dimension)
      
      def forward(self, x):
          x = self.layer_norm(x)
          attn_output, _ = self.mh_attn(query=x,
                                        key=x,
                                        value=x,
                                        need_weights=False) # we dont want attn_output_weights so we put an underscore
      return attn_output

class MLPBlock(nn.Module):
      '''
      Creates the MLP block.
      '''
      def __init__(self,
                  embedding_dim:int=768,
                  mlp_size:int=3072,
                  dropout:int=0.1): # From appendix B.1 we know to use dropout after every linear layer
          super().__init__()
      
          # LN
          self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
      
          # MLP
          self.mlp = nn.Sequential(
              nn.Linear(in_features=embedding_dim,
                        out_features=mlp_size),
              nn.GELU(),
              nn.Dropout(p=dropout), # dropout after every linear layer
              nn.Linear(in_features=mlp_size,
                        out_features=embedding_dim),
              nn.Dropout(p=dropout)
          )
      
      # pass data through layers
      def forward(self, x):
          x = self.layer_norm(x)
          x = self.mlp(x)
          return x

class TransformerEncoderBlock(nn.Module):
      '''
      Puts the MSA and MLP block together for the transformer encoder block.
      '''
      def __init__(self,
                  embedding_dim: int=768,
                  num_heads: int=12, # for MSA block
                  mlp_size:int=3072, # for MLP block
                  mlp_dropout:int=0.1, # for MLP block
                  attn_dropout:int=0):
          super().__init__()

          # MSA block from equation 2
          self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 attn_dropout=attn_dropout)

          # MLP block from equation 3
          self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                              mlp_size=mlp_size,
                              dropout=mlp_dropout)

      def forward(self, x):
          x = self.msa_block(x) + x # residual connection for equation 2
          x = self.mlp_block(x) + x # residual connection for equation 3
          return x
