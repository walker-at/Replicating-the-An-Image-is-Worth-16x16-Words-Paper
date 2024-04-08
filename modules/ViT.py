class ViT(nn.Module):
      """Vision Transformer model with default hyperparameters from ViT-Base."""
      # initialize with hyperparameters from Table 1 and Table 3
      def __init__(self,
                   img_size:int=224, # table 3
                   in_channels:int=3, # color channels
                   patch_size:int=16,
                   num_transformer_layers:int=12, # table 1
                   embedding_dim:int=768, # hidden size D from Table 1
                   mlp_size:int=3072, # Table 1
                   num_heads:int=12, # Table 1
                   attn_dropout:float=0, # no dropout for qkv attention projections
                   mlp_dropout:float=0.1, # for MLP layers
                   embedding_dropout:float=0.1, # for patch and position embedding
                   num_classes:int=1000): # Default for ImageNet, but we'll use 3
        super().__init__()
  
        # double check image size divisible by patch size
        assert img_size % patch_size == 0, f'Image size {img_size} must be divisible by patch size {patch_size}'
  
        # number of patches
        self.num_patches = (img_size * img_size) // patch_size**2
  
        # learnable class embedding (prepended to sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
  
        # learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
  
        # embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
  
        # patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
  
        # Transformer Encoder blocks (stack Transformer Encoder blocks with nn.Sequential())
        # "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
  
        # classifier head (only classification token goes through MLP head)
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
  
      # forward method -- we are including the class and position embedding that we left out of the patch_embedding module
      def forward(self, x):
  
        # batch size
        batch_size = x.shape[0]
  
        # class token embedding and expand to match the batch size
        class_token =self.class_embedding.expand(batch_size, -1, -1) # -1 infers dimension
  
        # patch embedding in equation 1
        x = self.patch_embedding(x)
  
        # preprend class token in equation 1
        x = torch.cat((class_token, x), dim=1)
  
        # add position embedding in equation 1
        x = self.position_embedding + x
  
        # embedding dropout in Appendix B.1
        x = self.embedding_dropout(x)
  
        # pass patch, position and class embedding through transformer encoder, equations 2 & 3
        x = self.transformer_encoder(x)
  
        # the classification token is at the 0 index, pass it through classifier for each image in a batch, equation 4
        x = self.classifier(x[:, 0])
  
        return x
