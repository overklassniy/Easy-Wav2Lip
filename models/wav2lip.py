import torch
from torch import nn
from torch.nn import functional as F

from utils.logger import get_logger
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

logger = get_logger()


class Wav2Lip(nn.Module):
    """Wav2Lip model for generating lip-synced frames from audio and face inputs."""

    def __init__(self) -> None:
        """Initialize the Wav2Lip model with face encoder/decoder and audio encoder."""
        super(Wav2Lip, self).__init__()

        # Build face encoder blocks.
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),  # 96x96

            nn.Sequential(
                Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48x48
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24x24
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12x12
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6x6
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3x3
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),

            nn.Sequential(
                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1x1
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),
        ])

        # Build audio encoder.
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        # Build face decoder blocks.
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3x3
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # 6x6

            nn.Sequential(
                Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # 12x12

            nn.Sequential(
                Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # 24x24

            nn.Sequential(
                Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # 48x48

            nn.Sequential(
                Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # 96x96
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        logger.debug("Initialized Wav2Lip model.")

    def forward(self, audio_sequences: torch.Tensor, face_sequences: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Wav2Lip model.

        Processes audio and face inputs through their respective encoders, decodes the fused
        representation, and produces a lip-synced output frame.

        Args:
            audio_sequences (torch.Tensor): Audio input tensor.
            face_sequences (torch.Tensor): Face input tensor.

        Returns:
            torch.Tensor: Output tensor containing generated lip-synced frame(s).
        """
        # Handle potential extra dimensions if inputs are provided as (B, T, ...).
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            # Collapse temporal dimension by concatenation.
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)  # (B, 512, 1, 1)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        # Use audio embedding as starting point for decoder.
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                # Concatenate skip connection features.
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                logger.error(f"Concatenation error: {e}")
                raise e
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            # Split and stack outputs to recover temporal dimension.
            x = torch.split(x, B, dim=0)  # list of (B, C, H, W)
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)
        else:
            outputs = x

        logger.debug("Wav2Lip forward pass completed.")
        return outputs


class Wav2Lip_disc_qual(nn.Module):
    """Discriminator network for quality assessment of generated lip-synced frames."""

    def __init__(self) -> None:
        """Initialize the quality discriminator network."""
        super(Wav2Lip_disc_qual, self).__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(
                nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),
                nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            ),

            nn.Sequential(
                nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            ),

            nn.Sequential(
                nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),
        ])

        self.binary_pred = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.label_noise: float = 0.0
        logger.debug("Initialized Wav2Lip_disc_qual network.")

    def get_lower_half(self, face_sequences: torch.Tensor) -> torch.Tensor:
        """Extract the lower half of the face sequences.

        Args:
            face_sequences (torch.Tensor): Input tensor with face images.

        Returns:
            torch.Tensor: Lower half of the input tensor.
        """
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences: torch.Tensor) -> torch.Tensor:
        """Collapse temporal dimension of face sequences to produce 2D inputs.

        Args:
            face_sequences (torch.Tensor): Input tensor with shape (B, C, T, H, W) or similar.

        Returns:
            torch.Tensor: Collapsed 2D tensor.
        """
        B = face_sequences.size(0)
        return torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

    def perceptual_forward(self, false_face_sequences: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss on false (generated) face sequences.

        Args:
            false_face_sequences (torch.Tensor): Generated face sequences.

        Returns:
            torch.Tensor: Perceptual loss value.
        """
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)
        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)
        loss = F.binary_cross_entropy(
            self.binary_pred(false_feats).view(len(false_feats), -1),
            torch.ones((len(false_feats), 1)).cuda()
        )
        return loss

    def forward(self, face_sequences: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator network.

        Args:
            face_sequences (torch.Tensor): Input face sequences.

        Returns:
            torch.Tensor: Binary prediction for each input sample.
        """
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
        return self.binary_pred(x).view(len(x), -1)
