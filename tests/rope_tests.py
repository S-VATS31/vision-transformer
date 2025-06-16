class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.head_dim = 64  # divisible by 4
        self.img_size = 32
        self.patch_size = 4
        self.batch_size = 2
        self.num_heads = 2
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size ** 2
        self.T = self.num_patches + 1  # +1 for CLS token
        self.d_model = self.head_dim * self.num_heads
        self.rope = RoPE(self.head_dim, self.img_size, self.patch_size).to(device)

    def test_invalid_head_dim(self):
        with self.assertRaises(ValueError):
            RoPE(head_dim=30, img_size=32, patch_size=4)  # Not divisible by 4

    def test_compute_sine_cosine_shapes(self):
        sin_x, cos_x, sin_y, cos_y = self.rope.compute_sine_cosine()
        expected_shape = (1, 1, self.num_patches, self.head_dim // 4)
        self.assertEqual(sin_x.shape, expected_shape)
        self.assertEqual(cos_x.shape, expected_shape)
        self.assertEqual(sin_y.shape, expected_shape)
        self.assertEqual(cos_y.shape, expected_shape)

        # Check values within [-1, 1]
        self.assertTrue((sin_x >= -1).all() and (sin_x <= 1).all())
        self.assertTrue((cos_x >= -1).all() and (cos_x <= 1).all())

    def test_create_rotary_output_shape(self):
        B, T, num_heads, head_dim = self.batch_size, self.num_patches, self.num_heads, self.head_dim
        x = torch.randn(B, T, num_heads, head_dim).to(device)
        sin_x, cos_x, sin_y, cos_y = self.rope.compute_sine_cosine()
        rotated = self.rope.create_rotary(x, sin_x, cos_x, sin_y, cos_y)
        self.assertEqual(rotated.shape, x.shape)

    def test_create_rotary_value_difference(self):
        B, T, num_heads, head_dim = self.batch_size, self.num_patches, self.num_heads, self.head_dim
        x = torch.randn(B, T, num_heads, head_dim).to(device)
        sin_x, cos_x, sin_y, cos_y = self.rope.compute_sine_cosine()
        rotated = self.rope.create_rotary(x, sin_x, cos_x, sin_y, cos_y)
        # Rotated should differ from input (except maybe rare exact matches)
        self.assertFalse(torch.allclose(x, rotated))

    def test_apply_rope_to_tensor_preserves_shape(self):
        B, num_heads, T, head_dim = self.batch_size, self.num_heads, self.T, self.head_dim
        x = torch.randn(B, num_heads, T, head_dim).to(device)
        out = self.rope.apply_rope_to_tensor(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_preserves_shape(self):
        B, T, d_model = self.batch_size, self.T, self.d_model
        x = torch.randn(B, T, d_model).to(device)
        out = self.rope(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_raises_with_bad_d_model(self):
        B, T = self.batch_size, self.T
        d_model_bad = self.d_model + 1  # Not divisible by head_dim
        x = torch.randn(B, T, d_model_bad).to(device)
        with self.assertRaises(ValueError):
            self.rope(x)

    def test_cls_token_not_rotated(self):
        B, T, d_model = self.batch_size, self.T, self.d_model
        x = torch.randn(B, T, d_model).to(device)
        out = self.rope(x)
        # CLS token at position 0 should be identical (or very close) before and after rotation
        # Extract CLS token from input and output (shape [B, d_model])
        cls_in = x[:, 0, :]
        cls_out = out[:, 0, :]
        self.assertTrue(torch.allclose(cls_in, cls_out, atol=1e-5), "CLS token should not be rotated")

    def test_apply_rope_to_tensor_and_forward_consistency(self):
        # Create tensor [B, T, d_model]
        B, T, d_model = self.batch_size, self.T, self.d_model
        x = torch.randn(B, T, d_model).to(device)
        # apply_rope_to_tensor expects [B, num_heads, T, head_dim]
        x_reshaped = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        out1 = self.rope.apply_rope_to_tensor(x_reshaped)
        # Convert back to [B, T, d_model]
        out1 = out1.transpose(1, 2).reshape(B, T, d_model)

        out2 = self.rope(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6),
                        "apply_rope_to_tensor and forward outputs should match")

if __name__ == "__main__":
    unittest.main()
