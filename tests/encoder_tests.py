class DummyRoPE(torch.nn.Module):
    # Minimal RoPE stub with identity apply_rope_to_tensor for testing
    def apply_rope_to_tensor(self, x):
        return x

class TestTransformerEncoder(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.num_heads = 4
        self.query_groups = 2
        self.d_ffn = 64
        self.dropout = 0.0
        self.rope = DummyRoPE()
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            query_groups=self.query_groups,
            rope_module=self.rope,
            d_ffn=self.d_ffn,
            dropout=self.dropout
        )
        self.batch_size = 2
        self.seq_len = 5
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_output_shape(self):
        out = self.encoder(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_forward_runs_without_error(self):
        out = self.encoder(self.x)
        self.assertIsInstance(out, torch.Tensor)

    def test_dropout_effect(self):
        encoder_dropout = TransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            query_groups=self.query_groups,
            rope_module=self.rope,
            d_ffn=self.d_ffn,
            dropout=1.0 # Drop all
        )
        out = encoder_dropout(self.x)
        self.assertTrue(torch.allclose(out, self.x, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
