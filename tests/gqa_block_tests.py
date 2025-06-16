class DummyRoPE:
    def apply_rope_to_tensor(self, x):
        return x

class DummyRMSNorm(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    def forward(self, x):
        return x 

class DummyGroupedQueryAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, query_groups, rope_module):
        super().__init__()
        self.d_model = d_model
    def forward(self, x):
        return x + 1

class TestGQABlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 32
        self.num_heads = 4
        self.query_groups = 2
        self.rope = DummyRoPE()
        
        # Patch RMSNorm and GroupedQueryAttention with dummy versions
        self.block = GQABlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            query_groups=self.query_groups,
            rope_module=self.rope,
            dropout=0.0 # Disable dropout to test residuals
        )
        self.block.rms_norm = DummyRMSNorm(self.d_model)
        self.block.attn = DummyGroupedQueryAttention(self.d_model, self.num_heads, self.query_groups, self.rope)

        self.batch_size = 2
        self.seq_len = 5

    def test_output_shape_and_residual(self):
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        output = self.block(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.allclose(output, torch.ones_like(x)))

    def test_dropout_effect(self):
        # Now enable dropout and check output changes (some values should be zeroed out)
        self.block.dropout = torch.nn.Dropout(p=1.0) # Drop all
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        output = self.block(x)
        # With full dropout, attn output after dropout is zero
        self.assertTrue(torch.allclose(output, torch.zeros_like(x)))

    def test_input_wrong_dim_raises(self):
        x = torch.randn(self.batch_size, self.seq_len)
        with self.assertRaises(Exception):
            self.block(x)

if __name__ == "__main__":
    unittest.main()
