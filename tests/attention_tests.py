class TestGroupedQueryAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.num_heads = 8
        self.query_groups = 4
        self.rope = DummyRoPE()
        self.attn = GroupedQueryAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            query_groups=self.query_groups,
            rope_module=self.rope,
        )
        self.batch_size = 2
        self.seq_len = 10

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.attn(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_empty_sequence(self):
        x = torch.randn(self.batch_size, 0, self.d_model)
        output = self.attn(x)
        self.assertEqual(output.shape, (self.batch_size, 0, self.d_model))

    def test_wrong_input_dim(self):
        x = torch.randn(self.batch_size, self.seq_len)
        with self.assertRaises(ValueError):
            self.attn(x)

    def test_wrong_feature_dim(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model + 1)
        with self.assertRaises(ValueError):
            self.attn(x)

    def test_divisible_d_model_num_heads(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(d_model=65, num_heads=8, query_groups=4, rope_module=self.rope)

    def test_divisible_num_heads_query_groups(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(d_model=64, num_heads=7, query_groups=4, rope_module=self.rope)

    def test_q_k_dim_mismatch(self):
        # Temporarily patch k_proj to produce wrong head_dim for keys
        self.attn.k_proj = torch.nn.Linear(self.d_model, self.d_model)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        with self.assertRaises(ValueError):
            self.attn(x)

    def test_softmax_v_dim_mismatch(self):
        # Patch v_proj to output wrong dim
        self.attn.v_proj = torch.nn.Linear(self.d_model, self.d_model)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        with self.assertRaises(ValueError):
            self.attn(x)

if __name__ == "__main__":
    unittest.main()
