class TestFFNBlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.d_ffn = 64
        self.dropout_prob = 0.0 # Disable dropout for shape test
        self.ffn_block = FFNBlock(self.d_model, self.d_ffn, dropout=self.dropout_prob)

        self.batch_size = 3
        self.seq_len = 7
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_output_shape(self):
        out = self.ffn_block(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_dropout_effect(self):
        ffn_block_dropout = FFNBlock(self.d_model, self.d_ffn, dropout=1.0)
        out = ffn_block_dropout(self.x)
        self.assertTrue(torch.allclose(out, self.x))

    def test_forward_runs_without_error(self):
        out = self.ffn_block(self.x)
        self.assertIsInstance(out, torch.Tensor)

if __name__ == "__main__":
    unittest.main()
