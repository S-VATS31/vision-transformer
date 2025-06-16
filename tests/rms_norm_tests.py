class TestRMSNorm(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.batch_size = 4
        self.seq_len = 10
        self.eps = 1e-7
        self.model = RMSNorm(self.d_model, eps=self.eps).to(device)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model).to(device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, x.shape, "Output shape should match input shape")

    def test_output_not_equal_input(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model).to(device)
        out = self.model(x)
        self.assertFalse(torch.allclose(out, x), "Output should differ from input after normalization")

    def test_rms_normalized_close_to_one(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model).to(device)
        out = self.model(x)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        # Since output is scaled by gamma, normalize by gamma mean
        gamma_mean = self.model.gamma.mean().item()
        rms_normalized = rms / gamma_mean
        self.assertTrue(torch.allclose(rms_normalized, torch.ones_like(rms), atol=1e-5),
                        "RMS of normalized output should be close to 1 (ignoring gamma scale)")

    def test_gamma_requires_grad(self):
        self.assertTrue(self.model.gamma.requires_grad, "Gamma parameter should require gradients")

    def test_backward_pass(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True).to(device)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, "Gradients should flow back to input")

    def test_no_nan_or_inf(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model).to(device)
        out = self.model(x)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaNs")
        self.assertFalse(torch.isinf(out).any(), "Output contains Infs")

if __name__ == "__main__":
    unittest.main()
