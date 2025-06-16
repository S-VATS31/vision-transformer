class TestPatchEmbeddingsCLS(unittest.TestCase):

    def setUp(self):
        self.img_size = 32
        self.patch_size = 8
        self.C_in = 3
        self.d_model = 64
        self.batch_size = 2

        self.model = PatchEmbeddings(
            img_size=self.img_size,
            patch_size=self.patch_size,
            C_in=self.C_in,
            d_model=self.d_model
        ).to(device)

    def test_cls_token_requires_grad(self):
        self.assertTrue(self.model.cls_token.requires_grad, "CLS token should be learnable")

    def test_cls_token_shape(self):
        self.assertEqual(
            self.model.cls_token.shape,
            (1, 1, self.d_model),
            f"Expected CLS token shape (1, 1, {self.d_model}), got {self.model.cls_token.shape}"
        )

    def test_cls_token_prepend(self):
        x = torch.randn(self.batch_size, self.C_in, self.img_size, self.img_size).to(device)
        with torch.no_grad():
            output = self.model(x)

        # The first token in the sequence should be the CLS token
        cls_token_output = output[:, 0, :]
        patch_tokens_output = output[:, 1:, :]

        self.assertEqual(
            cls_token_output.shape,
            (self.batch_size, self.d_model),
            "CLS token output shape mismatch"
        )

        # Check number of tokens
        expected_tokens = (self.img_size // self.patch_size) ** 2 + 1
        self.assertEqual(output.shape[1], expected_tokens)

    def test_cls_token_learns(self):
        # Save original CLS token value
        original_cls = self.model.cls_token.clone().detach()

        # Simple optimizer to update weights
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        x = torch.randn(self.batch_size, self.C_in, self.img_size, self.img_size).to(device)
        target = torch.randn(self.batch_size, ((self.img_size // self.patch_size) ** 2) + 1, self.d_model).to(device)

        # Forward + Backward + Step
        output = self.model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        updated_cls = self.model.cls_token.detach()

        # Check that the CLS token changed after optimization
        self.assertFalse(torch.equal(original_cls, updated_cls), "CLS token did not update after gradient step")
