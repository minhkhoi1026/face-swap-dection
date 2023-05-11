def test_step(self, batch, batch_idx):
        loss, accuracy, predictions = self.common_step(batch, batch_idx)     

        self.log_dict( {
            "loss": loss,
            "accuracy": accuracy
        } )
