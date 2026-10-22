
import unittest

from PIL import Image

from src.predict import CLASS_NAMES, model, predict_image


class TestImageClassifier(unittest.TestCase):

    def test_class_names(self):
        self.assertEqual(CLASS_NAMES, ["Cat", "Dog"])

    def test_model_is_loaded(self):
        self.assertIsNotNone(model)

    def test_prediction_structure(self):
        image = Image.new("RGB", (224, 224), "white")
        result = predict_image(image)

        self.assertIn("label", result)
        self.assertIn("confidence", result)

    def test_prediction_label(self):
        image = Image.new("RGB", (224, 224), "white")
        result = predict_image(image)

        self.assertIn(result["label"], CLASS_NAMES)

    def test_confidence_range(self):
        image = Image.new("RGB", (224, 224), "white")
        result = predict_image(image)

        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
