"""Raw-input regressions; optional CPU checkpoint checks via NER_MODEL_DIR."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import text_sentences, highlight_spans


class RawTextTests(unittest.TestCase):
    def test_acronyms_and_titles_keep_context(self):
        text = 'Dr. Smith met U.S. and U.N. officials in New York.'
        sentences = text_sentences(text)
        self.assertEqual(len(sentences), 1)
        self.assertEqual([w[0] for w in sentences[0]],
                         ['Dr.', 'Smith', 'met', 'U.S.', 'and', 'U.N.', 'officials', 'in', 'New', 'York', '.'])

    def test_hyphens_apostrophes_and_offsets(self):
        text = "Jean-Claude O’Neill visited Baden-Wuerttemberg.\nAlice returned."
        sentences = text_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0][0][0], 'Jean-Claude')
        self.assertEqual(sentences[0][1][0], 'O’Neill')
        for sentence in sentences:
            for word, a, b in sentence:
                self.assertEqual(text[a:b], word)

    def test_invisible_characters_preserve_original_spans(self):
        text = '\ufeffAlice\u200bSmith works at Micro\u00adsoft. 👩\u200d💻'
        words = text_sentences(text)[0]
        self.assertEqual(words[0][0], 'AliceSmith')
        self.assertEqual(words[3][0], 'Microsoft')
        spans = highlight_spans(text, words, ['B-PER', 'O', 'O', 'B-ORG', 'O'])
        self.assertEqual([s['text'] for s in spans], ['Alice\u200bSmith', 'Micro\u00adsoft'])
        self.assertEqual(text_sentences('\ufeff\u200b\x00'), [])


@unittest.skipUnless(os.environ.get('NER_MODEL_DIR'), 'Set NER_MODEL_DIR for offline checkpoint checks')
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inference import Predictor
        cls.predictor = Predictor(os.environ['NER_MODEL_DIR'])

    def test_former_crashes_and_offsets(self):
        for text in ['Alice\u200bSmith works at Microsoft.', '👩\u200d💻 Alice works at Microsoft.',
                     'Micro\u00adsoft is in London.', '\ufeffAlice lives in London.', '\x00', '\ufe0f']:
            with self.subTest(text=text):
                result = self.predictor.analyze(text)
                self.assertEqual(result['text'], text)
                for span in result['entities']:
                    self.assertEqual(text[span['start']:span['end']], span['text'])

    def test_complete_entity_boundaries(self):
        cases = [
            ('U.S. President Barack Obama met U.N. officials in New York.',
             [('U.S.', 'LOC'), ('Barack Obama', 'PER'), ('U.N.', 'ORG'), ('New York', 'LOC')]),
            ('Jean-Claude Van Damme visited Baden-Wuerttemberg.',
             [('Jean-Claude Van Damme', 'PER'), ('Baden-Wuerttemberg', 'LOC')]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                result = self.predictor.analyze(text)
                self.assertEqual([(s['text'], s['label']) for s in result['entities']], expected)

    def test_blank_length_and_long_input(self):
        self.assertEqual(self.predictor.analyze('  ')['entities'], [])
        with self.assertRaises(ValueError):
            self.predictor.analyze('x' * 20001)
        text = 'Alice works at Microsoft in London and ' * 100
        result = self.predictor.analyze(text)
        self.assertTrue(any(s['end'] == text.rfind('London') + 6 for s in result['entities']))


if __name__ == '__main__':
    unittest.main()
